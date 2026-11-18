"""Tác tử điều khiển robot vận chuyển bám line.

Chức năng: bám line (PID/LQR), xử lý vật cản và đứt line, phát hiện giao lộ/đầu mút, quét nhánh tại nút mới
(kèm tự hiệu chuẩn khoảng cách bánh), nắn hướng tại nút đã biết (kèm tự hiệu chuẩn chênh lệch bán kính bánh),
khám phá DFS, nhận dạng nút nhiều giả thuyết, tối ưu pose graph khi khép vòng, giao nhiều đơn với lập lịch động.

Logic chạy dạng generator: mỗi `tick()` là một chu kỳ điều khiển Ts = 10 ms, dùng chung cho GUI và thực nghiệm.
"""
import numpy as np
from .robot import Robot, RobotParams
from .control import LineEstimator, wheel_cmd, PID, PID_DEFAULT
from .mapping import TopoMap, Associator, match_signature, wrap

TS = 0.01


class Agent:
    def __init__(self, world, pose, ctrl=None, assoc='mht', params=None, seed=0, v_ref=0.25,
                 v_slow=0.10, w_scan=2.5, scan_known=False, posegraph=True, scan_calibrate=True, ed_calibrate=True):
        self.world = world
        self.p = params or RobotParams()
        self.rob = Robot(world, pose, self.p, seed=seed)
        self.est = LineEstimator(self.p)
        self.ctrl = ctrl or PID(**PID_DEFAULT)
        self.map = TopoMap()
        self.assoc = Associator(assoc)
        self.v_ref, self.v_slow, self.w_scan = v_ref, v_slow, w_scan
        self.scan_known, self.posegraph = scan_known, posegraph
        self.scan_calibrate, self.ed_calibrate = scan_calibrate, ed_calibrate
        self.n_sub = int(round(TS / self.rob.dt))
        self.events, self.traj = [], []
        self.cur = None
        self.home = None
        self.mode = 'idle'
        self.task = None
        self.done = False
        self.failed = False
        self.stats = dict(wrong_merge=0, duplicate=0, active_checks=0, relocalize=0, nodes_visited=0,
                          false_events=0, scans=0, rot_total=0.0, obstacle_stops=0, wait_time=0.0,
                          blocked_edges=0, replans=0, gap_crossings=0, lost=0)
        self.anchor_s = 0.0
        self.rot_since = 0.0           # góc đã quay kể từ lần neo (rad)
        self.d_stop, self.d_slow, self.t_block = 0.12, 0.35, 6.0
        self.orders = []               # hàng đợi đơn (lấy, trả) – GUI có thể thêm khi đang chạy
        self.stations = {}             # tên trạm -> id nút
        self.scheduler = None
        self.delivery = []
        self.cmd = (0.0, 0.0)
        self.ey_log = []
        self.trace = []                # mỗi chu kỳ 10 ms: dữ liệu để xuất và phân tích (xem sim/export.py)
        self._e = np.nan
        self.brake_est = {False: 0.0, True: 0.0}   # quãng trượt khi phanh, học riêng lúc không tải / có tải
        self.P_th = 0.0                # phương sai hướng ước lượng [rad^2] (dung hợp Kalman khi nắn hướng)
        self.R_th = 0.04 ** 2          # phương sai phép đo hướng tại nút (hướng đi vào/ra đo khi bám line)

    # ---------------------------------------------------------------- tiện ích
    def log(self, msg):
        self.events.append((round(self.rob.t, 2), msg))

    def truth_label(self, tol=0.12):
        best, bd = None, tol
        for k, P in self.world.nodes.items():
            d = np.hypot(self.rob.x - P[0], self.rob.y - P[1])
            if d < bd:
                best, bd = k, d
        return best

    def _apply(self, v, om):
        wl, wr = wheel_cmd(v, om, self.p)
        self.cmd = (v, om)
        self.rob.step_n(wl, wr, self.n_sub)
        ds, dth = self.rob.odom_update()
        if self.p.gyro:
            self.P_th += (0.002 ** 2) * TS                       # gyro: trôi chậm theo thời gian
        else:
            self.P_th += (0.1 ** 2) * abs(ds) + (0.02 ** 2) * abs(dth)   # encoder: theo quãng đường và góc quay
        self.rot_since += abs(dth) if v == 0 else 0.0
        if v == 0:
            self.stats['rot_total'] += abs(dth)
        r = self.rob
        self.traj.append((r.t, r.x, r.y, r.ox, r.oy))
        self.trace.append((r.t, self.mode, r.x, r.y, r.th, r.ox, r.oy, r.oth, v, om, r.wl, r.wr, self._e))
        self._e = np.nan

    def s(self):
        return self.rob.wheel_dist()

    def tick(self):
        if self.task is None:
            return False
        try:
            next(self.task)
            return True
        except StopIteration:
            self.task = None
            self.done = True
            return False

    def sigma(self):
        """Độ bất định vị trí: tăng theo quãng đường và theo góc đã quay kể từ lần neo gần nhất."""
        s = self.s() - self.anchor_s
        sig_th2 = (0.004 ** 2) * self.rot_since + (0.05 ** 2) * s       # rad^2
        return float(np.sqrt(0.02 ** 2 + (0.03 ** 2) * s + (s ** 2) * sig_th2))

    # ---------------------------------------------------------------- hành vi cơ bản
    def stop(self, t=0.25):
        for _ in range(int(t / TS)):
            self._apply(0.0, 0.0)
            yield

    def follow(self, ignore=0.10, v=None):
        """Bám line đến khi gặp giao lộ / đầu mút / bị chặn. Trả về (loại sự kiện, quãng đường encoder)."""
        v_set = v
        self.mode = 'follow'
        s0 = self.s()
        self.ctrl.reset()
        self.est.reset_track()
        lost_t, last_side, wait = 0.0, 0.0, 0.0
        eth_prev = 1.0
        coast = None
        while True:
            v = v_set or self.v_ref
            # --- vật cản phía trước (cảm biến ToF)
            rng = self.rob.range()
            if rng < self.d_stop:
                if wait == 0.0:
                    self.stats['obstacle_stops'] += 1
                    self.log(f'Vật cản cách {rng*100:.0f} cm: dừng chờ')
                self.mode = 'chờ vật cản'
                wait += TS
                self.stats['wait_time'] += TS
                self._apply(0.0, 0.0)
                yield
                if wait > self.t_block:
                    self.log(f'Vật cản không rời đi sau {self.t_block:.0f} s -> coi đoạn line bị chặn')
                    return 'blocked', self.s()
                continue
            if wait > 0:
                self.log(f'Đường thông sau {wait:.1f} s, chạy tiếp')
                wait = 0.0
                self.mode = 'follow'
            if rng < self.d_slow:
                v = v * float(np.clip((rng - self.d_stop) / (self.d_slow - self.d_stop), 0.3, 1.0))
            rf, rr = self.rob.read()
            ey, eth, yf, nf, nr = self.est.errors(rf, rr)
            trav = self.s() - s0
            if trav > ignore and 0 < nf <= 3:
                self.est.calibrate(nf)
            thr = self.est.junction_threshold()
            # giao lộ: đủ ngưỡng 2n̄+1 mắt; hoặc thiếu đúng 1 mắt nhưng robot đang thẳng hàng với line (nhánh rẽ cong rời
            # giao lộ chỉ phủ ít mắt) – khi lệch hướng lớn, line cắt dãy xiên cũng phủ nhiều mắt nên không tính
            if trav > ignore and (nf >= thr or (nf == thr - 1 and abs(eth_prev) < 0.3)):
                self._event_s = self.s()
                return 'junction', self.s()
            if eth is not None:
                eth_prev = eth
            if nf == 0 and nr > 0:
                yr = self.est.array(rr)[0]
                if abs(yr) < 0.015 and abs(eth) < 0.35 and trav > ignore:
                    # nghi đầu mút: đi thẳng chậm tới đúng vị trí tâm nút (trừ quãng phanh đã học) và kiểm tra trên
                    # đường đi; nếu dãy trước thấy lại line thì là line đứt. Gộp kiểm tra với vào nút để kịp phanh.
                    s_c = self.s()
                    target = s_c + self.p.x_front - self._half_w()
                    key = bool(self.p.loaded)
                    seen = False
                    while self.s() < target - self.brake_est[key]:
                        self._apply(self.v_slow, 0.0)
                        yield
                        rf, rr = self.rob.read()
                        if self.est.array(rf)[1] > 0:
                            seen = True
                            break
                    if not seen:
                        self.mode = 'vào nút'
                        yield from self.stop()
                        self._learn_brake(key, self.s() - target)
                        self._event_s = s_c
                        self._at_node = True
                        return 'deadend', s_c
                    self.stats['gap_crossings'] += 1
                    continue
            if ey is None:
                # mất cả hai dãy: nếu trước đó đang thẳng hàng thì đi thẳng tiếp tối đa 6 cm (line mòn/đứt)
                if coast is None and abs(last_side) < 0.012:
                    coast = self.s()
                if coast is not None and self.s() - coast < 0.06:
                    self._apply(self.v_slow, 0.0)
                    yield
                    continue
                if lost_t == 0.0:
                    self.stats['lost'] += 1
                lost_t += TS
                if lost_t > 2.5:
                    return 'lost', self.s()
                self.mode = 'tìm line'
                self._apply(0.05, 4.0 * (1 if last_side >= 0 else -1))
                yield
                continue
            if coast is not None:
                self.stats['gap_crossings'] += 1
                coast = None
            lost_t = 0.0
            self.mode = 'follow'
            last_side = yf if yf is not None else ey
            if hasattr(self.ctrl, 'load_gain'):
                self.ctrl.load_gain = self.p.tau / self.p.tau_load if self.p.loaded else 1.0
            om = self.ctrl(ey, eth, yf, v, getattr(self.rob, 'om_meas', None))
            self.ey_log.append(ey)
            self._e = ey
            self._apply(v, om)
            yield

    def advance_to(self, s_target):
        """Tiến chậm đến quãng đường encoder s_target; tự học quãng trượt khi phanh để bù."""
        self.mode = 'vào nút'
        key = bool(self.p.loaded)
        brake = self.brake_est[key]
        while self.s() < s_target - brake:
            self._apply(self.v_slow, 0.0)
            yield
        yield from self.stop()
        self._learn_brake(key, self.s() - s_target)

    def _learn_brake(self, key, over):
        self.brake_est[key] = float(np.clip(self.brake_est[key] + 0.5 * over, 0.0, 0.03))

    def _half_w(self):
        return (self.est.normal_count or 1.5) * self.p.pitch / 2

    def scan(self):
        """Quay tại chỗ đến khi cắt lại nhánh đầu tiên; ghi các góc nhánh; tự hiệu chuẩn khoảng cách bánh."""
        self.mode = 'quét nhánh'
        self.stats['scans'] += 1
        th0 = self.rob.oth
        acc, prev_th = 0.0, th0
        hits, armed = [], False
        while acc < 2 * np.pi * 2.3 and not (hits and acc > hits[0] + 2 * np.pi * 1.15):
            self._apply(0.0, self.w_scan)
            yield
            th = self.rob.oth
            acc += abs(wrap(th - prev_th))
            prev_th = th
            rf, _ = self.rob.read()
            y, n = self.est.array(rf)
            if y is not None and n < self.est.junction_threshold():
                if y > 0.003:
                    armed = True
                elif y <= 0 and armed:
                    hits.append(acc)
                    armed = False
            else:
                armed = False
        yield from self.stop(0.2)
        ratio, rs = 1.0, []
        for i, h0 in enumerate(hits):
            rep = [h for h in hits[i + 1:] if 2 * np.pi * 0.88 < h - h0 < 2 * np.pi * 1.12]
            if rep:
                rs.append((min(rep, key=lambda h: abs(h - h0 - 2 * np.pi)) - h0) / (2 * np.pi))
        # có gyro thì hướng do gyro đo (chính xác hơn) -> không hiệu chỉnh hướng theo line khi quét
        if rs and self.scan_calibrate and not self.p.gyro:
            ratio = float(np.median(rs))
            if abs(ratio - 1) < 0.15:
                self.rob.b_est *= 1 + 0.8 * (ratio - 1)
        self.rob.oth = wrap(self.rob.oth + acc * (1 / ratio - 1))
        angs = sorted([wrap(th0 + h / ratio) for h in hits])
        out = []
        for h in angs:
            if out and abs(wrap(h - out[-1][-1])) < np.deg2rad(25):
                out[-1].append(h)
            else:
                out.append([h])
        if len(out) > 1 and abs(wrap(out[0][0] - out[-1][-1])) < np.deg2rad(25):
            out[0] = out[-1] + out[0]
            out.pop()
        return [float(np.angle(np.mean(np.exp(1j * np.array(g))))) for g in out]

    def turn_to(self, target):
        """Quay về phía góc mục tiêu, dừng khi dãy trước bắt được line gần góc đó."""
        self.mode = 'rẽ'
        d = wrap(target - self.rob.oth)
        sgn = 1 if d >= 0 else -1
        acc, prev_th, prev_y = 0.0, self.rob.oth, None
        while acc < 2 * np.pi + 0.5:
            self._apply(0.0, sgn * self.w_scan * 0.8)
            yield
            acc += abs(wrap(self.rob.oth - prev_th))
            prev_th = self.rob.oth
            rf, _ = self.rob.read()
            y, n = self.est.array(rf)
            near = abs(wrap(target - self.rob.oth)) < np.deg2rad(30)
            if y is not None and near:
                if prev_y is not None and (prev_y * y <= 0 or abs(y) < 0.004):
                    break
                prev_y = y
            else:
                prev_y = None
        yield from self.stop(0.1)

    # ---------------------------------------------------------------- tới nút
    def enter_node(self, kind):
        half_w = self._half_w()
        if getattr(self, '_at_node', False):      # đầu mút: đã dừng đúng tâm nút trong lúc kiểm tra
            self._at_node = False
            return
            yield
        if kind == 'junction':
            yield from self.advance_to(self._event_s + self.p.x_front + half_w)
        else:
            yield from self.advance_to(self._event_s + self.p.x_front - half_w)

    def anchor(self, node, heading=None, offset=None, hard=False):
        """Neo theo nút đã nhận dạng: đặt vị trí = vị trí nút; nắn hướng bằng cập nhật Kalman 1 chiều
        (K = P/(P+R)): có gyro thì P nhỏ nên hầu như giữ hướng gyro; chỉ encoder thì P lớn nên nắn mạnh."""
        self.rob.ox, self.rob.oy = node.pos
        h = heading if heading is not None else (self.rob.oth - offset if offset is not None else None)
        if h is not None:
            K = 1.0 if hard else self.P_th / (self.P_th + self.R_th)
            self.rob.oth = wrap(self.rob.oth + K * wrap(h - self.rob.oth))
            self.P_th = (1 - K) * self.P_th
        self.anchor_s = self.s()
        self.rot_since = 0.0

    def mark_arrival(self):
        """Lưu tư thế odometry khi vừa tới tâm nút (trước khi quét/neo)."""
        self._arr = (self.rob.ox, self.rob.oy, self.rob.oth)

    def record_edge(self, prev, nid, pin):
        """Ràng buộc SE(2) của cạnh vừa đi: độ dời trong hệ robot lúc rời nút và góc quay."""
        dep = getattr(self, '_dep', None)
        if prev is None or dep is None or not self.posegraph or dep[0] != prev[0]:
            return
        x0, y0, t0 = dep[2], dep[3], dep[4]
        x1, y1, t1 = self._arr
        c, s_ = np.cos(-t0), np.sin(-t0)
        rel = (c * (x1 - x0) - s_ * (y1 - y0), s_ * (x1 - x0) + c * (y1 - y0))
        phi = wrap(t0 - dep[5])
        self.map.add_meas(prev[0], prev[1], nid, pin, rel, wrap(t1 - t0), float(np.hypot(*rel)), phi)

    def node_after_edge(self, prev, L, kind):
        while True:
            res = yield from self.process_node(prev, L, kind)
            if res != 'continue':
                return res
            s0 = self.s()
            kind, s_end = yield from self.follow()
            L += s_end - s0

    def process_node(self, prev, L, kind):
        tm = self.map
        dep_pos = self._dep_pos
        yield from self.enter_node(kind)
        self.mark_arrival()
        truth = self.truth_label()
        self.stats['nodes_visited'] += 1
        link = tm.nodes[prev[0]].links[prev[1]] if prev is not None else None
        # ---- cạnh đã biết: nút dự đoán, KHÔNG quét; nắn hướng theo cổng vào; tự hiệu chuẩn bán kính bánh
        if link is not None and not self.scan_known:
            Y = tm.nodes[link[0]]
            Lk = tm.nodes[prev[0]].lengths[prev[1]]
            if abs(L - Lk) < 0.25 * Lk + 0.05:
                pin = link[1]
                if Y.arr[pin] is None and Y.dep[pin] is not None:
                    # khứ hồi trên cùng một cạnh: hướng về = hướng lúc rời + π (đúng cả với line cong)
                    h = wrap(Y.th + Y.dep[pin] + np.pi)
                    e = wrap(self.rob.oth - h)
                    Ltot = L + Lk
                    if self.ed_calibrate and Ltot > 0.3 and abs(e) < 0.6:
                        self.rob.eps_est = float(np.clip(self.rob.eps_est - 0.5 * e * self.rob.b_est / Ltot, -0.05, 0.05))
                    Y.arr[pin] = wrap(Y.dep[pin] + np.pi)
                elif Y.arr[pin] is None:
                    Y.arr[pin] = wrap(self.rob.oth - Y.th)
                    h, e = self.rob.oth, 0.0
                else:
                    h = wrap(Y.th + Y.arr[pin])
                    e = wrap(self.rob.oth - h)
                    if self.ed_calibrate and not self.p.gyro and L > 0.3 and abs(e) < 0.5:
                        self.rob.eps_est = float(np.clip(self.rob.eps_est - 0.3 * e * self.rob.b_est / L, -0.05, 0.05))
                self.record_edge(prev, Y.id, pin)
                Kg = self.P_th / (self.P_th + self.R_th)
                self.anchor(Y, heading=h)
                Y.n_visits += 1
                if truth is not None and Y.truth != truth:
                    self.stats['wrong_merge'] += 1
                self.log(f'Đến nút {Y.id} theo cạnh đã biết (lệch đo {np.degrees(e):+.1f}°, nắn {np.degrees(Kg * e):+.1f}°)')
                return Y.id
            self.stats['relocalize'] += 1
            self.log(f'Chiều dài cạnh đo {L:.2f} m khác bản đồ {Lk:.2f} m -> quét để định vị lại')
        arr = wrap(self.rob.oth + np.pi)          # hướng cổng vào: lấy TRƯỚC khi quay quét
        angles = yield from self.scan()
        if not angles or min(abs(wrap(a - arr)) for a in angles) > np.deg2rad(35):
            angles.append(arr)
        pos = np.array([self.rob.ox, self.rob.oy])
        if len(angles) == 2:
            self.stats['false_events'] += 1
            self.log('Điểm gấp khúc (bậc 2), không phải nút -> đi tiếp')
            other = max(angles, key=lambda a: abs(wrap(a - arr)))
            yield from self.turn_to(other)
            return 'continue'
        if link is not None:   # chế độ quét mọi nút hoặc định vị lại
            Y = tm.nodes[link[0]]
            ok, off, asg = match_signature(Y.gports(), angles)
            if ok:
                self.record_edge(prev, Y.id, link[1])
                self.anchor(Y, offset=off)
                Y.n_visits += 1
                if truth is not None and Y.truth != truth:
                    self.stats['wrong_merge'] += 1
                return Y.id
        cands = self.assoc.candidates(tm, pos, angles, arr, self.s() - self.anchor_s,
                                      exclude=(prev[0],) if prev else (), sigma=self.sigma())
        best = cands[0]
        if self.assoc.method == 'mht' and best[1] < self.assoc.p_accept and len(cands) > 1:
            res = yield from self.active_check(prev, L, pos, angles, arr, cands, truth, dep_pos)
            return res
        return self.commit(best, prev, L, pos, angles, arr, truth, dep_pos)

    def commit(self, best, prev, L, pos, angles, arr, truth, dep_pos):
        tm = self.map
        nid, prob, off, assign = best
        j_in = int(np.argmin([abs(wrap(b - arr)) for b in angles]))
        if nid is None:
            if truth is not None and any(n.truth == truth for n in tm.nodes.values() if not n.tentative):
                self.stats['duplicate'] += 1
            N = tm.add_node(pos, angles, truth)
            N.arr[j_in] = wrap(self._arr[2] - N.th)
            if prev is not None:
                tm.connect(prev[0], prev[1], N.id, j_in, L)
            self.record_edge(prev, N.id, j_in)
            self.log(f'Tạo nút mới {N.id} (bậc {N.degree})')
            return N.id
        X = tm.nodes[nid]
        if truth is not None and X.truth != truth:
            self.stats['wrong_merge'] += 1
        pin = assign[j_in]
        if X.arr[pin] is None:
            X.arr[pin] = wrap(self._arr[2] - off - X.th)
        if prev is not None:
            if X.links[pin] is None:
                tm.connect(prev[0], prev[1], X.id, pin, L)
        self.record_edge(prev, X.id, pin)
        X.n_visits += 1
        self.loop_closure(X, off)
        self.log(f'Khép vòng: nhận lại nút {X.id} (P={prob:.2f})')
        return X.id

    def loop_closure(self, X, off):
        # robot đang ở X: hướng robot trong hệ cục bộ của X giữ nguyên qua tối ưu
        local = self.rob.oth - off - X.th
        if self.posegraph:
            sh = self.map.optimize(anchor=self.home)
            if sh > 0.005:
                self.log(f'Tối ưu pose graph SE(2): dịch nút tối đa {sh*100:.1f} cm')
        self.anchor(X, heading=X.th + local, hard=True)

    def active_check(self, prev, L, pos, angles, arr, cands, truth, dep_pos):
        tm = self.map
        self.stats['active_checks'] += 1
        T = tm.add_node(pos, angles, truth)
        T.tentative = True
        j_in = int(np.argmin([abs(wrap(b - arr)) for b in angles]))
        T.arr[j_in] = wrap(self._arr[2] - T.th)
        if prev is not None:
            tm.connect(prev[0], prev[1], T.id, j_in, L)
        self.record_edge(prev, T.id, j_in)
        olds = [c for c in cands if c[0] is not None and c[1] > 0.02]
        self.log(f'Mơ hồ: {len(olds)} nút cũ có thể khớp -> kiểm chứng chủ động')
        for nid, prob, off, assign in olds:
            X = tm.nodes[nid]
            for jt, px in assign.items():
                if jt == j_in or X.links[px] is None:
                    continue
                Y = tm.nodes[X.links[px][0]]
                L_xy = X.lengths[px]
                yield from self.turn_to(angles[jt])
                self._dep = (T.id, jt, self.rob.ox, self.rob.oy, self.rob.oth, T.th)
                self._dep_pos = np.array([self.rob.ox, self.rob.oy])
                s0 = self.s()
                kind, s_end = yield from self.follow()
                L2 = s_end - s0
                yield from self.enter_node(kind)
                self.mark_arrival()
                arr2 = wrap(self.rob.oth + np.pi)
                ang2 = yield from self.scan()
                pos2 = np.array([self.rob.ox, self.rob.oy])
                truth2 = self.truth_label()
                ok, off2, asg2 = match_signature(Y.gports(), ang2)
                if ok and abs(L2 - L_xy) < 0.2 * L_xy + 0.05:
                    j2 = int(np.argmin([abs(wrap(b - arr2)) for b in ang2]))
                    self.record_edge((T.id, jt), Y.id, asg2[j2])
                    tm.merge(T.id, X.id, assign)
                    if truth is not None and X.truth != truth:
                        self.stats['wrong_merge'] += 1
                    self.loop_closure(Y, off2)
                    self.log(f'Kiểm chứng xác nhận: nút tạm = nút {X.id}; đang ở nút {Y.id}')
                    return Y.id
                T.tentative = False
                if truth is not None and any(n.truth == truth and n.id != T.id for n in tm.nodes.values() if not n.tentative):
                    self.stats['duplicate'] += 1
                self.log(f'Kiểm chứng bác bỏ nút {X.id}: nút tạm {T.id} là nút mới')
                cands2 = self.assoc.candidates(tm, pos2, ang2, arr2, self.s() - self.anchor_s, exclude=(T.id,), sigma=self.sigma())
                return self.commit(cands2[0], (T.id, jt), L2, pos2, ang2, arr2, truth2, None)
        T.tentative = False
        if truth is not None and any(n.truth == truth and n.id != T.id for n in tm.nodes.values() if not n.tentative):
            self.stats['duplicate'] += 1
        self.log(f'Không có nhánh kiểm chứng: giữ nút {T.id} là nút mới')
        return T.id

    # ---------------------------------------------------------------- di chuyển trên bản đồ
    def go_port(self, node_id, port):
        N = self.map.nodes[node_id]
        yield from self.turn_to(N.gports()[port])
        if N.dep[port] is None:
            N.dep[port] = wrap(self.rob.oth - N.th)
        self._dep = (node_id, port, self.rob.ox, self.rob.oy, self.rob.oth, N.th)
        self._dep_pos = np.array([self.rob.ox, self.rob.oy])
        s0 = self.s()
        kind, s_end = yield from self.follow()
        return kind, s_end - s0

    def go_back(self, node_id, port):
        """Quay đầu tại chỗ, bám line về lại nút vừa rời (khi đoạn phía trước bị chặn)."""
        self.log('Quay đầu về nút trước')
        yield from self.turn_to(wrap(self.rob.oth + np.pi))
        kind, _ = yield from self.follow(ignore=0.03)
        yield from self.enter_node(kind)
        N = self.map.nodes[node_id]
        if N.arr[port] is None:
            N.arr[port] = wrap(self.rob.oth - N.th)
            self.anchor(N)
        else:
            self.anchor(N, heading=wrap(N.th + N.arr[port]))
        return node_id

    def explore(self, t_limit=900.0):
        self.mode = 'khám phá'
        if self.p.gyro:
            self.log('Đứng yên 2 s để hiệu chuẩn độ lệch 0 của gyro')
            yield from self.stop(2.0)
        self.log('Bắt đầu học map: tìm line, tự học độ rộng line')
        self._dep_pos = np.array([self.rob.ox, self.rob.oy])
        s0 = self.s()
        kind, s_end = yield from self.follow()
        cur = yield from self.node_after_edge(None, s_end - s0, kind)
        self.home = cur
        last_prog, n_links = self.rob.t, 0
        while self.rob.t < t_limit and not self.failed:
            # watchdog: phát hiện kẹt/lỗi sớm thay vì chạy hết giới hạn thời gian
            nl = sum(l is not None for n in self.map.nodes.values() for l in n.links)
            if nl > n_links:
                n_links, last_prog = nl, self.rob.t
            if self.rob.t - last_prog > 150 or len(self.map.nodes) > 80:
                self.fail_reason = 'không có tiến triển 150 s' if self.rob.t - last_prog > 150 else 'số nút tăng bất thường'
                self.log('Watchdog dừng học map: ' + self.fail_reason)
                self.failed = True
                break
            N = self.map.nodes[cur]
            un = N.unexplored()
            if un:
                port = un[0]
                kind, L = yield from self.go_port(cur, port)
                if kind == 'blocked':
                    N.blocked.add(port)
                    self.stats['blocked_edges'] += 1
                    cur = yield from self.go_back(cur, port)
                    continue
                if kind == 'lost':
                    self.log('Mất line, dừng khám phá')
                    self.failed = True
                    break
                cur = yield from self.node_after_edge((cur, port), L, kind)
                continue
            dist, _ = self.map.dijkstra(cur)
            targets = [(d, v) for v, d in dist.items() if self.map.nodes[v].unexplored()]
            if not targets:
                self.log('Học map xong: không còn nhánh chưa khám phá')
                break
            ok = yield from self.travel(cur, min(targets)[1])
            cur = self.cur
            if not ok and self.failed:
                break
        if self.posegraph and cur in self.map.nodes:
            C = self.map.nodes[cur]
            local = self.rob.oth - C.th
            self.map.optimize(anchor=self.home)
            self.anchor(C, heading=C.th + local, hard=True)
        self.cur = cur
        self.explore_time = self.rob.t
        self.explore_dist = self.rob.dist_true
        yield from self.stop(0.2)

    def travel(self, src, dst):
        """Đi theo tuyến Dijkstra; nếu bị chặn thì đánh dấu cạnh, quay về và lập lại tuyến."""
        cur = src
        for _ in range(30):
            if cur == dst:
                self.cur = cur
                return True
            seq, _ = self.map.path(cur, dst)
            if seq is None:
                self.log(f'Không còn đường từ nút {cur} tới nút {dst}')
                self.cur = cur
                return False
            u, p = seq[0]
            kind, L = yield from self.go_port(u, p)
            if kind == 'blocked':
                N = self.map.nodes[u]
                N.blocked.add(p)
                nb = N.links[p]
                if nb is not None:
                    self.map.nodes[nb[0]].blocked.add(nb[1])
                self.stats['blocked_edges'] += 1
                self.stats['replans'] += 1
                cur = yield from self.go_back(u, p)
                self.log('Lập lại tuyến tránh đoạn bị chặn')
                continue
            if kind == 'lost':
                self.failed = True
                self.cur = cur
                return False
            cur = yield from self.node_after_edge((u, p), L, kind)
        self.cur = cur
        return cur == dst

    def deliver(self, batch, method='dp', Q=1):
        """Giao một lượt đơn (chỉ chạy khi người dùng nhấn Giao hàng).
        batch: [(trạm lấy, trạm trả)]; method: 'fifo' / 'greedy' / 'dp'; Q: sức chứa (số kiện mỗi chuyến).
        Kế hoạch là dãy điểm dừng; các thao tác liên tiếp tại cùng một trạm được làm trong một lần dừng."""
        from .planning import dist_matrix, plan_stops, _start_dist
        self.mode = 'giao hàng'
        self.batch = list(batch)
        names = sorted(self.stations)
        ids = [self.stations[n] for n in names]
        idx = {n: i for i, n in enumerate(names)}
        D = dist_matrix(self.map, ids)
        d0 = _start_dist(self.map, self.cur, ids)
        od = [(idx[a], idx[b]) for a, b in self.batch]
        seq, cost = plan_stops(D, od, d0, Q, method)
        stops = []
        for st, act, k in seq:
            if stops and stops[-1][0] == st:
                stops[-1][1].append((act, k))
            else:
                stops.append((st, [(act, k)]))
        self.plan_text = ' → '.join(f"{names[s]}({', '.join(f'{a} đơn {k + 1}' for a, k in acts)})" for s, acts in stops)
        self.plan_cost = cost
        self.plan_inputs = dict(D=D, orders=od, d0=d0, Q=Q, method=method, names=names)   # để so sánh phương pháp khi xuất báo cáo
        self.log(f'Lượt giao {len(self.batch)} đơn, sức chứa {Q}, kế hoạch {cost:.2f} m: ' + self.plan_text)
        self.batch_status = ['chờ lấy'] * len(self.batch)
        load = 0
        for st_i, acts in stops:
            st = names[st_i]
            ok = yield from self.travel(self.cur, self.stations[st])
            if not ok:
                self.log(f'Không tới được trạm {st}, bỏ qua thao tác tại trạm này')
                continue
            yield from self.stop(0.3)
            tn = self.map.nodes[self.stations[st]].truth
            tx, ty = self.world.nodes.get(tn, (np.nan, np.nan))
            err = float(np.hypot(self.rob.x - tx, self.rob.y - ty)) * 1000
            ch, sh = np.cos(self.rob.th), np.sin(self.rob.th)
            err_along = float((self.rob.x - tx) * ch + (self.rob.y - ty) * sh) * 1000     # dương: vượt quá tâm nút
            err_lat = float(-(self.rob.x - tx) * sh + (self.rob.y - ty) * ch) * 1000
            correct = self.truth_label() == tn
            for act, k in acts:
                self.mode = f'servo {act} hàng'
                for _ in range(int(0.8 / TS)):
                    self._apply(0.0, 0.0)
                    yield
                load += 1 if act == 'lấy' else -1
                self.p.loaded = load > 0
                self.batch_status[k] = 'đang chở' if act == 'lấy' else 'đã giao'
                self.delivery.append(dict(station=st, action=act, order=k, t=self.rob.t, err_mm=err, err_along=err_along,
                                          err_lat=err_lat, correct=correct, load=load))
                self.log(f'{act.capitalize()} hàng đơn {k + 1} tại {st} (đang chở {load} kiện): sai số dừng {err:.0f} mm, đúng trạm = {correct}')
        self.mode = 'xong lượt giao'
        self.log('Hoàn thành lượt giao')
