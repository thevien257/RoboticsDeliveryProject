"""Mô hình robot vi sai hai bánh dùng động cơ N20 có encoder và hai dãy cảm biến line.

Phần "vật lý" (pose thật, bán kính bánh thật, trượt) chỉ dùng để mô phỏng.
Phần "robot biết" (odometry, đọc cảm biến) chỉ dùng thông số danh định + số xung encoder.
"""
import math
import numpy as np


class RobotParams:
    def __init__(self, **kw):
        self.r = 0.0215            # bán kính bánh danh định [m] (bánh 43 mm)
        self.b = 0.120             # khoảng cách hai bánh [m]
        self.cpr = 1400            # xung encoder / vòng bánh (7 PPR x 4 cạnh x hộp số 1:50)
        self.w_max = 31.4          # tốc độ bánh tối đa [rad/s] (~300 vòng/phút)
        self.tau = 0.10            # hằng số thời gian động cơ không tải [s] (N20 + driver PWM)
        self.tau_load = 0.15       # hằng số thời gian khi chở hàng [s]
        self.n_sens = 8            # số cảm biến mỗi dãy
        self.pitch = 0.012         # khoảng cách giữa hai cảm biến [m]
        self.x_front = 0.100       # vị trí dãy trước tính từ trục bánh [m]
        self.x_rear = 0.030        # vị trí dãy sau [m]
        self.sens_noise = 0.03     # nhiễu đọc cảm biến mỗi mẫu ADC (đã chuẩn hoá)
        self.tof_half_fov = 30.0   # nửa góc nhìn của cụm cảm biến khoảng cách [độ]
        self.adc_avg = 4           # số mẫu ADC lấy trung bình mỗi chu kỳ 10 ms (giảm nhiễu √4 = 2 lần, không thêm trễ)
        # sai lệch thật so với danh định (robot không biết)
        self.r_err_l = 0.0
        self.r_err_r = 0.0
        self.b_err = 0.0
        self.slip = 0.0            # độ lệch chuẩn trượt ngẫu nhiên (tỉ lệ) mỗi bước
        self.motor_bias = 0.0      # động cơ phải yếu hơn động cơ trái một tỉ lệ (nhiễu loạn kéo dài)
        self.latency = 0.02        # trễ từ đọc cảm biến tới khi lệnh tác động lên động cơ [s] (đọc ADC, tính toán, PWM)
        self.gyro = False          # có IMU (MPU6050) đo tốc độ quay
        self.gyro_bias = 0.012     # độ lệch 0 thật của gyro [rad/s] (~0,7°/s) – robot không biết
        self.gyro_noise = 0.003    # nhiễu trắng gyro [rad/s] mỗi mẫu 10 ms
        self.loaded = False
        for k, v in kw.items():
            setattr(self, k, v)

    @property
    def offsets(self):
        n = self.n_sens
        return (np.arange(n) - (n - 1) / 2) * self.pitch


class Robot:
    def __init__(self, world, pose, params=None, seed=0, dt=0.002):
        self.w = world
        self.p = params or RobotParams()
        self.rng = np.random.default_rng(seed)
        self.dt = dt
        self.x, self.y, self.th = pose
        self.wl = self.wr = 0.0                 # tốc độ bánh thật [rad/s]
        self.ang_l = self.ang_r = 0.0           # góc quay bánh thật tích luỹ
        self.cnt_l = self.cnt_r = 0             # xung encoder
        self.t = 0.0
        self.dist_true = 0.0
        # odometry (robot tự tính)
        self.ox, self.oy, self.oth = pose
        self._last_cnt = (0, 0)
        self.b_est = self.p.b          # khoảng cách bánh dùng cho odometry (tự hiệu chuẩn khi quét)
        self.eps_est = 0.0             # ước lượng chênh lệch bán kính (r_r - r_l)/r (tự hiệu chuẩn trên cạnh đã biết)
        self.gbias_est = 0.0           # độ lệch 0 gyro ước lượng (học khi đứng yên)
        self._th_prev = pose[2]
        self._still = 0
        self._gbuf = []
        self._q = []
        self.collisions = 0
        self._in_coll = False
        self.range_max = 1.2           # cảm biến khoảng cách ToF phía trước (VL53L0X)

    # ---------------------------------------------------------------- động lực học
    def step(self, wl_cmd, wr_cmd):
        p = self.p
        if p.latency > 0:                       # hàng đợi lệnh: lệnh hiện tại chỉ tác động sau `latency` giây
            nd = int(round(p.latency / self.dt))
            self._q.append((wl_cmd, wr_cmd))
            if len(self._q) > nd:
                wl_cmd, wr_cmd = self._q.pop(0)
            else:
                wl_cmd, wr_cmd = self.wl, self.wr
        wm = p.w_max
        wl_cmd = min(max(float(wl_cmd), -wm), wm)
        wr_cmd = min(max(float(wr_cmd), -wm), wm) * (1 - p.motor_bias)
        tau = p.tau_load if p.loaded else p.tau
        a = self.dt / tau
        self.wl += a * (wl_cmd - self.wl)
        self.wr += a * (wr_cmd - self.wr)
        rl = p.r * (1 + p.r_err_l)
        rr = p.r * (1 + p.r_err_r)
        slip = p.slip
        for c, rad, sg in self.w.slips:
            if math.hypot(self.x - c[0], self.y - c[1]) < rad:
                slip = max(slip, sg)
        sl = 1 + (self.rng.normal(0, slip) if slip else 0)
        sr = 1 + (self.rng.normal(0, slip) if slip else 0)
        vl = rl * self.wl * sl
        vr = rr * self.wr * sr
        bt = p.b * (1 + p.b_err)
        v = (vl + vr) / 2
        om = (vr - vl) / bt
        dt = self.dt
        hm = self.th + om * dt / 2
        self.x += v * math.cos(hm) * dt
        self.y += v * math.sin(hm) * dt
        self.th = (self.th + om * dt + math.pi) % (2 * math.pi) - math.pi
        self.dist_true += abs(v) * dt
        # encoder đo góc quay của bánh (không thấy trượt)
        self.ang_l += self.wl * dt
        self.ang_r += self.wr * dt
        k = p.cpr / (2 * math.pi)
        self.cnt_l = math.floor(self.ang_l * k)
        self.cnt_r = math.floor(self.ang_r * k)
        self.t += dt
        self.w.t = self.t
        for ob in self.w.obstacles:
            ob.update(self.t, self.x, self.y, self.th, v)
        return v, om

    def step_n(self, wl_cmd, wr_cmd, n):
        """n bước con liên tiếp với cùng lệnh – kết quả giống gọi step() n lần, nhanh hơn (biến cục bộ)."""
        p = self.p
        if self.w.slips or p.slip:
            for _ in range(n):
                self.step(wl_cmd, wr_cmd)
            return
        dt = self.dt
        nd = int(round(p.latency / dt)) if p.latency > 0 else 0
        q = self._q
        wm = p.w_max
        cl = min(max(float(wl_cmd), -wm), wm)
        cr = min(max(float(wr_cmd), -wm), wm)
        tau = p.tau_load if p.loaded else p.tau
        a = dt / tau
        rl, rr = p.r * (1 + p.r_err_l), p.r * (1 + p.r_err_r)
        bt = p.b * (1 + p.b_err)
        kb = 1 - p.motor_bias
        k = p.cpr / (2 * math.pi)
        x, y, th, wl, wr = self.x, self.y, self.th, self.wl, self.wr
        al, ar, dist, t = self.ang_l, self.ang_r, self.dist_true, self.t
        cos, sin, pi2 = math.cos, math.sin, 2 * math.pi
        for _ in range(n):
            if nd:
                q.append((cl, cr))
                if len(q) > nd:
                    ul, ur = q.pop(0)
                    ul = min(max(ul, -wm), wm)
                    ur = min(max(ur, -wm), wm)
                else:
                    ul, ur = wl, wr
            else:
                ul, ur = cl, cr
            wl += a * (ul - wl)
            wr += a * (ur * kb - wr)
            vl, vr = rl * wl, rr * wr
            v = (vl + vr) / 2
            om = (vr - vl) / bt
            hm = th + om * dt / 2
            x += v * cos(hm) * dt
            y += v * sin(hm) * dt
            th = (th + om * dt + math.pi) % pi2 - math.pi
            dist += abs(v) * dt
            al += wl * dt
            ar += wr * dt
            t += dt
        self.x, self.y, self.th, self.wl, self.wr = x, y, th, wl, wr
        self.ang_l, self.ang_r, self.dist_true, self.t = al, ar, dist, t
        self.cnt_l, self.cnt_r = math.floor(al * k), math.floor(ar * k)
        self.w.t = t
        for ob in self.w.obstacles:
            ob.update(t, x, y, th, (rl * wl + rr * wr) / 2)

    # ---------------------------------------------------------------- odometry (thông số danh định)
    def odom_update(self):
        p = self.p
        dl = (self.cnt_l - self._last_cnt[0]) * 2 * np.pi / p.cpr * p.r * (1 - self.eps_est / 2)
        dr = (self.cnt_r - self._last_cnt[1]) * 2 * np.pi / p.cpr * p.r * (1 + self.eps_est / 2)
        self.om_meas = (dr - dl) / self.b_est / 0.01     # tốc độ quay đo từ encoder (chu kỳ 10 ms)
        self._last_cnt = (self.cnt_l, self.cnt_r)
        self.last_inc = (self.cnt_l - self._last_cnt[0], self.cnt_r - self._last_cnt[1])
        ds = (dl + dr) / 2
        dth = (dr - dl) / self.b_est
        if p.gyro:
            dt = 0.01
            true_dth = (self.th - self._th_prev + np.pi) % (2 * np.pi) - np.pi
            g = true_dth / dt + p.gyro_bias + self.rng.normal(0, p.gyro_noise)
            # chỉ học độ lệch 0 khi encoder đứng yên LIÊN TỤC >= 0,3 s (tránh nhầm vòng quay chậm cuối pha dừng)
            self._still = self._still + 1 if (dl == 0 and dr == 0) else 0
            if self._still >= 30:
                self._gbuf.append(g)
                if len(self._gbuf) > 3000:          # cửa sổ trượt: theo kịp trôi nhiệt chậm
                    self._gbuf.pop(0)
                self.gbias_est = float(np.mean(self._gbuf))
            dth = (g - self.gbias_est) * dt
            self.last_gyro = g
        self._th_prev = self.th
        self.ox += ds * np.cos(self.oth + dth / 2)
        self.oy += ds * np.sin(self.oth + dth / 2)
        self.oth = (self.oth + dth + np.pi) % (2 * np.pi) - np.pi
        return ds, dth

    def wheel_dist(self):
        p = self.p
        return (self.cnt_l + self.cnt_r) / 2 * 2 * np.pi / p.cpr * p.r

    # ---------------------------------------------------------------- cảm biến khoảng cách + va chạm
    def range(self):
        """Khoảng cách tới vật cản gần nhất phía trước. Hai cảm biến ToF VL53L0X ở mũi xe, nghiêng ±15°, mỗi chiếc
        trường nhìn ±15° -> phủ ±30° (ở khoảng dừng 12 cm vùng nhìn rộng ≈ 14 cm, bằng thân xe)."""
        fx, fy = self.x + 0.11 * np.cos(self.th), self.y + 0.11 * np.sin(self.th)
        best = self.range_max
        hit = False
        for ob in self.w.obstacles:
            c = ob.pos(self.t)
            dx, dy = c[0] - fx, c[1] - fy
            d = np.hypot(dx, dy)
            if np.hypot(c[0] - self.x, c[1] - self.y) < ob.r + 0.075:
                hit = True
            ang = abs((np.arctan2(dy, dx) - self.th + np.pi) % (2 * np.pi) - np.pi)
            if ang < np.deg2rad(self.p.tof_half_fov) + np.arcsin(min(1, ob.r / max(d, 1e-6))):
                best = min(best, max(d - ob.r, 0.0))
        if hit and not self._in_coll:
            self.collisions += 1
        self._in_coll = hit
        return best

    # ---------------------------------------------------------------- cảm biến
    def sensor_points(self, x_off):
        c, s = math.cos(self.th), math.sin(self.th)
        ys = self.p.offsets
        px = self.x + x_off * c - ys * s
        py = self.y + x_off * s + ys * c
        return np.column_stack([px, py])

    def read(self):
        """Đọc hai dãy cảm biến (giá trị 0..1)."""
        p = self.p
        if getattr(self, '_sens_key', None) != (p.n_sens, p.pitch, p.x_front, p.x_rear):
            ys = p.offsets
            self._sens_key = (p.n_sens, p.pitch, p.x_front, p.x_rear)
            self._sx = np.concatenate([np.full(len(ys), p.x_front), np.full(len(ys), p.x_rear)])
            self._sy = np.concatenate([ys, ys])
        c, s = math.cos(self.th), math.sin(self.th)
        pts = np.empty((len(self._sx), 2))
        pts[:, 0] = self.x + self._sx * c - self._sy * s
        pts[:, 1] = self.y + self._sx * s + self._sy * c
        r = self.w.reflect(pts, self.rng, p.sens_noise, avg=p.adc_avg)
        n = p.n_sens
        return r[:n], r[n:]
