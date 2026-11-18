"""Mô phỏng bám line vector hoá: N robot (N bộ tham số / điều kiện) chạy song song trên cùng một đường thử.

Cùng mô hình và cùng trình tự với TrackRun (follow.py) – đọc hai dãy cảm biến, ước lượng e_y, e_θ, xử lý mất line,
bộ điều khiển, quy đổi tốc độ bánh, hàng đợi trễ, động cơ bậc 1, động học vi sai – nhưng mỗi bước là phép toán trên
mảng N phần tử, nên chỉnh định hàng trăm bộ tham số mất vài giây thay vì vài phút. Kết quả được đối chiếu với
TrackRun trong experiments/campaign.py (--kiem-tra-batch).
"""
import numpy as np
from .robot import RobotParams
from .control import dlqr_gain, bryson

TS = 0.01


class BatchPID:
    """PID dạng song song, sai lệch tại tâm trục bánh, D theo động học e_y' = v·sin(e_θ) (như PID mặc định)."""

    def __init__(self, Kp, Ki, Kd, i_lim=0.05, i_band=0.005):
        self.Kp, self.Ki, self.Kd = (np.asarray(x, float) for x in (Kp, Ki, Kd))
        self.i_lim, self.i_band = i_lim, i_band
        self.i = np.zeros_like(self.Kp)
        self.load_gain = np.ones_like(self.Kp)     # τ/τ_tải khi chở hàng (BatchTrack đặt)

    def __call__(self, m, ey, eth, yf, only_f, v):
        e = np.where(only_f, yf, ey)[m]
        upd = np.abs(e) < self.i_band if self.i_band is not None else np.ones_like(e, bool)
        self.i[m] = np.where(upd, np.clip(self.i[m] + e * TS, -self.i_lim, self.i_lim), self.i[m])
        d = v[m] * np.sin(eth[m])
        g = np.minimum(np.maximum(v[m], 0.03) / 0.3, 1.0) * self.load_gain[m]   # lập lịch theo tốc độ và tải như PID
        return g * (self.Kp[m] * e + self.Ki[m] * self.i[m]) + self.Kd[m] * d


class BatchLQR:
    """LQR/LQI mô hình có động cơ [d, ψ, ω(, ∫d)], ω lấy từ mô hình động cơ chạy song song (như LQR(model='dyn'))."""

    def __init__(self, rows, v, tau=None, z_lim=0.02):
        """rows: danh sách dict e_y_max_mm, e_theta_max_deg, omega_max, z_max_mm_s (0 = không tích phân)."""
        tau = tau or RobotParams().tau
        self.tau = tau
        N = len(rows)
        self.K = np.zeros((N, 4))
        self.integ = np.zeros(N, bool)
        cache = {}
        for k, (p, vk) in enumerate(zip(rows, v)):
            z = p.get('z_max_mm_s', 0) or 0
            key = (p['e_y_max_mm'], p['e_theta_max_deg'], p['omega_max'], z, round(max(vk, 0.05), 2))
            if key not in cache:
                Q, R = bryson(d_max=p['e_y_max_mm'] / 1000, psi_max=np.radians(p['e_theta_max_deg']), u_max=p['omega_max'],
                              model='dyn', integral=z > 0, **({'z_max': z / 1000} if z > 0 else {}))
                cache[key] = dlqr_gain(key[-1], TS, Q, R, 'dyn', tau, z > 0)[0]
            K = cache[key]
            self.K[k, :len(K)] = K
            self.integ[k] = z > 0
        self.z_lim = z_lim
        self.om = np.zeros(N)
        self.z = np.zeros(N)
        self.u_prev = np.zeros(N)

    def __call__(self, m, ey, eth, yf, only_f, v):
        a = np.exp(-TS / self.tau)
        self.om[m] = a * self.om[m] + (1 - a) * self.u_prev[m]
        d, psi = -ey[m], -eth[m]
        zi = np.where(self.integ[m], np.clip(self.z[m] + d * TS, -self.z_lim, self.z_lim), 0.0)
        self.z[m] = zi
        K = self.K[m]
        u = -(K[:, 0] * d + K[:, 1] * psi + K[:, 2] * self.om[m] + K[:, 3] * zi)
        self.u_prev[m] = u
        return u


class BatchTrack:
    def __init__(self, world, ctrl, v_ref, seed=0, pose=None, sens_noise=0.03, motor_bias=0.0, loaded=False,
                 t_max=40.0, params=None, dt=0.002, max_lost=20):
        p = params or RobotParams()
        self.p, self.w, self.ctrl = p, world, ctrl
        N = len(v_ref)
        self.N = N
        a = lambda x, dt_=float: np.broadcast_to(np.asarray(x, dt_), (N,)).copy()
        self.v_ref, self.noise, self.bias, self.loaded = a(v_ref), a(sens_noise), a(motor_bias), a(loaded, bool)
        self.t_max = t_max
        self.max_lost = max_lost                # mất line quá số lần này: dừng sớm (đã chắc chắn không đạt)
        poly = world.polylines[0]
        if pose is None:
            d0 = poly[1] - poly[0]
            pose = (poly[0][0], poly[0][1], np.arctan2(d0[1], d0[0]))
        pose = np.broadcast_to(np.asarray(pose, float), (N, 3))
        self.x, self.y, self.th = pose[:, 0].copy(), pose[:, 1].copy(), pose[:, 2].copy()
        self.wl, self.wr = np.zeros(N), np.zeros(N)
        self.end = poly[-1]
        self.dt = dt
        self.n_sub = int(round(TS / dt))
        self.nd = int(round(p.latency / dt))
        self.qbuf = np.zeros((max(self.nd, 1), N, 2))
        self.qn = 0
        self.rng = np.random.default_rng(np.asarray(seed, np.int64).ravel().tolist() + [N])
        ys = p.offsets
        self.sx = np.concatenate([np.full(len(ys), p.x_front), np.full(len(ys), p.x_rear)])
        self.sy = np.concatenate([ys, ys])
        self.ys = ys
        self.thr = 0.35
        self.active = np.ones(N, bool)
        self.finished = np.zeros(N, bool)
        self.failed = np.zeros(N, bool)
        self.lost = np.zeros(N, bool)
        self.lost_t = np.zeros(N)
        self.lost_events = np.zeros(N, int)
        self.last_side = np.zeros(N)
        self.t = 0.0
        self.log_e, self.log_om, self.log_x, self.log_y, self.log_ey, self.log_sat = [], [], [], [], [], []
        if isinstance(ctrl, BatchPID):
            ctrl.load_gain = np.where(self.loaded, p.tau / p.tau_load, 1.0)

    # -------------------------------------------------------------- cảm biến và ước lượng (như LineEstimator)
    def _arrays(self):
        c, s = np.cos(self.th)[:, None], np.sin(self.th)[:, None]
        pts = np.empty((self.N, len(self.sx), 2))
        pts[..., 0] = self.x[:, None] + self.sx[None] * c - self.sy[None] * s
        pts[..., 1] = self.y[:, None] + self.sx[None] * s + self.sy[None] * c
        d = self.w.dist_field(pts)
        r = 1.0 / (1.0 + np.exp((d - self.w.width / 2) / 0.0025))
        r = np.clip(r + self.rng.normal(size=r.shape) * (self.noise[:, None] / np.sqrt(max(self.p.adc_avg, 1))), 0, 1)
        n = len(self.ys)
        out = []
        for rr in (r[:, :n], r[:, n:]):
            act = rr > self.thr
            cnt = act.sum(1)
            wgt = np.clip(rr - self.thr * 0.5, 0, None) * act
            sw = wgt.sum(1)
            y = np.where(cnt > 0, (wgt * self.ys[None]).sum(1) / np.where(sw > 0, sw, 1), np.nan)
            out.append((y, cnt > 0))
        return out

    def _errors(self):
        (yf, hf), (yr, hr) = self._arrays()
        p = self.p
        both = hf & hr
        eth = np.zeros(self.N)
        ey = np.zeros(self.N)
        eb = np.arctan2(yf - yr, p.x_front - p.x_rear)
        eth = np.where(both, eb, eth)
        ey = np.where(both, yr - p.x_rear * np.tan(np.where(both, eb, 0)), ey)
        of = hf & ~hr
        eth = np.where(of, np.arctan2(np.where(of, yf, 0), p.x_front), eth)
        orr = hr & ~hf
        ey = np.where(orr, yr, ey)
        seen = hf | hr
        return ey, eth, np.where(hf, yf, np.nan), of, seen

    # -------------------------------------------------------------- một chu kỳ 10 ms
    def tick(self):
        m0 = self.active
        if not m0.any():
            return False
        ey, eth, yf, only_f, seen = self._errors()
        near_end = np.hypot(self.x - self.end[0], self.y - self.end[1]) < 0.12
        v = np.zeros(self.N)
        om = np.zeros(self.N)
        # mất line
        ml = m0 & ~seen
        fin = ml & near_end
        self.finished |= fin
        self.active &= ~fin
        ml &= ~fin
        self.lost_events += (ml & ~self.lost).astype(int)
        self.lost = np.where(ml, True, np.where(m0 & seen, False, self.lost))
        self.lost_t = np.where(ml, self.lost_t + TS, np.where(m0 & seen, 0.0, self.lost_t))
        fail = (ml & (self.lost_t > 2.0)) | (m0 & (self.lost_events > self.max_lost))
        self.failed |= fail
        self.active &= ~fail
        ml &= ~fail
        om[ml] = 4.0 * np.where(self.last_side[ml] >= 0, 1.0, -1.0)
        v[ml] = 0.05
        # bám line
        mf = m0 & seen
        self.last_side = np.where(mf, np.where(np.isnan(yf), ey, yf), self.last_side)
        v[mf] = self.v_ref[mf]
        if mf.any():
            om[mf] = self.ctrl(mf, ey, eth, yf, only_f, v)
        run = self.active.copy()
        if not run.any():
            return False
        wl, wr = self._wheel_cmd(v, om)
        for _ in range(self.n_sub):
            self._step(wl, wr, run)
        self.t += TS
        e = np.where(run, self.w.dist_field(np.c_[self.x, self.y]), np.nan)
        self.log_e.append(e)
        self.log_om.append(np.where(run, om, np.nan))
        self.log_x.append(self.x.copy())
        self.log_y.append(self.y.copy())
        self.log_ey.append(np.where(run & seen, ey, np.nan))
        self.log_sat.append(np.maximum(np.abs(self.wl), np.abs(self.wr)) > 0.98 * self.p.w_max)
        if self.t >= self.t_max - 1e-9:
            self.active[:] = False
        return self.active.any()

    def _wheel_cmd(self, v, om):
        p = self.p
        wl = (v - om * p.b / 2) / p.r
        wr = (v + om * p.b / 2) / p.r
        m = np.maximum(np.abs(wl), np.abs(wr))
        sat = m > p.w_max
        dv = (m - p.w_max) * p.r
        v2 = np.maximum(v - dv, 0.0)
        wl2 = (v2 - om * p.b / 2) / p.r
        wr2 = (v2 + om * p.b / 2) / p.r
        m2 = np.maximum(np.abs(wl2), np.abs(wr2))
        sc = np.where(m2 > p.w_max, p.w_max / np.maximum(m2, 1e-12), 1.0)
        return np.where(sat, wl2 * sc, wl), np.where(sat, wr2 * sc, wr)

    def _step(self, wl_c, wr_c, run):
        p = self.p
        if self.nd > 0:
            k = self.qn % self.nd
            if self.qn >= self.nd:
                ul, ur = self.qbuf[k, :, 0].copy(), self.qbuf[k, :, 1].copy()
            else:
                ul, ur = self.wl.copy(), self.wr.copy()
            self.qbuf[k, :, 0], self.qbuf[k, :, 1] = wl_c, wr_c
            self.qn += 1
        else:
            ul, ur = wl_c, wr_c
        ul = np.clip(ul, -p.w_max, p.w_max)
        ur = np.clip(ur, -p.w_max, p.w_max) * (1 - self.bias)
        tau = np.where(self.loaded, p.tau_load, p.tau)
        a = self.dt / tau
        wl = self.wl + a * (ul - self.wl)
        wr = self.wr + a * (ur - self.wr)
        vl, vr = p.r * (1 + p.r_err_l) * wl, p.r * (1 + p.r_err_r) * wr
        v = (vl + vr) / 2
        om = (vr - vl) / (p.b * (1 + p.b_err))
        hm = self.th + om * self.dt / 2
        self.x = np.where(run, self.x + v * np.cos(hm) * self.dt, self.x)
        self.y = np.where(run, self.y + v * np.sin(hm) * self.dt, self.y)
        self.th = np.where(run, (self.th + om * self.dt + np.pi) % (2 * np.pi) - np.pi, self.th)
        self.wl = np.where(run, wl, self.wl)
        self.wr = np.where(run, wr, self.wr)

    def run(self):
        while self.tick():
            pass
        return self

    # -------------------------------------------------------------- chỉ tiêu (như TrackRun.metrics)
    def metrics(self):
        E = np.array(self.log_e)
        OM = np.array(self.log_om)
        SAT = np.array(self.log_sat)
        out = []
        k0 = int(0.3 / TS)
        for j in range(self.N):
            e = E[:, j]
            e = e[~np.isnan(e)]
            ee = e[k0:] if len(e) > k0 else e
            if len(ee) == 0:
                ee = np.array([np.nan])
            o = OM[:, j]
            o = o[~np.isnan(o)]
            n = len(e)
            out.append(dict(rms=float(np.sqrt(np.nanmean(ee ** 2))) * 1000, emax=float(np.nanmax(ee)) * 1000,
                            p95=float(np.nanpercentile(ee, 95)) * 1000, time=n * TS, lost=int(self.lost_events[j]),
                            finished=bool(self.finished[j] and not self.failed[j]),
                            dom=float(np.sqrt(np.mean(np.diff(o) ** 2))) if len(o) > 2 else 0.0,
                            om_rms=float(np.sqrt(np.mean(o ** 2))) if len(o) else 0.0,
                            sat=float(np.mean(SAT[:n, j])) * 100 if n else 0.0))
        return out
