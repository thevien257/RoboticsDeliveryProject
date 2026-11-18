"""Xử lý cảm biến line và hai bộ điều khiển bám line: PID, LQR."""
import numpy as np
from scipy.linalg import solve_discrete_are


# ======================================================================== cảm biến -> sai lệch
class LineEstimator:
    """Ước lượng vị trí line trên từng dãy bằng trọng tâm có trọng số, tự hiệu chuẩn độ rộng line."""

    def __init__(self, params, thr=0.35):
        self.p = params
        self.thr = thr
        self.ys = params.offsets
        self.width_hist = []            # số mắt đen khi chạy trên đoạn line bình thường
        self.normal_count = None
        self.prev = {'f': None, 'r': None}

    def reset_track(self):
        self.prev = {'f': None, 'r': None}

    def array_track(self, r, key):
        """Như array() nhưng khi dãy thấy nhiều cụm đen tách rời (giao lộ, nhánh rẽ) thì chỉ lấy trọng tâm của cụm gần
        vị trí line ở chu kỳ trước nhất (hoặc gần tâm dãy nếu vừa bắt đầu) – robot không bị nhánh khác kéo lệch.
        Số mắt đen n vẫn đếm trên cả dãy để phát hiện giao lộ."""
        act = r > self.thr
        n = int(act.sum())
        if n == 0:
            return None, 0
        w = np.clip(r - self.thr * 0.5, 0, None) * act
        idx = np.flatnonzero(act)
        cuts = np.flatnonzero(np.diff(idx) > 1) + 1
        groups = np.split(idx, cuts)
        if len(groups) == 1:
            y = float((w * self.ys).sum() / w.sum())
        else:
            ref = self.prev[key] if self.prev[key] is not None else 0.0
            cents = [float((w[g] * self.ys[g]).sum() / w[g].sum()) for g in groups]
            y = min(cents, key=lambda c: abs(c - ref))
        self.prev[key] = y
        return y, n

    def array(self, r):
        act = r > self.thr
        n = int(act.sum())
        if n == 0:
            return None, 0
        w = np.clip(r - self.thr * 0.5, 0, None) * act
        y = float((w * self.ys).sum() / w.sum())
        return y, n

    def calibrate(self, n):
        """Học số mắt đen điển hình của line (không nhập trước bề rộng line)."""
        if n > 0 and len(self.width_hist) < 300:
            self.width_hist.append(n)
            if len(self.width_hist) >= 40:
                self.normal_count = float(np.median(self.width_hist))

    def junction_threshold(self):
        nc = self.normal_count if self.normal_count else 2.0
        return max(4, int(np.ceil(2 * nc + 1)))

    def errors(self, rf, rr):
        """Trả về (e_y, e_th, y_f, n_f, n_r): line nằm bên trái là dương."""
        p = self.p
        yf, nf = self.array_track(rf, 'f')
        yr, nr = self.array_track(rr, 'r')
        if yf is not None and yr is not None:
            # line trong hệ robot: y(x) ≈ d + ψ·x + κ·x²/2. Đường thẳng qua hai dãy cắt trục bánh tại d − κ·x_f·x_r/2:
            # trên đoạn cong e_y đo bị lệch dây cung (≈ 5 mm khi R = 0,3 m). Không bù bằng κ = ω/v vì ω chứa cả
            # phần quay sửa sai của chính bộ điều khiển -> hồi tiếp dương (đã thử, làm xấu bám line).
            eth = np.arctan2(yf - yr, p.x_front - p.x_rear)
            ey = yr - p.x_rear * np.tan(eth)
        elif yf is not None:
            eth, ey = np.arctan2(yf, p.x_front), 0.0
        elif yr is not None:
            eth, ey = 0.0, yr
        else:
            eth = ey = None
        return ey, eth, yf, nf, nr


# ======================================================================== bộ điều khiển
# tham số mặc định trên GUI = PID*, LQR* (LQI) do bộ thí nghiệm E0 chọn (experiments/campaign.py, results/campaign/tuned.json)
PID_DEFAULT = dict(Kp=120.0, Ki=80.0, Kd=30.0)
LQR_DEFAULT = dict(d_max_mm=1.0, psi_max_deg=10.0, u_max=3.0, z_max_mm_s=8.0)


class PID:
    """PID bám line, dạng song song u = Kp·e + Ki·∫e + Kd·de/dt (đạo hàm có lọc thông thấp).

    src='center': e = sai lệch ngang tại tâm trục bánh, ước lượng từ hai dãy cảm biến (mặc định).
        Mô hình d'' = v·ω là khâu tích phân kép giống con lắc: chỉ có P thì luôn dao động,
        Kp càng lớn càng vọt lố; Kd tạo giảm chấn; Ki triệt sai lệch tĩnh (động cơ lệch, đường cong).
    d_src='kin' : de/dt = v·sin(e_θ) với e_θ đo bằng hai dãy (mặc định); d_src='diff': sai phân có lọc.
    src='front' : e = vị trí line ở dãy trước (cách trục x_f). Vì y_f = d + x_f·ψ nên bộ P trên y_f
        đã ngầm chứa thành phần D (Kd ≈ Kp·x_f/v); dùng để so sánh trong thực nghiệm.
    """
    name = 'PID'

    def __init__(self, Kp=40.0, Ki=0.0, Kd=1.2, Ts=0.01, N=20.0, i_lim=0.05, src='center', d_src='kin', i_band=0.005, v0=0.3):
        self.Kp, self.Ki, self.Kd, self.Ts, self.N, self.i_lim, self.src = Kp, Ki, Kd, Ts, N, i_lim, src
        self.i_band = i_band          # tích phân có điều kiện: chỉ tích phân khi |e| < i_band (chống windup khi quá độ)
        self.v0 = v0                  # lập lịch theo tốc độ: dưới v0 thì K_p, K_i nhân v/v0 -> ζ không giảm khi robot chạy chậm
        self.load_gain = 1.0          # lập lịch theo tải: khi chở hàng K_p, K_i nhân τ/τ_tải (robot biết càng nâng đang nâng)
        self.d_src = d_src
        self.reset()

    def reset(self):
        self.i = 0.0
        self.e_prev = None
        self.d = 0.0

    def error(self, ey, yf):
        if self.src == 'front':
            return yf if yf is not None else (ey if ey is not None else 0.0)
        if ey is None:
            return 0.0
        if ey == 0.0 and yf is not None:    # chỉ dãy trước thấy line (LineEstimator trả ey = 0): dùng y_f
            return yf
        return ey

    def __call__(self, ey, eth, yf, v, om_meas=None):
        e = self.error(ey, yf)
        if self.i_band is None or abs(e) < self.i_band:
            self.i = min(max(self.i + e * self.Ts, -self.i_lim), self.i_lim)
        if self.e_prev is None:
            self.e_prev = e
        if self.d_src == 'kin' and self.src == 'center' and eth is not None:
            # de/dt theo động học: e_y' = v·sin(e_θ), e_θ đo trực tiếp từ hai dãy -> không trễ lọc, ít nhiễu lượng tử
            self.d = v * np.sin(eth)
        else:
            # sai phân có lọc thông thấp bậc 1 (tần số cắt N rad/s)
            a = self.N * self.Ts / (1 + self.N * self.Ts)
            self.d = (1 - a) * self.d + a * (e - self.e_prev) / self.Ts
        self.e_prev = e
        g = (min(max(v, 0.03) / self.v0, 1.0) if self.v0 else 1.0) * self.load_gain   # giảm hệ số khi chạy chậm / chở hàng
        return g * (self.Kp * e + self.Ki * self.i) + self.Kd * self.d


def dlqr_gain(v, Ts, Q, R, model='kin', tau=0.10, integral=False, nd=0):
    """Hệ số LQR rời rạc cho mô hình sai lệch bám line quanh vận tốc v.

    model='kin' : x = [d, psi],          d' = v psi,  psi' = omega
    model='dyn' : x = [d, psi, omega],   thêm động cơ bậc 1: omega' = (u - omega)/tau
    integral    : thêm z = ∫d (LQI) để triệt sai lệch tĩnh trên đường cong / khi động cơ lệch.
    nd          : số chu kỳ trễ đầu vào. Mô hình được mở rộng bằng nd lệnh cũ u[k-1..k-nd]
                  (x[k+1] = Ad x[k] + Bd u[k-nd]) nên LQR "biết" lệnh vừa phát chưa tác dụng.
    Rời rạc hoá ZOH bằng hàm mũ ma trận; giải phương trình Riccati rời rạc.
    """
    from scipy.linalg import expm
    if model == 'kin':
        A = np.array([[0, v], [0, 0.0]])
        B = np.array([[0.0], [1.0]])
    else:
        A = np.array([[0, v, 0], [0, 0, 1.0], [0, 0, -1.0 / tau]])
        B = np.array([[0.0], [0.0], [1.0 / tau]])
    if integral:
        n = A.shape[0]
        A = np.block([[A, np.zeros((n, 1))], [np.eye(1, n), np.zeros((1, 1))]])
        B = np.vstack([B, [[0.0]]])
    n = A.shape[0]
    M = np.zeros((n + 1, n + 1))
    M[:n, :n], M[:n, n:] = A * Ts, B * Ts
    Md = expm(M)
    Ad, Bd = Md[:n, :n], Md[:n, n:]
    Qm = np.diag(Q)
    if nd > 0:                          # mở rộng trạng thái bằng bộ đệm lệnh cũ
        N = n + nd
        Aa = np.zeros((N, N))
        Aa[:n, :n] = Ad
        Aa[:n, N - 1:N] = Bd            # lệnh trễ nd chu kỳ tác động lên hệ
        for i in range(1, nd):
            Aa[n + i, n + i - 1] = 1.0  # dịch bộ đệm
        Ba = np.zeros((N, 1))
        Ba[n, 0] = 1.0
        Qa = np.zeros((N, N))
        Qa[:n, :n] = Qm
        Ad, Bd, Qm = Aa, Ba, Qa
    Rm = np.array([[R]])
    P = solve_discrete_are(Ad, Bd, Qm, Rm)
    K = np.linalg.solve(Rm + Bd.T @ P @ Bd, Bd.T @ P @ Ad)
    return K.ravel(), Ad, Bd


def bryson(d_max=0.01, psi_max=0.5, w_max=3.0, u_max=1.0, z_max=0.004, model='kin', integral=False):
    """Quy tắc Bryson: Q_ii = 1/x_i,max^2, R = 1/u_max^2."""
    q = [1 / d_max ** 2, 1 / psi_max ** 2]
    if model == 'dyn':
        q.append(1 / w_max ** 2)
    if integral:
        q.append(1 / z_max ** 2)
    return tuple(q), 1 / u_max ** 2


class LQR:
    """LQR / LQI rời rạc, hệ số lập lịch theo tốc độ.

    delay > 0: dùng mô hình có trễ đầu vào (bộ đệm lệnh cũ), thường đặt bằng trễ điều khiển đo được.
    model='dyn': ω không đo trực tiếp mà lấy từ mô hình động cơ chạy song song (lệnh đã trễ qua khâu bậc 1).
    """
    name = 'LQR'

    def __init__(self, Q=None, R=None, Ts=0.01, model='kin', tau=0.10, integral=False, z_lim=0.02, delay=0.0, **bry):
        if Q is None or R is None:
            Q, R = bryson(model=model, integral=integral, **bry)
        self.Q, self.R, self.Ts, self.model, self.tau, self.integral, self.z_lim = Q, R, Ts, model, tau, integral, z_lim
        self.nd = int(round(delay / Ts))
        self.cache = {}
        self.reset()

    def reset(self):
        self.z = 0.0
        self.om = 0.0
        self.buf = [0.0] * self.nd      # u[k-1], ..., u[k-nd]
        self.u_prev = 0.0

    def gain(self, v):
        key = round(max(v, 0.05), 2)
        if key not in self.cache:
            self.cache[key] = dlqr_gain(key, self.Ts, self.Q, self.R, self.model, self.tau, self.integral, self.nd)[0]
        return self.cache[key]

    def __call__(self, ey, eth, yf, v, om_meas=None):
        if ey is None:
            return 0.0
        K = self.gain(v)            # lập lịch hệ số theo tốc độ
        # mô hình động cơ song song: lệnh tác động lên robot là lệnh đã trễ nd chu kỳ
        u_act = self.buf[-1] if self.nd else self.u_prev
        a = np.exp(-self.Ts / self.tau)
        self.om = a * self.om + (1 - a) * u_act
        x = [-ey, -eth]             # d (robot lệch trái là dương), psi
        if self.model == 'dyn':
            x.append(self.om if om_meas is None else om_meas)
        if self.integral:
            self.z = float(np.clip(self.z + x[0] * self.Ts, -self.z_lim, self.z_lim))
            x.append(self.z)
        x += self.buf
        u = float(-K @ np.array(x))
        if self.nd:
            self.buf = [u] + self.buf[:-1]
        self.u_prev = u
        return u


def wheel_cmd(v, om, params):
    """Chuyển (v, omega) sang tốc độ bánh, ưu tiên giữ omega khi bão hoà."""
    p = params
    wl = (v - om * p.b / 2) / p.r
    wr = (v + om * p.b / 2) / p.r
    m = max(abs(wl), abs(wr))
    if m > p.w_max:
        dv = (m - p.w_max) * p.r
        v2 = max(v - dv, 0.0)
        wl = (v2 - om * p.b / 2) / p.r
        wr = (v2 + om * p.b / 2) / p.r
        m = max(abs(wl), abs(wr))
        if m > p.w_max:
            wl, wr = wl * p.w_max / m, wr * p.w_max / m
    return wl, wr
