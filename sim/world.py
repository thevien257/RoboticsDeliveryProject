"""Mô hình sân line: mạng các đoạn line (polyline) trên nền trắng và mô hình cảm biến hồng ngoại phản xạ.

Sân line chỉ được dùng để sinh tín hiệu cảm biến và để chấm điểm (ground truth).
Robot KHÔNG được đọc cấu trúc này; mọi thông tin bản đồ robot có đều do nó tự khám phá.
"""
import numpy as np


def bezier(p0, d0, p1, d1, n=60, k=0.4):
    """Đường cong Bézier bậc 3 đi từ p0 theo hướng d0, kết thúc tại p1 theo hướng d1 (hướng tiếp tuyến khi đến)."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    L = np.linalg.norm(p1 - p0)
    c0 = p0 + k * L * np.array([np.cos(d0), np.sin(d0)])
    c1 = p1 - k * L * np.array([np.cos(d1), np.sin(d1)])
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * c0 + 3 * (1 - t) * t ** 2 * c1 + t ** 3 * p1


def straight(p0, p1, step=0.01):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    n = max(2, int(np.ceil(np.linalg.norm(p1 - p0) / step)) + 1)
    t = np.linspace(0, 1, n)[:, None]
    return p0 + t * (p1 - p0)


def resample(poly, step=0.005):
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    ss = np.arange(0, s[-1], step)
    ss = np.append(ss, s[-1])
    return np.column_stack([np.interp(ss, s, poly[:, 0]), np.interp(ss, s, poly[:, 1])])


class Obstacle:
    """Vật cản hình tròn. kind='static' đứng yên; kind='moving' đi qua lại theo hướng `dirn` (người đi bộ).
    Người đi bộ nhường xe (yields=True): nếu bước tiếp theo đi vào vùng robot đang chiếm hoặc sắp đi qua thì đứng lại
    chờ – người không tự đâm vào xe; robot vẫn phải tự phát hiện người phía trước bằng cảm biến khoảng cách."""

    def __init__(self, center, radius=0.08, kind='static', dirn=0.0, amp=0.45, speed=0.35, pause=2.0, yields=True):
        self.c = np.array(center, float)
        self.r = radius
        self.kind = kind
        self.u = np.array([np.cos(dirn), np.sin(dirn)])
        self.amp, self.speed, self.pause = amp, speed, pause
        self.yields = yields and kind == 'moving'
        self.delay = 0.0                 # tổng thời gian đã đứng chờ nhường robot
        self._t_last = None

    def update(self, t, rx, ry, rth=0.0, rv=0.0, robot_r=0.075, margin=0.03, horizon=0.8):
        """Gọi mỗi chu kỳ mô phỏng: nếu bước tiếp theo đi vào vùng quanh robot hoặc quanh các vị trí robot sắp tới
        (dự đoán theo hướng và tốc độ trong `horizon` giây) thì đứng chờ."""
        if not self.yields:
            return
        if self._t_last is None or t < self._t_last:
            self._t_last = t
            return
        dt = t - self._t_last
        self._t_last = t
        now = self.pos(t)
        nxt = self._pos_at(t - self.delay + 0.05)
        lim = self.r + robot_r + margin
        for k in np.linspace(0.0, 1.0, 5):
            px, py = rx + k * horizon * rv * np.cos(rth), ry + k * horizon * rv * np.sin(rth)
            dn, dx = np.hypot(now[0] - px, now[1] - py), np.hypot(nxt[0] - px, nxt[1] - py)
            if dx < lim and dx < dn:
                self.delay += dt
                return

    def pos(self, t):
        if self.kind == 'static':
            return self.c
        return self._pos_at(t - self.delay)

    def _pos_at(self, t):
        T = 4 * self.amp / self.speed + 2 * self.pause      # chu kỳ: đi sang, dừng, đi về, dừng
        tt = t % T
        a, P = self.amp, self.pause
        leg = 2 * a / self.speed
        if tt < leg:
            x = -a + self.speed * tt
        elif tt < leg + P:
            x = a
        elif tt < 2 * leg + P:
            x = a - self.speed * (tt - leg - P)
        else:
            x = -a
        return self.c + x * self.u


class LineWorld:
    """Tập các polyline đen rộng `width` trên nền trắng."""

    def __init__(self, polylines, width=0.018, nodes=None, edges=None):
        self.width = width
        self.polylines = [resample(np.asarray(p, float)) for p in polylines]
        a, b, pid = [], [], []
        for i, p in enumerate(self.polylines):
            a.append(p[:-1]); b.append(p[1:]); pid += [i] * (len(p) - 1)
        self.A = np.vstack(a)
        self.B = np.vstack(b)
        self.D = self.B - self.A
        self.DD = np.maximum((self.D ** 2).sum(1), 1e-12)
        self.seg_pid = np.array(pid)
        self.lo = np.minimum(self.A, self.B)      # hộp bao từng đoạn (để chỉ xét các đoạn gần cảm biến)
        self.hi = np.maximum(self.A, self.B)
        self._F = None                            # bản đồ khoảng cách (tính khi cần lần đầu)
        self.nodes = nodes or {}      # ground truth: tên -> toạ độ (chỉ dùng để chấm điểm)
        self.edges = edges or []      # ground truth: (u, v, chỉ số polyline)
        # tình huống môi trường (do người dùng đặt trên GUI)
        self.gaps = []                # (tâm, bán kính): line bị mòn/đứt -> cảm biến thấy nền trắng
        self.slips = []               # (tâm, bán kính, sigma trượt): vùng sàn trơn
        self.obstacles = []           # đối tượng Obstacle
        self.light_noise = 0.0        # nhiễu ánh sáng cộng thêm vào cảm biến
        self.t = 0.0

    def dist(self, pts):
        """Khoảng cách từ các điểm (N,2) đến line gần nhất."""
        pts = np.atleast_2d(pts)
        rel = pts[:, None, :] - self.A[None, :, :]
        t = np.clip((rel * self.D[None]).sum(2) / self.DD[None], 0, 1)
        proj = self.A[None] + t[..., None] * self.D[None]
        d = np.linalg.norm(pts[:, None, :] - proj, axis=2)
        return d.min(1)

    def dist_near(self, pts, R=0.03):
        """Như dist() nhưng chỉ xét các đoạn có hộp bao cách cụm điểm ≤ R; điểm xa hơn R trả về R.
        Dùng cho cảm biến (phản xạ ≈ 0 khi cách line > R), nhanh hơn nhiều lần trên sân dài."""
        pts = np.atleast_2d(pts)
        q0, q1 = pts.min(0) - R, pts.max(0) + R
        m = np.all(self.hi >= q0, axis=1) & np.all(self.lo <= q1, axis=1)
        if not m.any():
            return np.full(len(pts), R)
        A, D, DD = self.A[m], self.D[m], self.DD[m]
        rel = pts[:, None, :] - A[None]
        t = np.clip((rel * D[None]).sum(2) / DD[None], 0, 1)
        dx = rel - t[..., None] * D[None]
        return np.minimum(np.sqrt((dx ** 2).sum(2)).min(1), R)

    # ------------------------------------------------------------ bản đồ khoảng cách (tra cứu nhanh)
    FIELD_H, FIELD_R = 0.001, 0.06            # ô 1 mm; chỉ lưu khoảng cách ≤ 60 mm (xa hơn coi là 60 mm)

    _FCACHE = {}

    def field(self):
        """Lưới khoảng cách tới line gần nhất, ô 1 mm, tính một lần (≈ 0,1–0,5 s) rồi tra cứu nội suy song tuyến.
        Lưới được nhớ theo hình học sân, nên các lượt chạy cùng sân trong một tiến trình không phải tính lại."""
        if self._F is None:
            key = (self.A.tobytes(), self.B.tobytes())
            if key in LineWorld._FCACHE:
                self._F = LineWorld._FCACHE[key]
                return self._F
            h, R = self.FIELD_H, self.FIELD_R
            x0, y0 = np.minimum(self.A, self.B).min(0) - R - 2 * h
            x1, y1 = np.maximum(self.A, self.B).max(0) + R + 2 * h
            nx, ny = int(np.ceil((x1 - x0) / h)) + 1, int(np.ceil((y1 - y0) / h)) + 1
            F = np.full((ny, nx), R, np.float32)
            for a, d, dd, lo, hi in zip(self.A, self.D, self.DD, self.lo, self.hi):
                i0, j0 = int((lo[0] - R - x0) / h), int((lo[1] - R - y0) / h)
                i1, j1 = int((hi[0] + R - x0) / h) + 2, int((hi[1] + R - y0) / h) + 2
                gx = x0 + h * np.arange(i0, i1)
                gy = y0 + h * np.arange(j0, j1)
                rx, ry = gx[None, :] - a[0], gy[:, None] - a[1]
                t = np.clip((rx * d[0] + ry * d[1]) / dd, 0, 1)
                dist = np.sqrt((rx - t * d[0]) ** 2 + (ry - t * d[1]) ** 2)
                np.minimum(F[j0:j1, i0:i1], dist.astype(np.float32), out=F[j0:j1, i0:i1])
            self._F = (F, x0, y0, h, R)
            if len(LineWorld._FCACHE) > 8:
                LineWorld._FCACHE.clear()
            LineWorld._FCACHE[key] = self._F
        return self._F

    def dist_field(self, pts):
        """Khoảng cách tới line (≤ FIELD_R) bằng tra bản đồ, nội suy song tuyến; pts (..., 2)."""
        F, x0, y0, h, R = self.field()
        u = (pts[..., 0] - x0) / h
        v = (pts[..., 1] - y0) / h
        i = np.floor(u).astype(np.int64)
        j = np.floor(v).astype(np.int64)
        ok = (i >= 0) & (j >= 0) & (i < F.shape[1] - 1) & (j < F.shape[0] - 1)
        i = np.where(ok, i, 0)
        j = np.where(ok, j, 0)
        fu, fv = u - i, v - j
        d = (F[j, i] * (1 - fu) * (1 - fv) + F[j, i + 1] * fu * (1 - fv) + F[j + 1, i] * (1 - fu) * fv + F[j + 1, i + 1] * fu * fv)
        return np.where(ok, d, R)

    def dist_to_poly(self, pts, pid, R=0.15):
        pts = np.atleast_2d(pts)
        q0, q1 = pts.min(0) - R, pts.max(0) + R
        m = (self.seg_pid == pid) & np.all(self.hi >= q0, axis=1) & np.all(self.lo <= q1, axis=1)
        if m.any():                       # chỉ xét đoạn gần; nếu kết quả < R thì đó là khoảng cách đúng
            A, D, DD = self.A[m], self.D[m], self.DD[m]
            rel = pts[:, None, :] - A[None]
            t = np.clip((rel * D[None]).sum(2) / DD[None], 0, 1)
            d = np.sqrt(((rel - t[..., None] * D[None]) ** 2).sum(2)).min(1)
            if (d < R).all():
                return d
        m = self.seg_pid == pid
        A, D, DD = self.A[m], self.D[m], self.DD[m]
        pts = np.atleast_2d(pts)
        rel = pts[:, None, :] - A[None]
        t = np.clip((rel * D[None]).sum(2) / DD[None], 0, 1)
        proj = A[None] + t[..., None] * D[None]
        return np.linalg.norm(pts[:, None, :] - proj, axis=2).min(1)

    def reflect(self, pts, rng=None, noise=0.03, edge=0.0025, avg=1):
        """Giá trị cảm biến chuẩn hoá: 1 = đen hoàn toàn, 0 = trắng. Mô hình biên mềm + nhiễu Gauss."""
        d = self.dist_field(pts)
        r = 1.0 / (1.0 + np.exp((d - self.width / 2) / edge))
        for c, rad in self.gaps:
            r[np.hypot(pts[:, 0] - c[0], pts[:, 1] - c[1]) < rad] = 0.0
        noise = (noise + self.light_noise) / np.sqrt(max(avg, 1))   # trung bình `avg` mẫu ADC độc lập
        if rng is not None and noise > 0:
            r = r + rng.normal(0, noise, size=r.shape)
        return np.clip(r, 0, 1)


# ------------------------------------------------------------------ các sân dùng trong thực nghiệm

def warehouse_map():
    """Mạng line mô hình kho: 5 đầu mút (vị trí đặt giá hàng), 5 giao lộ chữ T, một chu trình vuông.
    Các giao lộ J2..J5 có cùng hình dạng chữ T -> tạo tình huống dễ nhầm nút (perceptual aliasing)."""
    P = {
        'H': (0.0, 0.0), 'J1': (0.6, 0.0), 'A': (0.25, 0.75),
        'J2': (1.4, 0.0), 'B': (1.4, -0.7), 'J3': (2.2, 0.0), 'C': (2.85, -0.55),
        'J4': (1.4, 0.8), 'D': (0.7, 1.35), 'J5': (2.2, 0.8), 'E': (2.9, 1.25),
    }
    E, N, W, S = 0.0, np.pi / 2, np.pi, -np.pi / 2
    polys, edges = [], []

    def add(u, v, poly):
        edges.append((u, v, len(polys)))
        polys.append(poly)

    add('H', 'J1', straight(P['H'], P['J1']))
    add('J1', 'A', bezier(P['J1'], N, P['A'], W))
    add('J1', 'J2', straight(P['J1'], P['J2']))
    add('J2', 'B', bezier(P['J2'], S, P['B'], S, k=0.3))
    add('J2', 'J3', straight(P['J2'], P['J3']))
    add('J3', 'C', bezier(P['J3'], S, P['C'], E))
    add('J2', 'J4', straight(P['J2'], P['J4']))
    add('J4', 'J5', straight(P['J4'], P['J5']))
    add('J3', 'J5', straight(P['J3'], P['J5']))
    add('J4', 'D', bezier(P['J4'], W, P['D'], N))
    add('J5', 'E', bezier(P['J5'], E, P['E'], N))
    # J2 là ngã tư (W, E, S, N); J1, J3, J4, J5 là chữ T
    return LineWorld(polys, nodes=P, edges=edges)


def test_track(kind='mixed'):
    """Đường thử cho so sánh bộ điều khiển: thẳng, cong R=0.30, cong R=0.15, chữ S."""
    pts = [np.array([[0.0, 0.0], [1.0, 0.0]])]

    def arc(c, r, a0, a1, n=80):
        a = np.linspace(a0, a1, n)
        return np.column_stack([c[0] + r * np.cos(a), c[1] + r * np.sin(a)])
    if kind == 'mixed':
        pts.append(arc((1.0, 0.30), 0.30, -np.pi / 2, 0))            # rẽ trái R=0.30
        pts.append(straight((1.30, 0.30), (1.30, 0.80)))
        pts.append(arc((1.15, 0.80), 0.15, 0, np.pi))                  # cua chữ U R=0.15
        pts.append(straight((1.0, 0.80), (1.0, 0.60)))
        pts.append(arc((0.80, 0.60), 0.20, 0, -np.pi / 2))             # S: phải R=0.20
        pts.append(arc((0.80, 0.20), 0.20, np.pi / 2, np.pi))           # S: trái R=0.20
    elif kind == 'wave':
        x = np.linspace(1.0, 3.4, 300)
        pts.append(np.column_stack([x, 0.12 * np.sin(2 * np.pi * (x - 1.0) / 0.8)]))   # sóng hình sin, bán kính cong nhỏ nhất ~0.135 m
        pts.append(straight((3.4, 0.0), (3.8, 0.0)))
        pts.append(arc((3.8, 0.12), 0.12, -np.pi / 2, np.pi / 2))                   # quay đầu R=0.12
        pts.append(straight((3.8, 0.24), (3.0, 0.24)))
        pts.append(arc((3.0, 0.54), 0.30, -np.pi / 2, -np.pi))                     # rẽ phải R=0.30
        pts.append(straight((2.7, 0.54), (2.7, 1.0)))
    elif kind == 'straight_long':
        pts = [np.array([[0.0, 0.0], [6.0, 0.0]])]
    elif kind.startswith('circle'):
        R = float(kind.split('_')[1])
        pts = [np.array([[0.0, 0.0], [0.4, 0.0]]), arc((0.4, R), R, -np.pi / 2, -np.pi / 2 + 1.5 * np.pi, 200)]
    poly = np.vstack(pts)
    # bỏ điểm trùng
    keep = np.concatenate([[True], np.linalg.norm(np.diff(poly, axis=0), axis=1) > 1e-9])
    return LineWorld([poly[keep]])


def grid_map(sp=0.7):
    """Lưới 3x3 (12 cạnh) + 4 nhánh cụt ở góc: 8 giao lộ chữ T giống nhau, 1 ngã tư -> khó nhận dạng nút."""
    P, polys, edges = {}, [], []
    for i in range(3):
        for j in range(3):
            P[f'G{i}{j}'] = (i * sp, j * sp)
    for i in range(3):
        for j in range(3):
            if i < 2:
                edges.append((f'G{i}{j}', f'G{i+1}{j}', len(polys))); polys.append(straight(P[f'G{i}{j}'], P[f'G{i+1}{j}']))
            if j < 2:
                edges.append((f'G{i}{j}', f'G{i}{j+1}', len(polys))); polys.append(straight(P[f'G{i}{j}'], P[f'G{i}{j+1}']))
    spurs = {'S1': ('G00', (-0.45, -0.25), np.pi), 'S2': ('G20', (2 * sp + 0.45, -0.25), 0.0),
             'S3': ('G02', (-0.45, 2 * sp + 0.25), np.pi), 'S4': ('G22', (2 * sp + 0.45, 2 * sp + 0.25), 0.0)}
    for k, (g, pt, d0) in spurs.items():
        P[k] = pt
        dy = np.sign(pt[1] - P[g][1])
        edges.append((g, k, len(polys)))
        polys.append(bezier(P[g], d0, pt, np.pi / 2 * dy))
    return LineWorld(polys, nodes=P, edges=edges)


MAPS = {'Kho (11 nút, 1 chu trình)': warehouse_map, 'Lưới 3x3 (13 nút, nhiều nút giống nhau)': grid_map}
TRACKS = {'Đường thử hỗn hợp (thẳng + cua R0.30/R0.15 + S)': lambda: test_track('mixed'),
          'Đường sóng + quay đầu R0.12': lambda: test_track('wave')}


# ====================================================================== sân do người dùng vẽ
def _seg_inter(p, p2, q, q2):
    r, s = p2 - p, q2 - q
    den = r[0] * s[1] - r[1] * s[0]
    if abs(den) < 1e-12:
        return None
    t = ((q - p)[0] * s[1] - (q - p)[1] * s[0]) / den
    u = ((q - p)[0] * r[1] - (q - p)[1] * r[0]) / den
    if 1e-6 < t < 1 - 1e-6 and 1e-6 < u < 1 - 1e-6:
        return t, u
    return None


def build_from_strokes(strokes, snap=0.02, width=0.018):
    """Tạo LineWorld từ các nét vẽ (mỗi nét là danh sách điểm). Tự tách tại giao cắt, tự xác định nút thật
    (điểm có bậc khác 2) để chấm điểm. Robot không đọc phần nút/cạnh này."""
    segs = []
    for st in strokes:
        st = [np.array(p, float) for p in st]
        for a, b in zip(st[:-1], st[1:]):
            if np.linalg.norm(b - a) > 1e-4:
                segs.append([a, b])
    # tách tại giao cắt và tại điểm đầu mút nằm trên đoạn khác
    changed = True
    while changed:
        changed = False
        for i in range(len(segs)):
            for j in range(i + 1, len(segs)):
                a, b = segs[i]
                c, d = segs[j]
                r = _seg_inter(a, b, c, d)
                if r is not None:
                    x = a + r[0] * (b - a)
                    segs[i] = [a, x]; segs.append([x, b])
                    segs[j] = [c, x]; segs.append([x, d])
                    changed = True
                    break
                for e in (c, d):
                    ab = b - a
                    t = np.dot(e - a, ab) / max(np.dot(ab, ab), 1e-12)
                    if 1e-3 < t < 1 - 1e-3 and np.linalg.norm(a + t * ab - e) < snap:
                        segs[i] = [a, e.copy()]; segs.append([e.copy(), b])
                        changed = True
                        break
                if changed:
                    break
                for e in (a, b):
                    cd = d - c
                    t = np.dot(e - c, cd) / max(np.dot(cd, cd), 1e-12)
                    if 1e-3 < t < 1 - 1e-3 and np.linalg.norm(c + t * cd - e) < snap:
                        segs[j] = [c, e.copy()]; segs.append([e.copy(), d])
                        changed = True
                        break
                if changed:
                    break
            if changed:
                break
    # gộp đỉnh gần nhau
    verts = []

    def vid(p):
        for k, v in enumerate(verts):
            if np.linalg.norm(v - p) < 0.004:
                return k
        verts.append(p.copy())
        return len(verts) - 1
    E = []
    for a, b in segs:
        u, v = vid(a), vid(b)
        if u != v:
            E.append((u, v))
    adj = {k: [] for k in range(len(verts))}
    for idx, (u, v) in enumerate(E):
        adj[u].append((v, idx)); adj[v].append((u, idx))
    deg = {k: len(adj[k]) for k in adj}
    node_ids = [k for k in adj if deg[k] != 2 and deg[k] > 0]
    names = {k: f'N{i}' for i, k in enumerate(node_ids)}
    used = set()
    polys, edges = [], []
    for k in node_ids:
        for v0, e0 in adj[k]:
            if e0 in used:
                continue
            chain, prev, cur, e = [k], k, v0, e0
            used.add(e)
            chain.append(cur)
            while deg[cur] == 2 and cur not in names:
                nxt = [(w, ee) for w, ee in adj[cur] if ee not in used]
                if not nxt:
                    break
                w, ee = nxt[0]
                used.add(ee)
                prev, cur = cur, w
                chain.append(cur)
            polys.append(np.array([verts[c] for c in chain]))
            edges.append((names[k], names.get(cur, f'V{cur}'), len(polys) - 1))
    # vòng kín không có nút (chỉ đỉnh bậc 2)
    for idx, (u, v) in enumerate(E):
        if idx not in used:
            polys.append(np.array([verts[u], verts[v]]))
            used.add(idx)
    nodes = {names[k]: tuple(verts[k]) for k in node_ids}
    w = LineWorld(polys if polys else [np.array([[0, 0], [0.01, 0]])], width=width, nodes=nodes, edges=edges)
    w.strokes = [[tuple(map(float, p)) for p in st] for st in strokes]
    return w


def save_strokes(path, strokes):
    import json
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump({'strokes': strokes}, fh, indent=1)


def load_strokes(path):
    import json
    with open(path, encoding='utf-8') as fh:
        return json.load(fh)['strokes']
