"""Bản đồ topo do robot tự xây dựng và hai cách nhận dạng nút (gộp nút).

- 'naive' : gộp vào nút cũ gần nhất cùng bậc nếu vị trí odometry nằm trong bán kính R cố định.
- 'mht'   : nhiều giả thuyết (bộ lọc Bayes rời rạc) + ràng buộc topo (cổng đến phải chưa khám phá)
            + kiểm chứng chủ động khi còn mơ hồ.
"""
import numpy as np


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class Node:
    def __init__(self, nid, pos, ports, truth=None):
        self.id = nid
        self.pos = np.array(pos, float)
        self.ports = list(ports)                 # góc tuyệt đối (hệ odometry) của các nhánh
        self.links = [None] * len(ports)          # cổng -> (nút, cổng) hoặc None nếu chưa khám phá
        self.lengths = [None] * len(ports)
        self.truth = truth                        # nhãn thật (chỉ để chấm điểm)
        self.tentative = False
        self.n_visits = 1
        self.station = None
        self.blocked = set()                      # cổng bị vật cản chặn
        self.th = 0.0                             # góc xoay hệ cục bộ của nút (ports lưu trong hệ cục bộ)
        self.arr = [None] * len(ports)            # hướng robot (cục bộ) khi ĐI VÀO qua cổng; khác port+π nếu nhánh cong
        self.dep = [None] * len(ports)            # hướng robot (cục bộ) khi RỜI qua cổng (sau khi bắt line)

    def arr_dir(self, p):
        return self.arr[p] if self.arr[p] is not None else wrap(self.ports[p] + np.pi)

    def gports(self):
        """Góc các nhánh trong hệ bản đồ."""
        return [wrap(a + self.th) for a in self.ports]

    @property
    def degree(self):
        return len(self.ports)

    def port_of(self, angle, tol=np.deg2rad(35)):
        d = [abs(wrap(angle - a)) for a in self.gports()]
        i = int(np.argmin(d))
        return i if d[i] < tol else None

    def unexplored(self):
        return [i for i, l in enumerate(self.links) if l is None and i not in self.blocked]


def match_signature(ports_a, ports_b, tol=np.deg2rad(30)):
    """Hai tập góc nhánh có khớp nhau không (cho phép lệch hướng chung do trôi odometry).
    Trả về (khớp, độ lệch hướng ước lượng, phép gán cổng b->a)."""
    if len(ports_a) != len(ports_b):
        return False, 0.0, None
    best = (False, 0.0, None, 1e9)
    for off_ref in ports_a:
        for j0 in range(len(ports_b)):
            off = wrap(ports_b[j0] - off_ref)
            if abs(off) > np.deg2rad(40):
                continue
            assign, cost, ok = {}, 0.0, True
            used = set()
            for j, b in enumerate(ports_b):
                d = [abs(wrap(b - off - a)) if i not in used else 9 for i, a in enumerate(ports_a)]
                i = int(np.argmin(d))
                if d[i] > tol:
                    ok = False
                    break
                used.add(i)
                assign[j] = i
                cost += d[i]
            if ok and cost < best[3]:
                diffs = [wrap(ports_b[j] - ports_a[i]) for j, i in assign.items()]
                best = (True, float(np.mean(diffs)), assign, cost)
    return best[0], best[1], best[2]


class TopoMap:
    def __init__(self):
        self.nodes = {}
        self.next_id = 0
        self.log = []
        self.meas = []          # (i, j, dx, dy, L): độ dời odometry đo được khi đi cạnh i -> j

    def add_meas(self, i, pi, j, pj, rel, dth, L, phi):
        """Ràng buộc SE(2): rời nút i (hướng robot phi trong hệ cục bộ của i), đi độ dời rel (hệ robot lúc rời),
        quay dth, đến nút j qua cổng pj."""
        self.meas.append([i, pi, j, pj, float(rel[0]), float(rel[1]), float(dth), float(L), float(phi)])

    def optimize(self, anchor=None):
        """Tối ưu pose graph SE(2) (Grisetti et al.): ẩn (x, y, θ) của mọi nút trừ nút neo; Gauss–Newton/LM."""
        from scipy.optimize import least_squares
        ids = [k for k in self.nodes if not self.nodes[k].tentative]
        ms = [m for m in self.meas if m[0] in self.nodes and m[2] in self.nodes and m[0] != m[2]
              and m[1] < self.nodes[m[0]].degree and m[3] < self.nodes[m[2]].degree]
        if len(ids) < 2 or not ms:
            return 0.0
        anchor = anchor if anchor in self.nodes else ids[0]
        var = [k for k in ids if k != anchor]
        if not var:
            return 0.0
        vi = {k: n for n, k in enumerate(var)}
        x0 = np.concatenate([[self.nodes[k].pos[0], self.nodes[k].pos[1], self.nodes[k].th] for k in var])

        def pose(x, k):
            if k == anchor:
                N = self.nodes[k]
                return N.pos[0], N.pos[1], N.th
            n = vi.get(k)
            if n is None:
                N = self.nodes[k]
                return N.pos[0], N.pos[1], N.th
            return x[3 * n], x[3 * n + 1], x[3 * n + 2]

        def res(x):
            r = []
            for i, pi, j, pj, dx, dy, dth, L, phi in ms:
                xi, yi, ti = pose(x, i)
                xj, yj, tj = pose(x, j)
                h = ti + phi
                c, s_ = np.cos(h), np.sin(h)
                sp = 0.01 + 0.03 * np.sqrt(L)
                st = 0.05 + 0.06 * np.sqrt(L)
                r.append((xj - (xi + c * dx - s_ * dy)) / sp)
                r.append((yj - (yi + s_ * dx + c * dy)) / sp)
                r.append(wrap(h + dth - (tj + self.nodes[j].arr_dir(pj))) / st)
            return np.array(r)
        sol = least_squares(res, x0, method='lm' if len(res(x0)) >= len(x0) else 'trf')
        shift = 0.0
        for k in var:
            n = vi[k]
            new = np.array([sol.x[3 * n], sol.x[3 * n + 1]])
            shift = max(shift, float(np.linalg.norm(new - self.nodes[k].pos)))
            self.nodes[k].pos = new
            self.nodes[k].th = float(wrap(sol.x[3 * n + 2]))
        return shift

    def add_node(self, pos, ports, truth=None):
        n = Node(self.next_id, pos, ports, truth)
        self.nodes[n.id] = n
        self.next_id += 1
        return n

    def connect(self, a, pa, b, pb, length):
        A, B = self.nodes[a], self.nodes[b]
        A.links[pa] = (b, pb)
        A.lengths[pa] = length
        B.links[pb] = (a, pa)
        B.lengths[pb] = length

    def merge(self, t_id, x_id, assign):
        """Gộp nút tạm t vào nút x; assign: cổng của t -> cổng của x."""
        T, X = self.nodes[t_id], self.nodes[x_id]
        for pt, link in enumerate(T.links):
            if link is None:
                continue
            px = assign[pt]
            nb, pnb = link
            if nb == t_id:
                continue
            X.links[px] = (nb, pnb)
            X.lengths[px] = T.lengths[pt]
            if X.arr[px] is None and T.arr[pt] is not None:
                X.arr[px] = wrap(T.arr[pt] + T.th - X.th)
            self.nodes[nb].links[pnb] = (x_id, px)
        X.n_visits += 1
        for m in self.meas:
            if m[0] == t_id:
                m[0], m[1] = x_id, assign.get(m[1], m[1])
            if m[2] == t_id:
                m[2], m[3] = x_id, assign.get(m[3], m[3])
        if t_id in X.blocked:
            pass
        del self.nodes[t_id]

    def edges(self):
        out = set()
        for n in self.nodes.values():
            for p, l in enumerate(n.links):
                if l is not None:
                    a, b = (n.id, p), l
                    out.add((min(a, b), max(a, b), n.lengths[p]))
        return out

    def dijkstra(self, src, dst=None):
        import heapq
        dist = {src: 0.0}
        prev = {}
        pq = [(0.0, src)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist.get(u, 1e18):
                continue
            if u == dst:
                break
            for p, l in enumerate(self.nodes[u].links):
                if l is None or p in self.nodes[u].blocked:
                    continue
                v = l[0]
                nd = d + self.nodes[u].lengths[p]
                if nd < dist.get(v, 1e18):
                    dist[v] = nd
                    prev[v] = (u, p)
                    heapq.heappush(pq, (nd, v))
        return dist, prev

    def path(self, src, dst):
        """Danh sách (nút, cổng ra) để đi từ src đến dst."""
        dist, prev = self.dijkstra(src, dst)
        if dst not in dist:
            return None, np.inf
        seq, v = [], dst
        while v != src:
            u, p = prev[v]
            seq.append((u, p))
            v = u
        return seq[::-1], dist[dst]


class Associator:
    """Quyết định quan sát tại giao lộ là nút nào."""

    def __init__(self, method='mht', R_naive=0.25, sigma0=0.03, k_dist=0.06, p_accept=0.9):
        self.method = method
        self.R = R_naive
        self.sigma0 = sigma0
        self.k = k_dist
        self.p_accept = p_accept

    def candidates(self, tmap, pos, ports, arr_angle, s_since_anchor, exclude=(), sigma=None):
        """Trả về list (node_id, posterior, offset, assign) gồm cả 'new' (node_id=None)."""
        cands = []
        if self.method == 'naive':
            best, bd = None, self.R
            for n in tmap.nodes.values():
                if n.id in exclude or n.tentative or n.degree != len(ports):
                    continue
                d = np.linalg.norm(n.pos - pos)
                if d < bd:
                    ok, off, assign = match_signature(n.gports(), ports, tol=np.deg2rad(45))
                    best, bd = (n.id, off, assign if ok else None), d
            if best is None:
                return [(None, 1.0, 0.0, None)]
            nid, off, assign = best
            if assign is None:
                # gán cổng theo góc gần nhất
                X = tmap.nodes[nid]
                assign = {j: int(np.argmin([abs(wrap(b - a)) for a in X.gports()])) for j, b in enumerate(ports)}
            return [(nid, 1.0, off, assign)]
        sig = sigma if sigma is not None else self.sigma0 + self.k * s_since_anchor
        lik = []
        for n in tmap.nodes.values():
            if n.id in exclude or n.tentative:
                continue
            ok, off, assign = match_signature(n.gports(), ports)
            if not ok:
                continue
            # ràng buộc topo: cổng đi vào của nút cũ phải chưa được khám phá
            j_in = int(np.argmin([abs(wrap(b - arr_angle)) for b in ports]))
            if n.links[assign[j_in]] is not None:
                continue
            d2 = float(np.sum((n.pos - pos) ** 2))
            L = np.exp(-d2 / (2 * sig ** 2)) / (2 * np.pi * sig ** 2)
            lik.append((n.id, L, off, assign))
        # mật độ "nút mới": đều trên vùng đã biết mở rộng
        P = np.array([n.pos for n in tmap.nodes.values()] + [pos])
        area = max((np.ptp(P[:, 0]) + 1.0) * (np.ptp(P[:, 1]) + 1.0), 1.0)
        L_new = 1.0 / area
        prior_old = 0.5
        tot = sum(L for _, L, _, _ in lik) * prior_old / max(len(lik), 1) + L_new * (1 - prior_old)
        cands = [(nid, L * prior_old / max(len(lik), 1) / tot, off, a) for nid, L, off, a in lik]
        cands.append((None, L_new * (1 - prior_old) / tot, 0.0, None))
        cands.sort(key=lambda c: -c[1])
        return cands
