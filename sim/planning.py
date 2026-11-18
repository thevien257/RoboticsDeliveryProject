"""Lập lịch thực hiện nhiều đơn lấy–trả hàng với ràng buộc chở một kiện mỗi lần."""
import itertools
import numpy as np


def dist_matrix(tmap, node_ids):
    n = len(node_ids)
    D = np.full((n, n), np.inf)
    for i, u in enumerate(node_ids):
        dist, _ = tmap.dijkstra(u)
        for j, v in enumerate(node_ids):
            D[i, j] = dist.get(v, np.inf)
    return D


def _start_dist(tmap, start_node, node_ids):
    dist, _ = tmap.dijkstra(start_node)
    return np.array([dist.get(v, np.inf) for v in node_ids])


def cost_of(order, D, orders, d0):
    c, cur = 0.0, None
    for k in order:
        p, q = orders[k]
        c += (d0[p] if cur is None else D[cur, p]) + D[p, q]
        cur = q
    return c


def schedule_fifo(D, orders, s_idx, start_node, tmap, node_ids):
    return list(range(len(orders)))


def schedule_greedy(D, orders, s_idx, start_node, tmap, node_ids):
    d0 = _start_dist(tmap, start_node, node_ids)
    left, out, cur = set(range(len(orders))), [], None
    while left:
        k = min(left, key=lambda k: (d0[orders[k][0]] if cur is None else D[cur, orders[k][0]]))
        out.append(k)
        left.remove(k)
        cur = orders[k][1]
    return out


def schedule_dp(D, orders, s_idx, start_node, tmap, node_ids):
    """Quy hoạch động trên tập con (Held–Karp): f(S, j) = chi phí nhỏ nhất làm xong tập S, đơn cuối là j."""
    n = len(orders)
    d0 = _start_dist(tmap, start_node, node_ids)
    INF = np.inf
    f = np.full((1 << n, n), INF)
    par = np.full((1 << n, n), -1, dtype=int)
    for j in range(n):
        p, q = orders[j]
        f[1 << j, j] = d0[p] + D[p, q]
    for S in range(1, 1 << n):
        for j in range(n):
            if not (S >> j) & 1 or f[S, j] == INF:
                continue
            qj = orders[j][1]
            for k in range(n):
                if (S >> k) & 1:
                    continue
                p, q = orders[k]
                c = f[S, j] + D[qj, p] + D[p, q]
                T = S | (1 << k)
                if c < f[T, k]:
                    f[T, k] = c
                    par[T, k] = j
    S = (1 << n) - 1
    j = int(np.argmin(f[S]))
    out = []
    while j >= 0:
        out.append(j)
        pj = par[S, j]
        S ^= 1 << j
        j = pj
    return out[::-1]


def schedule_brute(D, orders, s_idx, start_node, tmap, node_ids):
    d0 = _start_dist(tmap, start_node, node_ids)
    best = min(itertools.permutations(range(len(orders))), key=lambda o: cost_of(o, D, orders, d0))
    return list(best)


SCHEDULERS = {'FIFO (theo thứ tự nhập)': schedule_fifo, 'Tham lam (gần nhất)': schedule_greedy,
              'Quy hoạch động (Held–Karp)': schedule_dp}


# ====================================================================== lấy–trả có sức chứa Q kiện
def plan_stops(D, orders, d0, Q=1, method='dp'):
    """Lập kế hoạch dãy điểm dừng cho bài toán lấy–trả có sức chứa (capacitated pickup & delivery).

    D: ma trận khoảng cách giữa các trạm; orders: [(lấy, trả)] theo chỉ số trạm; d0: khoảng cách từ vị trí robot.
    Trạng thái mỗi đơn: 0 chưa lấy, 1 đang chở, 2 đã trả. Ràng buộc: số đơn đang chở <= Q; lấy trước khi trả.
    method='dp': tìm kiếm Dijkstra trên không gian trạng thái (tối ưu, 3^n trạng thái × vị trí);
    method='greedy': luôn đi tới điểm dừng hợp lệ gần nhất; method='fifo': làm từng đơn theo thứ tự nhập.
    Trả về (danh sách (trạm, 'lấy'/'trả', chỉ số đơn), tổng quãng đường).
    """
    import heapq
    n = len(orders)
    if n == 0:
        return [], 0.0

    def moves(st, loc):
        load = sum(1 for s in st if s == 1)
        for k, s in enumerate(st):
            if s == 0 and load < Q:
                yield k, orders[k][0], 'lấy'
            elif s == 1:
                yield k, orders[k][1], 'trả'

    def dist(loc, to):
        return d0[to] if loc is None else D[loc, to]

    if method == 'fifo':
        seq, cost, loc = [], 0.0, None
        for k, (a, b) in enumerate(orders):
            cost += dist(loc, a) + D[a, b]
            seq += [(a, 'lấy', k), (b, 'trả', k)]
            loc = b
        return seq, cost
    if method == 'greedy':
        st, loc, seq, cost = [0] * n, None, [], 0.0
        while any(s != 2 for s in st):
            k, to, act = min(moves(tuple(st), loc), key=lambda m: (dist(loc, m[1]), m[2] != 'trả'))
            cost += dist(loc, to)
            st[k] += 1
            seq.append((to, act, k))
            loc = to
        return seq, cost
    start = (tuple([0] * n), None)
    best = {start: 0.0}
    prev = {}
    pq = [(0.0, 0, start)]
    cnt = 0
    goal = tuple([2] * n)
    while pq:
        c, _, (st, loc) = heapq.heappop(pq)
        if c > best.get((st, loc), 1e18):
            continue
        if st == goal:
            seq, key = [], (st, loc)
            while key in prev:
                key, step = prev[key]
                seq.append(step)
            return seq[::-1], c
        for k, to, act in moves(st, loc):
            ns = list(st)
            ns[k] += 1
            key = (tuple(ns), to)
            nc = c + dist(loc, to)
            if nc < best.get(key, 1e18) - 1e-12:
                best[key] = nc
                prev[key] = ((st, loc), (to, act, k))
                cnt += 1
                heapq.heappush(pq, (nc, cnt, key))
    return [], float('inf')


STOP_METHODS = {'FIFO (theo thứ tự nhập)': 'fifo', 'Tham lam (gần nhất)': 'greedy', 'Quy hoạch động (Held–Karp)': 'dp'}
