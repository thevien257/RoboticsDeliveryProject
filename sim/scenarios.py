"""Các kịch bản mô phỏng dùng chung cho GUI và thực nghiệm hàng loạt.

KB1 - Bám line: so sánh PID / LQR (mục 3.3, 4.1 của báo cáo).
KB2 - Tự khám phá và lập bản đồ topo: so sánh gộp nút đơn giản và nhiều giả thuyết (mục 3.3, 4.1).
KB3 - Gán trạm và giao nhiều đơn: so sánh FIFO / tham lam / quy hoạch động (mục 3.3, 4.1).
"""
import numpy as np
from .world import warehouse_map, test_track
from .robot import RobotParams
from .control import PID, LQR, PID_DEFAULT, LQR_DEFAULT

# tham số mặc định (cùng bộ với GUI)
TUNED = {
    'PID': dict(PID_DEFAULT),
    'LQR': dict(model='dyn', d_max=LQR_DEFAULT['d_max_mm'] / 1000, psi_max=np.radians(LQR_DEFAULT['psi_max_deg']),
                u_max=LQR_DEFAULT['u_max'], integral=LQR_DEFAULT['z_max_mm_s'] > 0, z_max=LQR_DEFAULT['z_max_mm_s'] / 1000),
}

# mức sai số encoder / cơ khí (robot không biết)
NOISE_LEVELS = {
    'Lý tưởng': dict(r_err_l=0.0, r_err_r=0.0, b_err=0.0, slip=0.0),
    'Thấp': dict(r_err_l=0.005, r_err_r=-0.005, b_err=0.02, slip=0.01),
    'Trung bình': dict(r_err_l=0.01, r_err_r=-0.01, b_err=0.04, slip=0.02),
    'Cao': dict(r_err_l=0.02, r_err_r=-0.02, b_err=0.06, slip=0.04),
}


def make_ctrl(name, **over):
    kw = dict(TUNED[name])
    kw.update(over)
    return {'PID': PID, 'LQR': LQR}[name](**kw)


def map_score(world, tmap):
    """So bản đồ học được với sân thật."""
    nodes = [n for n in tmap.nodes.values()]
    labels = [n.truth for n in nodes]
    true_edges = {frozenset((u, v)) for u, v, _ in world.edges}
    learned = set()
    bad_edges = 0
    for n in nodes:
        for l in n.links:
            if l is None:
                continue
            e = frozenset((n.truth, tmap.nodes[l[0]].truth))
            learned.add(e)
            if e not in true_edges:
                bad_edges += 1
    correct = (len(nodes) == len(world.nodes) and len(set(labels)) == len(world.nodes)
               and None not in labels and learned == true_edges and bad_edges == 0)
    return dict(n_nodes=len(nodes), n_true=len(world.nodes), bad_edges=bad_edges // 2,
                edges_found=len(learned & true_edges), edges_true=len(true_edges), correct=bool(correct))


# kịch bản tình huống khi giao hàng trên sân kho (dùng trong bộ thí nghiệm E3)
SCEN = ['Bình thường', 'Người đi ngang tuyến', 'Vật cản chắn đoạn J2–J4', 'Line mòn/đứt 2 chỗ', 'Sàn trơn',
        'Ánh sáng nhiễu mạnh', 'Pin yếu (ω_max −40%)', 'Tải nặng (τ = 0,18 s)']


def apply_scen(ag, w, scen):
    from .world import Obstacle
    if scen == 'Người đi ngang tuyến':
        w.obstacles.append(Obstacle((1.8, 0.0), 0.10, 'moving', dirn=np.pi / 2, amp=0.45, speed=0.4, pause=1.5))
        w.obstacles.append(Obstacle((2.2, 0.4), 0.10, 'moving', dirn=0.0, amp=0.45, speed=0.4, pause=1.5))
    elif scen.startswith('Vật cản chắn'):
        w.obstacles.append(Obstacle((1.4, 0.4), 0.07, 'static'))
    elif scen.startswith('Line mòn'):
        w.gaps += [((1.0, 0.0), 0.02), ((2.2, 0.45), 0.02)]
    elif scen == 'Sàn trơn':
        w.slips.append(((1.8, 0.0), 0.35, 0.15))
    elif scen.startswith('Ánh sáng'):
        w.light_noise = 0.12
    elif scen.startswith('Pin'):
        ag.p.w_max *= 0.6
    elif scen.startswith('Tải'):
        ag.p.tau_load = 0.18
