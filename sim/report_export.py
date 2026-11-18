"""Xuất dữ liệu cho báo cáo cuối kỳ (nút "Xuất dữ liệu cho báo cáo" trên GUI).

Bám theo hướng dẫn viết báo cáo và rubric của học phần:
  mục 3.1  tiêu chí đánh giá, đầu vào/đầu ra            -> B2_tieu_chi_danh_gia.csv
  mục 3.4  thông số, môi trường mô phỏng, điều kiện      -> B1_thong_so_mo_phong.csv, thiet_lap_tai_lap.json
  mục 4.1  kết quả có minh chứng (bảng, hình)           -> B3_ket_qua_cac_luot.csv, H1–H6
  mục 4.2  so sánh theo tiêu chí rõ ràng, mức đạt mục tiêu -> B4_so_sanh_phuong_phap.csv, B5_doi_chieu_muc_tieu.csv,
                                                           B6_lap_lich.csv
  mục 4.3  hạn chế quan sát được                        -> B9_han_che.csv
  phụ lục  kết quả chi tiết                              -> B7_nut_da_hoc.csv, B8_giao_hang.csv, phu_luc_chuoi_thoi_gian/
  rubric C5: hình, bảng đánh số, có chú thích và nguồn   -> DANH_MUC_HINH_BANG.md
Dữ liệu là các lượt chạy trong phiên GUI hiện tại (học map, giao hàng), không phải số liệu soạn sẵn.
"""
import os, json, time
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from .export import timeseries, summarize
from .planning import plan_stops

# ---------------------------------------------------------------- tiêu chí và mục tiêu
CRITERIA = [
    ('RMS lệch line', 'track_rms_mm', 'mm', 'sqrt(mean(d²)), d = khoảng cách từ tâm trục bánh tới line gần nhất (vị trí thật), chỉ tính lúc bám line'),
    ('P95 lệch line', 'track_p95_mm', 'mm', 'phân vị 95 % của d'),
    ('Lệch line lớn nhất', 'track_max_mm', 'mm', 'max(d)'),
    ('Độ giật lệnh ω', 'om_jerk', 'rad/s²', 'RMS(Δω/Ts), Ts = 10 ms; đặc trưng độ êm, mức tải động cơ'),
    ('Số lần mất line', 'stat_lost', 'lần', 'số lần cả hai dãy cảm biến không thấy line quá thời gian chạy trớn'),
    ('Sai số odometry TB', 'odo_err_mean_mm', 'mm', 'trung bình |p_thật − p_odometry| theo thời gian'),
    ('Sai số hướng lớn nhất', 'th_err_max_deg', 'độ', 'max |θ_thật − θ_odometry|'),
    ('Bản đồ đúng hoàn toàn', 'map_correct', '–', 'đủ nút, đủ cạnh, không gộp nhầm, không cạnh sai so với sân thật'),
    ('Sai số vị trí nút TB', 'node_err_mean_mm', 'mm', 'trung bình |vị trí nút học − vị trí giao lộ thật|'),
    ('Sai số dừng tại trạm TB', 'stop_err_mean_mm', 'mm', 'khoảng cách tâm robot – điểm nút thật khi dừng lấy/trả'),
    ('Tỉ lệ dừng đúng trạm', 'stops_correct', '–', 'số lần dừng đúng nút / tổng số lần dừng'),
    ('Va chạm', 'collisions', 'lần', 'số lần thân robot chạm vật cản'),
    ('Quãng đường', 'distance_m', 'm', 'tổng quãng đường robot đi trong lượt'),
    ('Thời gian lượt', 'time_s', 's', 'thời gian mô phỏng từ lúc bắt đầu tới lúc xong lượt'),
]

METHOD_NAME = {'fifo': 'FIFO', 'greedy': 'Tham lam', 'dp': 'Quy hoạch động'}

OBJECTIVES = [   # (mã, nội dung, tiêu chí kiểm chứng) – trùng bảng mục tiêu ở mục 1.2 của báo cáo
    ('MT1', 'Bám line ổn định', 'RMS lệch line < 5 mm và không mất line'),
    ('MT2', 'Tự học bản đồ mạng line không dùng mốc', 'Bản đồ đúng hoàn toàn (đủ nút, đủ cạnh, không gộp nhầm)'),
    ('MT3', 'Định vị nút đủ chính xác để gán trạm', 'Sai số vị trí nút TB < 100 mm'),
    ('MT4', 'Giao đúng trạm, dừng chính xác', 'Đúng trạm 100 %, sai số dừng TB < 10 mm'),
    ('MT5', 'An toàn và hoàn thành đơn khi có tình huống', 'Không va chạm, hoàn thành mọi đơn của lượt'),
    ('MT6', 'Lập lịch nhiều đơn tối ưu', 'Kế hoạch QHĐ không dài hơn FIFO và tham lam trên cùng bộ đơn'),
    ('MT7', 'GUI cho toàn bộ quy trình', 'Vẽ sân → học map → gán trạm → giao hàng, đổi thuật toán khi chạy'),
]


def snapshot(agent, world, task, settings, start, label=None):
    """Chụp lại một lượt chạy (gọi khi lượt xong) – giữ trong bộ nhớ đến khi người dùng xuất."""
    df = timeseries(agent, world, start)
    summ = summarize(agent, world, df, task, settings)
    nodes = []
    for n in agent.map.nodes.values():
        tp = world.nodes.get(n.truth)
        nodes.append(dict(nut=n.id, x_hoc=n.pos[0], y_hoc=n.pos[1], bac=n.degree, tram=n.station, nut_that=n.truth,
                          x_that=tp[0] if tp is not None else np.nan, y_that=tp[1] if tp is not None else np.nan,
                          sai_so_mm=np.hypot(*(n.pos - np.array(tp))) * 1000 if tp is not None else np.nan))
    edges = [(n.pos.copy(), agent.map.nodes[l[0]].pos.copy()) for n in agent.map.nodes.values() for l in n.links
             if l is not None and n.id < l[0]]
    t0 = df.t.iloc[0] if len(df) else 0.0
    dl = [dict(d) for d in agent.delivery if d['t'] >= t0]
    plan = None
    if task == 'giao_hang' and getattr(agent, 'plan_inputs', None):
        pi = agent.plan_inputs
        plan = dict(orders=[(pi['names'][a], pi['names'][b]) for a, b in pi['orders']], Q=pi['Q'], method=pi['method'],
                    chosen=float(agent.plan_cost), text=str(agent.plan_text), costs=[])
        if len(pi['orders']) <= 7:
            for q in sorted({1, 2, 3, pi['Q']}):
                for m, name in [('fifo', 'FIFO'), ('greedy', 'Tham lam'), ('dp', 'Quy hoạch động')]:
                    t1 = time.perf_counter()
                    _, c = plan_stops(pi['D'], pi['orders'], pi['d0'], q, m)
                    plan['costs'].append(dict(suc_chua=q, phuong_phap=name, quang_duong_m=float(c),
                                              thoi_gian_tinh_ms=(time.perf_counter() - t1) * 1000))
        summ['metrics']['orders'] = len(pi['orders'])
        summ['metrics']['orders_done'] = int(sum(1 for d in dl if d['action'] == 'trả'))
    return dict(label=label or task, task=task, summary=summ, df=df, nodes=nodes, edges=edges, deliveries=dl, plan=plan,
                events=[e for e in agent.events if e[0] >= t0], params=dict(vars(agent.p)), world=world)


def _run_label(i, r):
    c = r['summary']['controller']
    s = r['summary']['settings']
    g = f"PID {c['Kp']:g}/{c['Ki']:g}/{c['Kd']:g}" if c['controller'] == 'PID' else 'LQR'
    extra = []
    if not s.get('gyro'):
        extra.append('không IMU')
    if s.get('scenario') and not str(s['scenario']).startswith('Không'):
        extra.append(str(s['scenario']).split('(')[0].strip()[:22])
    for k, t in [('light', 'nhiễu sáng'), ('low_battery', 'pin yếu'), ('heavy', 'tải nặng')]:
        if s.get(k):
            extra.append(t)
    task = 'học map' if r['task'] == 'hoc_map' else 'giao hàng'
    return f"L{i + 1} {task}, {g}" + (', ' + ', '.join(extra) if extra else '')


def _save(fig, path):
    FigureCanvasAgg(fig)
    fig.tight_layout()
    fig.savefig(path, dpi=200)


def _csv(df, path):
    df.to_csv(path, index=False, encoding='utf-8-sig', float_format='%.4g')


def _objectives(runs):
    rows = []
    learn = [r for r in runs if r['task'] == 'hoc_map']
    deliv = [r for r in runs if r['task'] == 'giao_hang']
    M = lambda r: r['summary']['metrics']
    for code, what, crit in OBJECTIVES:
        meas, ok = [], []
        if code == 'MT1':
            for r in runs:
                m = M(r)
                if 'track_rms_mm' in m:
                    meas.append(f"{r['ten']}: {m['track_rms_mm']:.1f} mm, mất line {int(m.get('stat_lost', 0))}")
                    ok.append(m['track_rms_mm'] < 5 and m.get('stat_lost', 0) == 0)
        elif code == 'MT2':
            for r in learn:
                m = M(r)
                meas.append(f"{r['ten']}: {m.get('map_n_nodes', 0)}/{m.get('map_n_true', 0)} nút, "
                            f"{m.get('map_edges_found', 0)}/{m.get('map_edges_true', 0)} cạnh")
                ok.append(bool(m.get('map_correct')))
        elif code == 'MT3':
            for r in learn:
                m = M(r)
                if 'node_err_mean_mm' in m:
                    meas.append(f"{r['ten']}: TB {m['node_err_mean_mm']:.0f} mm, max {m['node_err_max_mm']:.0f} mm")
                    ok.append(m['node_err_mean_mm'] < 100)
        elif code == 'MT4':
            for r in deliv:
                m = M(r)
                if 'stops' in m:
                    meas.append(f"{r['ten']}: đúng trạm {m['stops_correct'] * 100:.0f} %, sai số dừng TB {m['stop_err_mean_mm']:.1f} mm")
                    ok.append(m['stops_correct'] == 1 and m['stop_err_mean_mm'] < 10)
        elif code == 'MT5':
            for r in deliv:
                m = M(r)
                meas.append(f"{r['ten']}: va chạm {m['collisions']}, xong {m.get('orders_done', 0)}/{m.get('orders', 0)} đơn")
                ok.append(m['collisions'] == 0 and m.get('orders_done', 0) == m.get('orders', -1))
        elif code == 'MT6':
            for r in deliv:
                if r['plan'] and r['plan']['costs']:
                    c = pd.DataFrame(r['plan']['costs'])
                    c = c[c.suc_chua == r['plan']['Q']].set_index('phuong_phap').quang_duong_m
                    meas.append(f"{r['ten']} (sức chứa {r['plan']['Q']}): QHĐ {c['Quy hoạch động']:.2f} m, "
                                f"tham lam {c['Tham lam']:.2f} m, FIFO {c['FIFO']:.2f} m")
                    ok.append(c['Quy hoạch động'] <= min(c['Tham lam'], c['FIFO']) + 1e-9)
        elif code == 'MT7':
            meas.append('Toàn bộ các lượt trong bảng B3 được thực hiện trên GUI')
            ok.append(True)
        verdict = '–' if not ok else ('Đạt' if all(ok) else (f'Đạt {sum(ok)}/{len(ok)} lượt' if any(ok) else 'Chưa đạt'))
        rows.append(dict(ma=code, muc_tieu=what, tieu_chi_kiem_chung=crit, ket_qua_do=' ; '.join(meas) or 'chưa có lượt phù hợp',
                         danh_gia=verdict))
    return pd.DataFrame(rows)


def _limitations(runs, world):
    rows = []
    for r in runs:
        m = r['summary']['metrics']
        st = {k[5:]: v for k, v in m.items() if k.startswith('stat_')}
        checks = [
            (st.get('lost', 0) > 0, f"mất line {int(st.get('lost', 0))} lần", 'Bám line', 'Kp quá lớn / tốc độ cao / đoạn line đứt; xem hình H3'),
            (st.get('wrong_merge', 0) > 0, f"gộp nhầm nút {int(st.get('wrong_merge', 0))} lần", 'Học map', 'sai số odometry vượt cổng nhận dạng'),
            (st.get('duplicate', 0) > 0, f"tạo nút trùng {int(st.get('duplicate', 0))} lần", 'Học map', 'nhận dạng chưa khớp nút cũ'),
            (r['task'] == 'hoc_map' and not m.get('map_correct', True), 'bản đồ chưa đúng hoàn toàn', 'Học map', 'xem B7 và hình H1'),
            (m.get('collisions', 0) > 0, f"va chạm {m.get('collisions')} lần", 'An toàn', 'vật cản vào vùng mù của cảm biến khoảng cách'),
            (st.get('wait_time', 0) > 0, f"chờ vật cản {st.get('wait_time', 0):.1f} s, lập lại đường {int(st.get('replans', 0))} lần",
             'Tình huống', 'đường bị chắn, robot dừng/đổi đường'),
            (m.get('odo_err_max_mm', 0) > 100, f"sai số odometry lớn nhất {m.get('odo_err_max_mm', 0):.0f} mm", 'Định vị',
             'trôi tích luỹ giữa hai lần neo tại nút'),
            (m.get('stop_err_mean_mm', 0) >= 10, f"sai số dừng TB {m.get('stop_err_mean_mm', 0):.1f} mm", 'Giao hàng',
             'quán tính khi phanh, bề rộng line'),
            (m.get('orders', 0) and m.get('orders_done', 0) < m.get('orders', 0),
             f"chỉ xong {m.get('orders_done', 0)}/{m.get('orders', 0)} đơn", 'Giao hàng', 'không tới được trạm (cạnh bị chặn)'),
        ]
        for cond, what, area, why in checks:
            if cond:
                rows.append(dict(luot=r['ten'], khia_canh=area, hien_tuong=what, nguyen_nhan_co_the=why))
    if not rows:
        rows.append(dict(luot='–', khia_canh='–', hien_tuong='không ghi nhận lỗi trong các lượt đã chạy', nguyen_nhan_co_the='–'))
    rows.append(dict(luot='chung', khia_canh='Phạm vi', hien_tuong='kết quả từ mô phỏng, chưa kiểm chứng trên robot thật',
                     nguyen_nhan_co_the='mô hình động cơ bậc 1, trễ cố định, cảm biến lý tưởng hoá'))
    return pd.DataFrame(rows)


def export_report(runs, world, out_root, strokes=None, gui_note=''):
    """Ghi toàn bộ bảng, hình, danh mục vào out_root/bao_cao_<thời điểm>/. Trả về đường dẫn thư mục."""
    if not runs:
        raise ValueError('chưa có lượt chạy')
    out = os.path.join(out_root, 'bao_cao_' + time.strftime('%Y%m%d_%H%M%S'))
    os.makedirs(os.path.join(out, 'phu_luc_chuoi_thoi_gian'), exist_ok=True)
    for i, r in enumerate(runs):
        r['ten'] = _run_label(i, r)
    catalog = []   # (loại, số, tệp, chú thích, mục báo cáo, nguồn/cách tính)

    def tab(df, fn, cap, sec, src):
        _csv(df, os.path.join(out, fn))
        catalog.append(('Bảng', fn, cap, sec, src))

    def fig_(f, fn, cap, sec, src):
        _save(f, os.path.join(out, fn))
        catalog.append(('Hình', fn, cap, sec, src))

    last = runs[-1]
    P = last['params']
    world = last['world']
    # B1 – thông số
    lengths = [float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1))) for p in world.polylines]
    hw = [('Bán kính bánh (danh định)', P['r'] * 1000, 'mm'), ('Khoảng cách hai bánh (danh định)', P['b'] * 1000, 'mm'),
          ('Độ phân giải encoder', P['cpr'], 'xung/vòng bánh'), ('Tốc độ bánh tối đa', P['w_max'], 'rad/s'),
          ('Hằng số thời gian động cơ (không tải / có tải)', f"{P['tau'] * 1000:.0f} / {P['tau_load'] * 1000:.0f}", 'ms'),
          ('Trễ điều khiển', P['latency'] * 1000, 'ms'), ('Chu kỳ điều khiển', 10, 'ms'),
          ('Số cảm biến mỗi dãy × khoảng cách', f"{P['n_sens']} × {P['pitch'] * 1000:.0f}", 'mm'),
          ('Vị trí dãy trước / dãy sau tính từ trục bánh', f"{P['x_front'] * 1000:.0f} / {P['x_rear'] * 1000:.0f}", 'mm'),
          ('Nhiễu cảm biến line (chuẩn hoá)', P['sens_noise'], '–'), ('IMU (gyro)', 'có' if P['gyro'] else 'không', '–'),
          ('Độ lệch 0 gyro thật / nhiễu', f"{P['gyro_bias']:.4f} / {P['gyro_noise']:.4f}", 'rad/s'),
          ('Sai lệch bán kính bánh trái / phải (robot không biết)', f"{P['r_err_l'] * 100:+.2f} / {P['r_err_r'] * 100:+.2f}", '%'),
          ('Sai lệch khoảng cách bánh (robot không biết)', P['b_err'] * 100, '%'), ('Trượt bánh ngẫu nhiên (độ lệch chuẩn)', P['slip'] * 100, '%'),
          ('Sân line: số nét vẽ / nút thật / cạnh thật', f"{len(strokes or [])} / {len(world.nodes)} / {len(world.edges)}", '–'),
          ('Tổng chiều dài line', sum(lengths), 'm'), ('Bề rộng line', world.width * 1000, 'mm')]
    tab(pd.DataFrame(hw, columns=['thong_so', 'gia_tri', 'don_vi']), 'B1_thong_so_mo_phong.csv',
        'Thông số robot, cảm biến và môi trường mô phỏng', '3.4', 'RobotParams và sân do người dùng vẽ trên GUI')
    # B2 – tiêu chí
    tab(pd.DataFrame([dict(tieu_chi=a, ma=b, don_vi=c, cach_tinh=d) for a, b, c, d in CRITERIA]), 'B2_tieu_chi_danh_gia.csv',
        'Tiêu chí đánh giá và cách tính', '3.1', 'định nghĩa trong sim/export.py; vị trí thật chỉ dùng để chấm điểm')
    # B3 – kết quả từng lượt
    rows = []
    for r in runs:
        m, c, s = r['summary']['metrics'], r['summary']['controller'], r['summary']['settings']
        row = dict(luot=r['ten'], tac_vu=r['task'], bo_dieu_khien=c['controller'],
                   tham_so=(f"Kp={c['Kp']:g}, Ki={c['Ki']:g}, Kd={c['Kd']:g}" if c['controller'] == 'PID'
                            else 'K=[' + ', '.join(f'{k:.2f}' for k in c.get('K', [])) + ']'),
                   toc_do=s.get('speed'), imu=s.get('gyro'), sai_so_banh=s.get('wheel_error'), kich_ban=s.get('scenario'),
                   nhan_dang=s.get('association'), pose_graph=s.get('pose_graph'))
        row.update({b: m.get(b) for _, b, _, _ in CRITERIA})
        rows.append(row)
    B3 = pd.DataFrame(rows)
    tab(B3, 'B3_ket_qua_cac_luot.csv', 'Kết quả đo của từng lượt chạy trong phiên GUI', '4.1',
        'đo trực tiếp từ mô phỏng, mỗi chu kỳ 10 ms')
    # B4 – so sánh phương pháp: gom theo bộ điều khiển + tác vụ, kèm chênh lệch so với nhóm đầu
    keys = ['track_rms_mm', 'track_p95_mm', 'om_jerk', 'stat_lost', 'odo_err_mean_mm', 'node_err_mean_mm', 'stop_err_mean_mm', 'time_s']
    B3['nhom'] = B3.bo_dieu_khien + ' | ' + B3.tham_so + ' | IMU=' + B3.imu.astype(str)
    g = B3.groupby(['tac_vu', 'nhom'])[[k for k in keys if k in B3]].mean(numeric_only=True).reset_index()
    g.insert(2, 'so_luot', B3.groupby(['tac_vu', 'nhom']).size().values)
    tab(g, 'B4_so_sanh_phuong_phap.csv', 'So sánh các cấu hình/bộ điều khiển theo cùng tiêu chí (trung bình các lượt)', '4.2',
        'gộp từ B3; mỗi nhóm là một bộ điều khiển với bộ tham số và cấu hình cảm biến')
    # B5 – đối chiếu mục tiêu
    tab(_objectives(runs), 'B5_doi_chieu_muc_tieu.csv', 'Đối chiếu kết quả với mục tiêu MT1–MT7', '4.2, 5.1', 'B3, B6')
    # B6 – lập lịch
    pr = []
    for r in runs:
        if r['plan'] and r['plan']['costs']:
            for c in r['plan']['costs']:
                pr.append(dict(luot=r['ten'], don=' ; '.join(f'{a}→{b}' for a, b in r['plan']['orders']), **c,
                               duoc_chon=(c['suc_chua'] == r['plan']['Q'] and c['phuong_phap'] == METHOD_NAME.get(r['plan']['method']))))
    if pr:
        B6 = pd.DataFrame(pr)
        tab(B6, 'B6_lap_lich.csv', 'Quãng đường kế hoạch theo phương pháp lập lịch và sức chứa (cùng bộ đơn, cùng bản đồ học được)',
            '4.2', 'tính lại bằng sim/planning.py trên ma trận khoảng cách Dijkstra của bản đồ đã học')
    # B7, B8 – chi tiết
    lr = [r for r in runs if r['task'] == 'hoc_map']
    if lr:
        tab(pd.DataFrame(lr[-1]['nodes']), 'B7_nut_da_hoc.csv', f"Nút đã học và sai số vị trí ({lr[-1]['ten']})", 'phụ lục / 4.1',
            'vị trí ước lượng sau tối ưu pose graph so với giao lộ thật')
    dl = [dict(luot=r['ten'], **d) for r in runs for d in r['deliveries']]
    if dl:
        tab(pd.DataFrame(dl), 'B8_giao_hang.csv', 'Chi tiết từng lần lấy/trả hàng', 'phụ lục / 4.1', 'ghi khi robot dừng tại trạm')
    tab(_limitations(runs, world), 'B9_han_che.csv', 'Hiện tượng/hạn chế ghi nhận được trong các lượt chạy', '4.3',
        'tự động từ thống kê của robot (B3) và nhật ký sự kiện')

    # ------------------------------------------------------------ hình
    cols = [f'C{i % 10}' for i in range(len(runs))]
    if lr:
        r = lr[-1]
        world = r['world']
        f = Figure(figsize=(7.5, 6))
        ax = f.add_subplot(111)
        for p in world.polylines:
            ax.plot(p[:, 0], p[:, 1], color='#bbb', lw=5, solid_capstyle='butt', zorder=1)
        for a, b in r['edges']:
            ax.plot([a[0], b[0]], [a[1], b[1]], color='#9c27b0', lw=1, ls=':', zorder=2)
        for n in r['nodes']:
            ax.plot(n['x_hoc'], n['y_hoc'], 'o', mfc='none', mec='#9c27b0', ms=10, mew=1.6, zorder=3)
            if np.isfinite(n['x_that']):
                ax.plot([n['x_that'], n['x_hoc']], [n['y_that'], n['y_hoc']], color='r', lw=1, zorder=3)
                ax.plot(n['x_that'], n['y_that'], 'k+', ms=8, zorder=4)
            ax.annotate(f"{n['nut']}" + (f" ({n['tram']})" if n['tram'] else ''), (n['x_hoc'], n['y_hoc']), xytext=(5, 5),
                        textcoords='offset points', fontsize=8, color='#6a1b9a')
        ax.plot([], [], color='#bbb', lw=5, label='line thật'); ax.plot([], [], 'k+', label='giao lộ thật')
        ax.plot([], [], 'o', mfc='none', mec='#9c27b0', label='nút đã học'); ax.plot([], [], 'r-', label='sai số vị trí nút')
        ax.set_aspect('equal'); ax.grid(alpha=0.3); ax.legend(fontsize=8, loc='best'); ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
        m = r['summary']['metrics']
        ax.set_title(f"{r['ten']}: {m.get('map_n_nodes')}/{m.get('map_n_true')} nút, sai số TB {m.get('node_err_mean_mm', 0):.0f} mm", fontsize=10)
        fig_(f, 'H1_ban_do_hoc_duoc.png', 'Bản đồ topo học được so với sân line thật', '4.1', f"lượt {r['ten']}; B7")
        f = Figure(figsize=(11, 4.2))
        ax1, ax2 = f.add_subplot(121), f.add_subplot(122)
        d = r['df']
        for p in world.polylines:
            ax1.plot(p[:, 0], p[:, 1], color='#ddd', lw=4, zorder=1)
        ax1.plot(d.x, d.y, lw=0.9, label='quỹ đạo thật'); ax1.plot(d.x_odo, d.y_odo, lw=0.9, ls='--', label='ước lượng (odometry + neo nút)')
        ax1.set_aspect('equal'); ax1.legend(fontsize=8); ax1.grid(alpha=0.3); ax1.set_xlabel('x [m]'); ax1.set_ylabel('y [m]')
        for c, rr in zip(cols, runs):
            ax2.plot(rr['df'].t - rr['df'].t.iloc[0], rr['df'].odo_err * 1000, lw=0.8, color=c, label=rr['ten'])
        ax2.set_xlabel('t [s]'); ax2.set_ylabel('sai số vị trí ước lượng [mm]'); ax2.grid(alpha=0.3); ax2.legend(fontsize=7)
        fig_(f, 'H2_quy_dao_va_odometry.png', 'Quỹ đạo thật và quỹ đạo ước lượng; sai số định vị theo thời gian', '4.1, 4.2',
             'cột x, y, x_odo, y_odo, odo_err trong chuỗi thời gian')
    f = Figure(figsize=(11, 4.2))
    ax1, ax2 = f.add_subplot(121), f.add_subplot(122)
    box, names = [], []
    for c, r in zip(cols, runs):
        fol = r['df'][r['df']['mode'] == 'follow']
        if len(fol):
            ax1.plot(fol.t - r['df'].t.iloc[0], fol.e_true * 1000, lw=0.6, color=c, label=r['ten'])
            box.append(fol.e_true.to_numpy() * 1000); names.append(r['ten'].split(' ')[0])
    ax1.set_xlabel('t [s]'); ax1.set_ylabel('lệch line thật [mm]'); ax1.grid(alpha=0.3); ax1.legend(fontsize=7)
    if box:
        ax2.boxplot(box, showfliers=False)
        ax2.set_xticks(range(1, len(names) + 1)); ax2.set_xticklabels(names)
    ax2.set_ylabel('lệch line thật [mm]'); ax2.grid(alpha=0.3)
    fig_(f, 'H3_sai_lech_bam_line.png', 'Sai lệch bám line theo thời gian và phân bố theo từng lượt', '4.1, 4.2',
         'cột e_true (khoảng cách thật tới line) khi robot ở chế độ bám line')
    f = Figure(figsize=(11, 4.2))
    ax = f.add_subplot(111)
    for c, r in zip(cols, runs):
        fol = r['df'][r['df']['mode'] == 'follow']
        if len(fol):
            ax.plot(fol.t - r['df'].t.iloc[0], fol.om_cmd, lw=0.6, color=c, label=r['ten'])
    ax.set_xlabel('t [s]'); ax.set_ylabel('lệnh ω [rad/s]'); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    fig_(f, 'H4_lenh_dieu_khien.png', 'Lệnh vận tốc góc của bộ điều khiển (độ êm, độ giật)', '4.2', 'cột om_cmd')
    ks = [(k, t, u) for t, k, u, _ in CRITERIA if k in B3 and B3[k].notna().any() and k not in ('map_correct', 'stops_correct')]
    nrow = (len(ks) + 2) // 3
    f = Figure(figsize=(13, 2.6 * nrow))
    for j, (k, t, u) in enumerate(ks):
        ax = f.add_subplot(nrow, 3, j + 1)
        v = B3[k].astype(float)
        ax.barh(range(len(B3)), v, color=cols)
        ax.set_yticks(range(len(B3))); ax.set_yticklabels([s.split(' ')[0] for s in B3.luot], fontsize=8)
        ax.invert_yaxis(); ax.set_title(f'{t} [{u}]', fontsize=9); ax.grid(axis='x', alpha=0.3)
        for y, val in enumerate(v):
            if np.isfinite(val):
                ax.text(val, y, f' {val:.3g}', va='center', fontsize=7)
    fig_(f, 'H5_so_sanh_chi_tieu.png', 'So sánh các lượt theo từng tiêu chí (L1, L2… xem B3)', '4.2', 'B3')
    if pr:
        f = Figure(figsize=(8, 4))
        ax = f.add_subplot(111)
        B6l = B6[B6.luot == B6.luot.iloc[-1]]
        qs = sorted(B6l.suc_chua.unique())
        for j, (m, c) in enumerate([('FIFO', '#9e9e9e'), ('Tham lam', '#ff9800'), ('Quy hoạch động', '#2e7d32')]):
            vals = [B6l[(B6l.suc_chua == q) & (B6l.phuong_phap == m)].quang_duong_m.iloc[0] for q in qs]
            ax.bar(np.arange(len(qs)) + (j - 1) * 0.27, vals, 0.27, color=c, label=m)
            for x, val in zip(np.arange(len(qs)) + (j - 1) * 0.27, vals):
                ax.text(x, val, f'{val:.2f}', ha='center', va='bottom', fontsize=7)
        ax.set_xticks(range(len(qs))); ax.set_xticklabels([f'sức chứa {q}' for q in qs])
        ax.set_ylabel('quãng đường kế hoạch [m]'); ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)
        ax.set_title(B6l.don.iloc[0], fontsize=9)
        fig_(f, 'H6_lap_lich.png', 'Quãng đường kế hoạch theo phương pháp lập lịch và sức chứa', '4.2', 'B6')

    # ------------------------------------------------------------ phụ lục + tái lập
    for i, r in enumerate(runs):
        r['df'].to_csv(os.path.join(out, 'phu_luc_chuoi_thoi_gian', f'L{i + 1}_{r["task"]}.csv'), index=False, float_format='%.5g')
        with open(os.path.join(out, 'phu_luc_chuoi_thoi_gian', f'L{i + 1}_nhat_ky.txt'), 'w', encoding='utf-8') as fh:
            for t, msg in r['events']:
                fh.write(f'[{t:8.2f}s] {msg}\n')
    rep = dict(ghi_chu='Thiết lập đủ để chạy lại từng lượt trên GUI (cùng sân, cùng seed, cùng tham số).',
               san_line_cac_net=[[list(map(float, p)) for p in s] for s in (strokes or [])],
               luot=[dict(ten=r['ten'], tac_vu=r['task'], thiet_lap=r['summary']['settings'], bo_dieu_khien=r['summary']['controller'],
                          phan_cung=r['summary']['hardware'], v_dat=r['summary']['v_ref'], ke_hoach=r['plan']) for r in runs])
    with open(os.path.join(out, 'thiet_lap_tai_lap.json'), 'w', encoding='utf-8') as fh:
        json.dump(rep, fh, ensure_ascii=False, indent=1, default=str)
    # danh mục hình/bảng (rubric C5: đánh số, chú thích, dẫn nguồn; phân biệt dữ liệu tự đo)
    nb = nh = 0
    lines = ['# Danh mục bảng và hình xuất từ GUI', '',
             f'Xuất lúc {time.strftime("%d/%m/%Y %H:%M")} – {len(runs)} lượt chạy trong phiên GUI. {gui_note}', '',
             'Toàn bộ số liệu do mô phỏng của nhóm tạo ra (tự thực hiện). Vị trí thật của robot và giao lộ chỉ dùng để chấm điểm, '
             'robot không biết các giá trị này. Khi đưa vào báo cáo, đánh lại số theo chương (ví dụ Bảng 4.x, Hình 4.x).', '',
             '| Loại | Số tạm | Tệp | Chú thích | Mục báo cáo | Nguồn / cách tính |', '|---|---|---|---|---|---|']
    for kind, fn, cap, sec, src in catalog:
        if kind == 'Bảng':
            nb += 1; no = f'Bảng {nb}'
        else:
            nh += 1; no = f'Hình {nh}'
        lines.append(f'| {kind} | {no} | `{fn}` | {cap} | {sec} | {src} |')
    lines += ['', '## Các lượt chạy', ''] + [f'- **{r["ten"]}**: {r["summary"]["settings"].get("scenario")}, tốc độ đặt '
                                              f'{r["summary"]["v_ref"]:.2f} m/s, seed {r["summary"]["settings"].get("seed")}' for r in runs]
    lines += ['', '## Gợi ý dùng trong báo cáo', '',
              '- Mục 3.1: B2 là tiêu chí đánh giá; nêu rõ đầu vào (sân line, trạm, đơn) và đầu ra (bản đồ, kế hoạch, quỹ đạo).',
              '- Mục 3.4: B1 là bảng thông số mô phỏng; thiet_lap_tai_lap.json để người đọc kiểm tra lại.',
              '- Mục 4.1: B3, H1–H4 là minh chứng kết quả.',
              '- Mục 4.2: B4, H5 so sánh các bộ điều khiển/cấu hình theo cùng tiêu chí; B6, H6 so sánh phương pháp lập lịch; '
              'B5 đối chiếu mục tiêu.',
              '- Mục 4.3: B9 liệt kê hạn chế quan sát được và nguyên nhân có thể.',
              '- Phụ lục: B7, B8 và thư mục phu_luc_chuoi_thoi_gian.']
    with open(os.path.join(out, 'DANH_MUC_HINH_BANG.md'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')
    return out
