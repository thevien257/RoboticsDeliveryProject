"""Bộ thí nghiệm cho báo cáo: chạy toàn bộ, lưu đệm từng lượt, xuất bảng/hình tự động.

Dùng:
  python experiments/campaign.py              # chạy E0 → E4 rồi xuất báo cáo (lượt đã có trong đệm thì bỏ qua)
  python experiments/campaign.py E0 E1        # chỉ chạy các giai đoạn này (E1, E3 cần E0 để có PID*, LQR*)
  python experiments/campaign.py --report     # chỉ xuất lại bảng/hình từ dữ liệu đã có
  python experiments/campaign.py --nhanh      # bản rút gọn (ít tham số, ít seed) để kiểm tra nhanh quy trình

Thiết kế (xem experiments/campaign_config.json):
  E0  Chỉnh định: lưới Kp×Ki×Kd cho PID và lưới giới hạn Bryson cho LQR; mỗi bộ tham số được thử bằng đáp ứng bước
      (vọt lố, xác lập, dao động còn lại), đường hỗn hợp ở hai tốc độ (RMS, mất line, độ giật) và đường tròn
      (sai lệch tĩnh). Luật chọn: không mất line, hoàn thành, vọt lố và dao động còn lại trong giới hạn -> RMS nhỏ nhất.
      Dữ liệu lưới đồng thời cho thấy ảnh hưởng của từng tham số (Kp, Ki, Kd; Q, R).
  E1  PID* so với LQR* trên đường KHÔNG dùng khi chỉnh định (sóng, tròn R = 0,15 / 0,5 m), nhiều tốc độ, nhiều mức nhiễu
      cảm biến, động cơ lệch, chở hàng; 5 seed mỗi điều kiện -> trung bình ± độ lệch chuẩn, khoảng tin cậy 95 %, kiểm định Welch.
  E2  Học map: 2 sân × 4 mức sai số bánh × 4 cấu hình (IMU/encoder, nhận dạng nút, pose graph).
  E3  Tình huống khi giao hàng: 8 kịch bản × PID*/LQR*.
  E4  Lập lịch: FIFO / tham lam / quy hoạch động × sức chứa × số đơn.
Mỗi lượt có giới hạn thời gian mô phỏng và bắt lỗi; tiến độ in liên tục; dữ liệu ghi dần vào results/campaign/raw/*.csv
nên dừng giữa chừng rồi chạy lại sẽ tiếp tục từ chỗ dừng.
"""
import os, sys, json, time, traceback, itertools, argparse
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')         # mỗi tiến trình 1 luồng BLAS: tránh tranh chấp khi chạy song song
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from multiprocessing import Pool
from sim.world import test_track, warehouse_map, grid_map
from sim.robot import RobotParams
from sim.control import PID, LQR
from sim.follow import TrackRun
from sim.agent import Agent
from sim.scenarios import NOISE_LEVELS, map_score

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'results', 'campaign')
RAW = os.path.join(OUT, 'raw')
CFG = json.load(open(os.path.join(HERE, 'campaign_config.json'), encoding='utf-8'))


# ====================================================================== bộ điều khiển
def make_ctrl(ctrl, p):
    if ctrl.startswith('PID'):
        return PID(Kp=p['Kp'], Ki=p['Ki'], Kd=p['Kd'])
    z = p.get('z_max_mm_s', 0) or 0
    return LQR(model='dyn', tau=RobotParams().tau, d_max=p['e_y_max_mm'] / 1000, psi_max=np.radians(p['e_theta_max_deg']),
               u_max=p['omega_max'], integral=z > 0, **({'z_max': z / 1000} if z > 0 else {}))


def structure(ctrl, p):
    """Cấu trúc bộ điều khiển: P, PI, PD, PID hoặc LQR, LQI."""
    if ctrl.startswith('PID'):
        return 'P' + ('I' if p['Ki'] > 0 else '') + ('D' if p['Kd'] > 0 else '')
    return 'LQI' if (p.get('z_max_mm_s', 0) or 0) > 0 else 'LQR'


# ====================================================================== các lượt (chạy trong tiến trình con)
def job_ctrl(a):
    """Một phép thử bộ điều khiển: đáp ứng bước hoặc chạy hết một đường."""
    try:
        c = make_ctrl(a['ctrl'], a['p'])
        prm = RobotParams(sens_noise=a.get('noise', 0.03), motor_bias=a.get('bias', 0.0), loaded=a.get('loaded', False))
        out = {}
        if a['test'] in ('buoc', 'buoc_lech_dong_co'):
            w = test_track('straight_long')
            P = w.polylines[0]
            d = P[1] - P[0]
            th = np.arctan2(d[1], d[0])
            nrm = np.array([-np.sin(th), np.cos(th)])
            x0 = P[0] + a['offset'] * nrm
            tr = TrackRun(w, c, v_ref=a['v'], seed=a['seed'], t_max=8.0 if a['test'] == 'buoc_lech_dong_co' else 5.0,
                          params=prm, pose=(x0[0], x0[1], th))
            while tr.tick():
                pass
            t = np.array(tr.ts) - tr.ts[0]
            s = (np.c_[tr.xs, tr.ys] - P[0]) @ nrm
            big = np.where(np.abs(s) > 0.002)[0]
            om = np.array(tr.oms)
            out = dict(overshoot=100 * max(0.0, -float(s.min())) / a['offset'], settle=float(t[big[-1]]) if len(big) else 0.0,
                       residual=float(s[-100:].max() - s[-100:].min()) * 1000 / 2, ess=float(s[-100:].mean()) * 1000,
                       lost=tr.lost_events, jerk=float(np.sqrt(np.mean(np.diff(om) ** 2))) / 0.01 if len(om) > 2 else 0.0)
        else:
            tr = TrackRun(test_track(a['track']), c, v_ref=a['v'], seed=a['seed'], params=prm)
            while tr.tick():
                pass
            m = tr.metrics()
            e = np.array(tr.eys, float)
            out = dict(rms=m['rms'], p95=m['p95'], emax=m['emax'], lost=m['lost'], finished=bool(m['finished']),
                       jerk=m['dom'] / 0.01, om_rms=m['om_rms'], sat=m['sat'], t_run=m['time'],
                       ess_meas=float(np.nanmean(e[len(e) // 2:])) * 1000 if len(e) else np.nan)
            if a['track'].startswith('circle'):
                # sai lệch tĩnh THẬT trên cung tròn: khoảng cách tới tâm − R (dương: lệch ra ngoài), nửa sau của vòng
                R = float(a['track'].split('_')[1])
                xy = np.c_[tr.xs, tr.ys]
                rr = np.hypot(xy[:, 0] - 0.4, xy[:, 1] - R) - R
                on = xy[:, 0] > 0.42
                out['ess'] = float(np.mean(rr[on][len(rr[on]) // 2:])) * 1000 if on.sum() > 10 else np.nan
        return out
    except Exception:
        return dict(error=traceback.format_exc()[-500:])


def make_agent(mapname, lvl, seed, ctrl, gyro=True, assoc='mht', posegraph=True, v=0.3):
    rng = np.random.default_rng(100 + seed)
    base = NOISE_LEVELS[lvl]
    kw = dict(r_err_l=base['r_err_l'] * rng.uniform(0.5, 1) * rng.choice([-1, 1]),
              r_err_r=base['r_err_r'] * rng.uniform(0.5, 1) * rng.choice([-1, 1]),
              b_err=base['b_err'] * rng.uniform(0.5, 1) * rng.choice([-1, 1]), slip=base['slip'],
              gyro=gyro, gyro_bias=float(rng.uniform(-0.015, 0.015)))
    w = warehouse_map() if mapname == 'kho' else grid_map()
    start = (0.05, 0.0, 0.0) if mapname == 'kho' else (-0.45, -0.235, np.pi / 2)
    return Agent(w, start, ctrl=ctrl, assoc=assoc, params=RobotParams(**kw), seed=seed, v_ref=v, posegraph=posegraph), w


def job_map(a):
    """E2: tự học map một sân."""
    try:
        opt = {'encoder + gyro (đề xuất)': dict(gyro=True), 'chỉ encoder': dict(gyro=False),
               'gyro, nhận dạng đơn giản': dict(gyro=True, assoc='naive'),
               'gyro, không pose graph': dict(gyro=True, posegraph=False)}[a['cfg']]
        ag, w = make_agent(a['map'], a['level'], a['seed'], make_ctrl('PID', a['p']), **opt)
        ag.task = ag.explore(t_limit=600)
        while ag.tick():
            pass
        sc = map_score(w, ag.map)
        ne = [np.hypot(*(n.pos - np.array(w.nodes[n.truth]))) for n in ag.map.nodes.values() if n.truth in w.nodes]
        tr = np.array(ag.traj)
        oe = np.hypot(tr[:, 1] - tr[:, 3], tr[:, 2] - tr[:, 4])
        return dict(**sc, wrong_merge=ag.stats['wrong_merge'], duplicate=ag.stats['duplicate'], scans=ag.stats['scans'],
                    active_checks=ag.stats['active_checks'], lost=ag.stats['lost'],
                    node_err_mean=float(np.mean(ne)) * 1000 if ne else np.nan, node_err_max=float(np.max(ne)) * 1000 if ne else np.nan,
                    odo_err_mean=float(oe.mean()) * 1000, odo_err_max=float(oe.max()) * 1000,
                    t_explore=float(ag.rob.t), dist=float(ag.rob.dist_true), failed=bool(ag.failed))
    except Exception:
        return dict(error=traceback.format_exc()[-500:])


def scen_group(jobs):
    """E3: học map MỘT lần cho (bộ điều khiển, seed), rồi mỗi kịch bản chạy trên một bản sao robot + sân đã học."""
    import copy
    from sim.scenarios import apply_scen
    a0 = jobs[0]
    t0 = time.time()
    try:
        ag0, w0 = make_agent('kho', 'Trung bình', a0['seed'], make_ctrl(a0['ctrl'], a0['p']))
        ag0.task = ag0.explore(t_limit=600)
        while ag0.tick():
            pass
        ok_map = map_score(w0, ag0.map)['correct']
        ag0.stations = {n.truth: n.id for n in ag0.map.nodes.values() if n.degree == 1}
    except Exception:
        err = traceback.format_exc()[-500:]
        return [(a, dict(error=err)) for a in jobs]
    t_map = time.time() - t0
    out = []
    for a in jobs:
        t1 = time.time()
        try:
            F = w0.field()
            ag = copy.deepcopy(ag0, {id(F): F})          # bản sao robot + bản đồ đã học (dùng chung lưới khoảng cách)
            w = ag.world
            apply_scen(ag, w, a['scen'])
            s0, tt0, st0 = ag.rob.dist_true, ag.rob.t, dict(ag.stats)
            n0 = len(ag.trace)
            ag.task = ag.deliver([('A', 'C'), ('B', 'E'), ('D', 'H')], method='dp', Q=1)
            while ag.tick():
                if ag.rob.t - tt0 > 600:
                    break
            dl = ag.delivery
            tr = np.array([(r[2], r[3]) for r in ag.trace[n0:] if r[1] == 'follow'])
            e = w.dist(tr) * 1000 if len(tr) else np.array([np.nan])
            r = dict(map_ok=bool(ok_map), done=len(dl), success=float(len(dl) == 6 and all(d['correct'] for d in dl)),
                     t=ag.rob.t - tt0, dist=ag.rob.dist_true - s0,
                     stop_err=float(np.mean([d['err_mm'] for d in dl])) if dl else np.nan,
                     stop_err_max=float(np.max([d['err_mm'] for d in dl])) if dl else np.nan,
                     collisions=ag.rob.collisions, waits=ag.stats['obstacle_stops'] - st0['obstacle_stops'],
                     wait_time=ag.stats['wait_time'] - st0['wait_time'], replans=ag.stats['replans'] - st0['replans'],
                     lost=ag.stats['lost'] - st0['lost'], track_rms=float(np.sqrt(np.nanmean(e ** 2))))
        except Exception:
            r = dict(error=traceback.format_exc()[-500:])
        r['wall_s'] = time.time() - t1 + t_map / len(jobs)
        out.append((a, r))
    return out


# ====================================================================== động cơ chạy + đệm
def key_of(a):
    return json.dumps(a, sort_keys=True, ensure_ascii=False)


def _wrap(a):
    fn = {'ctrl': job_ctrl, 'map': job_map}[a['_kind']]
    t = time.time()
    r = fn(a)
    r['wall_s'] = time.time() - t
    return [(a, r)]


def _track_tmax(track, v):
    w = test_track(track)
    L = sum(float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1))) for p in w.polylines)
    return 1.6 * L / max(v, 0.05) + 4.0


def batch_group(jobs):
    """Chạy một nhóm phép thử bộ điều khiển (cùng loại phép thử, cùng đường, cùng họ PID/LQR) bằng mô phỏng vector hoá."""
    from sim.batch import BatchTrack, BatchPID, BatchLQR, TS
    t0 = time.time()
    try:
        a0 = jobs[0]
        test, track = a0['test'], a0['track']
        w = test_track(track)
        v = np.array([a['v'] for a in jobs], float)
        if a0['ctrl'].startswith('PID'):
            ctrl = BatchPID([a['p']['Kp'] for a in jobs], [a['p']['Ki'] for a in jobs], [a['p']['Kd'] for a in jobs])
        else:
            ctrl = BatchLQR([a['p'] for a in jobs], v)
        kw = dict(seed=[a['seed'] for a in jobs], sens_noise=[a.get('noise', 0.03) for a in jobs],
                  motor_bias=[a.get('bias', 0.0) for a in jobs], loaded=[a.get('loaded', False) for a in jobs])
        if test.startswith('buoc'):
            P = w.polylines[0]
            d = P[1] - P[0]
            th = np.arctan2(d[1], d[0])
            nrm = np.array([-np.sin(th), np.cos(th)])
            x0 = P[0] + a0['offset'] * nrm
            bt = BatchTrack(w, ctrl, v, pose=(x0[0], x0[1], th), t_max=8.0 if test == 'buoc_lech_dong_co' else 5.0, **kw).run()
        else:
            bt = BatchTrack(w, ctrl, v, t_max=_track_tmax(track, v.min()), **kw).run()
        M = bt.metrics()
        E, OM, X, Y, EY = (np.array(z) for z in (bt.log_e, bt.log_om, bt.log_x, bt.log_y, bt.log_ey))
        out = []
        for j, (a, m) in enumerate(zip(jobs, M)):
            ok = ~np.isnan(E[:, j])
            om = OM[ok, j]
            if test.startswith('buoc'):
                s = (np.c_[X[ok, j], Y[ok, j]] - P[0]) @ nrm
                t = np.arange(len(s)) * TS
                big = np.where(np.abs(s) > 0.002)[0]
                r = dict(overshoot=100 * max(0.0, -float(s.min())) / a['offset'], settle=float(t[big[-1]]) if len(big) else 0.0,
                         residual=float(s[-100:].max() - s[-100:].min()) * 1000 / 2, ess=float(s[-100:].mean()) * 1000,
                         lost=m['lost'], jerk=float(np.sqrt(np.mean(np.diff(om) ** 2))) / TS if len(om) > 2 else 0.0)
            else:
                ey = EY[ok, j]
                r = dict(rms=m['rms'], p95=m['p95'], emax=m['emax'], lost=m['lost'], finished=m['finished'],
                         jerk=m['dom'] / TS, om_rms=m['om_rms'], sat=m['sat'], t_run=m['time'],
                         ess_meas=float(np.nanmean(ey[len(ey) // 2:])) * 1000 if len(ey) else np.nan)
                if track.startswith('circle'):
                    R = float(track.split('_')[1])
                    xy = np.c_[X[ok, j], Y[ok, j]]
                    rr = np.hypot(xy[:, 0] - 0.4, xy[:, 1] - R) - R
                    on = xy[:, 0] > 0.42
                    r['ess'] = float(np.mean(rr[on][len(rr[on]) // 2:])) * 1000 if on.sum() > 10 else np.nan
            r['wall_s'] = (time.time() - t0) / len(jobs)
            out.append((a, r))
        return out
    except Exception:
        err = traceback.format_exc()[-500:]
        return [(a, dict(error=err)) for a in jobs]


def _wrap_any(task):
    if isinstance(task, list):
        return scen_group(task) if task[0]['_kind'] == 'scen' else batch_group(task)
    return _wrap(task)


def make_tasks(todo, chunk=128):
    """Gom các phép thử bộ điều khiển thành nhóm chạy vector hoá; các lượt học map / tình huống chạy riêng."""
    groups, single, scen = {}, [], {}
    for a in todo:
        if a['_kind'] == 'ctrl':
            groups.setdefault((a['ctrl'][:3], a['test'], a['track'], a.get('offset')), []).append(a)
        elif a['_kind'] == 'scen':
            scen.setdefault((a['ctrl'], a['seed']), []).append(a)
        else:
            single.append(a)
    long_ = []
    for g in scen.values():                    # mỗi (bộ điều khiển, seed): học map 1 lần, chia 2 kịch bản / phần để cân tải
        long_ += [g[i:i + 2] for i in range(0, len(g), 2)]
    tasks = []
    for g in groups.values():
        tasks += [g[i:i + chunk] for i in range(0, len(g), chunk)]
    tasks.sort(key=lambda g: -len(g))
    return long_ + single + tasks              # việc dài trước, việc ngắn sau (cân tải giữa các lõi)


def run_stage(name, jobs, procs):
    """Chạy một giai đoạn (xem run_stages)."""
    run_stages({name: jobs}, procs)


def run_stages(stage_jobs, procs):
    """Chạy các lượt chưa có trong đệm của một hoặc nhiều giai đoạn trong CÙNG một nhóm tiến trình (không để lõi rảnh
    khi một giai đoạn sắp xong); ghi dần mỗi lượt một dòng JSON vào raw/<giai đoạn>.jsonl; in tiến độ và lỗi ngay."""
    os.makedirs(RAW, exist_ok=True)
    todo, paths = [], {}
    for name, jobs in stage_jobs.items():
        path = os.path.join(RAW, f'{name}.jsonl')
        paths[name] = path
        done = set()
        if os.path.exists(path):
            with open(path, encoding='utf-8') as f:
                done = {json.loads(l)['key'] for l in f if l.strip()}
        tj = [dict(j, stage=name) for j in jobs if key_of(dict(j, stage=name)) not in done]
        print(f'[{name}] {len(jobs)} lượt, đã có {len(jobs) - len(tj)}, cần chạy {len(tj)}', flush=True)
        todo += tj
    if not todo:
        return
    label = '+'.join(stage_jobs)
    t0, n_err = time.time(), 0
    buf = {n: [] for n in stage_jobs}
    cnt = {n: 0 for n in stage_jobs}
    tot = {n: sum(1 for a in todo if a['stage'] == n) for n in stage_jobs}
    step = max(1, len(todo) // 20)

    def flush(n):                              # mỗi lượt một dòng JSON: các phép thử có cột khác nhau vẫn ghi chung được
        if buf[n]:
            with open(paths[n], 'a', encoding='utf-8') as f:
                for r in buf[n]:
                    f.write(json.dumps(r, ensure_ascii=False, default=float) + '\n')
            buf[n].clear()
    i, last = 0, 0
    with Pool(procs) as P:
        for res in P.imap_unordered(_wrap_any, make_tasks(todo)):
            for a, r in res:
                n = a['stage']
                row = {k: v for k, v in a.items() if k not in ('p', '_kind')}
                row.update({f'p_{k}': v for k, v in a.get('p', {}).items()})
                row.update(r)
                row['key'] = key_of(a)
                buf[n].append(row)
                i += 1
                cnt[n] += 1
                if 'error' in r:
                    n_err += 1
                    if n_err <= 3:
                        print(f'[{n}] LỖI: {r["error"][-300:]}', flush=True)
                if cnt[n] == tot[n]:
                    flush(n)
                    print(f'[{n}] XONG {tot[n]} lượt, {time.time() - t0:.1f} s', flush=True)
            for n in buf:
                if len(buf[n]) >= 25:
                    flush(n)
            if i - last >= step or i == len(todo):
                last = i
                el = time.time() - t0
                print(f'[{label}] {i}/{len(todo)} lượt, {el:.0f} s, còn khoảng {el / i * (len(todo) - i):.0f} s'
                      + (f', {n_err} lỗi' if n_err else ''), flush=True)
    for n in buf:
        flush(n)


def load(name):
    p = os.path.join(RAW, f'{name}.jsonl')
    if not os.path.exists(p):
        p = os.path.join(RAW, f'{name}.csv')
        if not os.path.exists(p):
            return None
        d = pd.read_csv(p)
    else:
        d = pd.read_json(p, lines=True)
    return d[d['error'].isna()] if 'error' in d.columns else d


# ====================================================================== E0 chỉnh định
def grid_params(algo, quick=False):
    g = CFG['E0'][algo]
    keys = list(g)
    vals = [v[::2] if quick and len(v) > 3 else v for v in g.values()]
    return [dict(zip(keys, c)) for c in itertools.product(*vals)]


def e0_jobs(quick=False):
    jobs = []
    for algo in ['PID', 'LQR']:
        for p in grid_params(algo, quick):
            for t in CFG['E0']['phep_thu']:
                for s in (t['seeds'][:1] if quick else t['seeds']):
                    j = dict(_kind='ctrl', stage='E0', ctrl=algo, p=p, v=t['v'], seed=s)
                    if t['loai'].startswith('buoc'):
                        j.update(test=t['loai'], offset=t['lech_m'], track='straight_long', bias=t.get('dong_co_lech', 0.0),
                                 loaded=t.get('dong_co_lech', 0.0) > 0)
                    else:
                        j.update(test='duong', track=t['duong'], loaded=t.get('cho_hang', False))
                    jobs.append(j)
    return jobs


_E0_CACHE = {}


def e0_summary(algo):
    """Gộp các phép thử theo bộ tham số (vector hoá bằng groupby); đánh dấu khả thi; xếp hạng
    (cấu trúc đầy đủ trước, ít ràng buộc vi phạm trước, RMS nhỏ trước). Kết quả được nhớ theo thời điểm sửa tệp dữ liệu."""
    path = os.path.join(RAW, 'E0.jsonl')
    if not os.path.exists(path):
        return None
    key = (algo, path, os.path.getmtime(path))
    if key in _E0_CACHE:
        return _E0_CACHE[key].copy()
    d = load('E0')
    d = d[d.ctrl == algo].copy()
    pk = [c for c in d.columns if c.startswith('p_') and d[c].notna().any()]
    circ = d.track.astype(str).str.startswith('circle')
    d['loai'] = np.where(d.test == 'buoc', 'st', np.where(d.test == 'buoc_lech_dong_co', 'sb', np.where(circ, 'ci', 'tk')))
    d['ess_abs'] = d['ess'].abs() if 'ess' in d else np.nan
    g = d.groupby(pk)
    st, sb, tk, ci = (d[d.loai == k].groupby(pk) for k in ('st', 'sb', 'tk', 'ci'))
    s = pd.DataFrame({'overshoot': st.overshoot.mean(), 'settle': st.settle.mean(), 'residual': st.residual.max(),
                      'ess_bias': sb.ess_abs.mean(), 'rms': tk.rms.mean(), 'p95': tk.p95.mean(), 'jerk': tk.jerk.mean(),
                      'lost': g.lost.sum().astype(int),
                      'finished': tk.finished.apply(lambda x: bool(x.astype(bool).all())) & ci.finished.apply(lambda x: bool(x.astype(bool).all())),
                      'ess_circle': ci.ess_abs.mean(), 'rms_circle': ci.rms.mean()}).reset_index()
    s.columns = [c[2:] if c.startswith('p_') else c for c in s.columns]
    s.insert(len(pk), 'cau_truc', [structure(algo, r) for r in s[[c[2:] for c in pk]].to_dict('records')])
    rule = CFG['E0']['luat_chon']
    viol = ((s.lost > rule['mat_line_toi_da']).astype(int) + ((~s.finished.astype(bool)) & rule['phai_hoan_thanh']).astype(int)
            + (s.overshoot > rule['vot_lo_toi_da_pct']).astype(int) + (s.residual > rule['dao_dong_con_lai_toi_da_mm']).astype(int)
            + (~(s.ess_bias <= rule['sai_lech_tinh_toi_da_mm'])).astype(int))
    s['so_vi_pham'] = viol.astype(int)
    s['kha_thi'] = s.so_vi_pham == 0
    s['day_du'] = s.cau_truc.isin(['PID', 'LQI'])
    s = s.sort_values(['day_du', 'so_vi_pham', 'rms', 'jerk'], ascending=[False, True, True, True]).reset_index(drop=True)
    s.insert(0, 'hang', np.arange(1, len(s) + 1))
    _E0_CACHE.clear()
    _E0_CACHE[key] = s
    return s.copy()


def tuned():
    """Tham số tốt nhất của E0 (PID*, LQR*): cấu trúc đầy đủ (PID; LQI), khả thi, RMS nhỏ nhất. Ghi tuned.json."""
    best = {}
    for algo in ['PID', 'LQR']:
        s = e0_summary(algo)
        if s is None or s.empty:
            continue
        r = s.iloc[0]
        keys = [k for k in CFG['E0'][algo]]
        best[algo] = dict(params={k: float(r[k]) for k in keys}, cau_truc=r.cau_truc, kha_thi=bool(r.kha_thi),
                          metrics={k: float(r[k]) for k in ['overshoot', 'settle', 'residual', 'ess_bias', 'ess_circle', 'rms', 'p95',
                                                            'jerk', 'lost']})
    if best:
        with open(os.path.join(OUT, 'tuned.json'), 'w', encoding='utf-8') as f:
            json.dump(best, f, ensure_ascii=False, indent=1)
    return best


# ====================================================================== E1 so sánh PID* với LQR*
def e1_jobs(best, quick=False):
    c = CFG['E1']
    seeds = c['seeds'][:2] if quick else c['seeds']
    jobs = []
    for algo in ['PID', 'LQR']:
        p = best[algo]['params']
        base = dict(_kind='ctrl', stage='E1', ctrl=algo + '*', p=p, test='duong')
        for tr, v, s in itertools.product(c['duong_toc_do']['duong'], c['duong_toc_do']['v'], seeds):
            jobs.append(dict(base, group='đường × tốc độ', track=tr, v=v, seed=s, noise=0.03, bias=0.0, loaded=False))
        n = c['nhieu_cam_bien']
        for lv, s in itertools.product(n['muc'], seeds):
            jobs.append(dict(base, group='nhiễu cảm biến', track=n['duong'], v=n['v'], seed=s, noise=lv, bias=0.0, loaded=False))
        n = c['nhieu_loan']
        for b, l, s in itertools.product(n['dong_co_lech'], n['cho_hang'], seeds):
            jobs.append(dict(base, group='nhiễu loạn', track=n['duong'], v=n['v'], seed=s, noise=0.03, bias=b, loaded=l))
    return jobs


def e2_jobs(best, quick=False):
    c = CFG['E2']
    lv = c['muc_sai_so_banh'][1::2] if quick else c['muc_sai_so_banh']
    seeds = c['seeds'][:1] if quick else c['seeds']
    return [dict(_kind='map', stage='E2', map=m, level=l, cfg=g, seed=s, p=best['PID']['params'])
            for m in c['san'] for l in lv for g in c['cau_hinh'] for s in seeds]


def e3_jobs(best, quick=False):
    c = CFG['E3']
    sc = c['kich_ban'][:3] if quick else c['kich_ban']
    seeds = c['seeds'][:1] if quick else c['seeds']
    return [dict(_kind='scen', stage='E3', scen=k, ctrl=b, p=best[b[:3]]['params'], seed=s)
            for k in sc for b in c['bo_dieu_khien'] for s in seeds]


def e4_run(quick=False):
    """Lập lịch trên đồ thị thật của sân kho (nhanh, chạy trong tiến trình chính)."""
    from sim.mapping import TopoMap
    from sim.planning import dist_matrix, _start_dist, plan_stops
    c = CFG['E4']
    w = warehouse_map()
    tm = TopoMap()
    ids = {k: tm.add_node(P, [0.0] * 4, k).id for k, P in w.nodes.items()}
    used = {k: 0 for k in w.nodes}
    for u, v, pid in w.edges:
        L = float(np.sum(np.linalg.norm(np.diff(w.polylines[pid], axis=0), axis=1)))
        tm.connect(ids[u], used[u], ids[v], used[v], L)
        used[u] += 1
        used[v] += 1
    names = ['A', 'B', 'C', 'D', 'E', 'H']
    nid = [ids[n] for n in names]
    D = dist_matrix(tm, nid)
    d0 = _start_dist(tm, ids['H'], nid)
    rows = []
    reps = 5 if quick else c['lap_lai']
    for n in c['so_don']:
        for rep in range(reps):
            rng = np.random.default_rng(1000 * n + rep)
            od = [tuple(int(x) for x in rng.choice(6, 2, replace=False)) for _ in range(n)]
            for q in c['suc_chua']:
                for m, lab in [('fifo', 'FIFO'), ('greedy', 'Tham lam'), ('dp', 'Quy hoạch động')]:
                    t = time.perf_counter()
                    _, cost = plan_stops(D, od, d0, q, m)
                    rows.append(dict(n=n, rep=rep, Q=q, method=lab, cost=cost, ms=(time.perf_counter() - t) * 1000))
    os.makedirs(RAW, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(RAW, 'E4.csv'), index=False)
    print(f'[E4] XONG {len(rows)} dòng', flush=True)


# ====================================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stages', nargs='*', default=['E0', 'E1', 'E2', 'E3', 'E4'])
    ap.add_argument('--report', action='store_true', help='chỉ xuất lại bảng/hình')
    ap.add_argument('--nhanh', action='store_true', help='bản rút gọn để kiểm tra quy trình')
    ap.add_argument('--procs', type=int, default=CFG.get('so_tien_trinh', 4))
    a = ap.parse_args()
    global OUT, RAW
    if a.nhanh:
        OUT = os.path.join(HERE, '..', 'results', 'campaign_nhanh')
        RAW = os.path.join(OUT, 'raw')
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    if not a.report:
        if 'E0' in a.stages:
            run_stage('E0', e0_jobs(a.nhanh), a.procs)
            print(f'--- E0 xong sau {time.time() - t0:.1f} s', flush=True)
        best = tuned()
        if best:
            for k, b in best.items():
                m = b['metrics']
                print(f'{k}* ({b["cau_truc"]}): {b["params"]}  khả thi = {b["kha_thi"]}, RMS {m["rms"]:.2f} mm, '
                      f'vọt lố {m["overshoot"]:.0f} %, sai lệch tĩnh {m["ess_bias"]:.2f} mm (tròn {m["ess_circle"]:.2f})', flush=True)
        need = [s for s in ['E1', 'E2', 'E3'] if s in a.stages]
        if need and len(best) < 2:
            print('Chưa có kết quả E0 cho cả PID và LQR – chạy E0 trước.', flush=True)
            need = []
        if need:
            run_stages({s: {'E1': e1_jobs, 'E2': e2_jobs, 'E3': e3_jobs}[s](best, a.nhanh) for s in need}, a.procs)
        if 'E4' in a.stages:
            e4_run(a.nhanh)
    t_r = time.time()
    print(f'--- chạy thí nghiệm xong sau {t_r - t0:.1f} s, đang xuất báo cáo…', flush=True)
    from campaign_report import build_report
    out = build_report(OUT, CFG, cmd='python experiments/campaign.py ' + ' '.join(sys.argv[1:]), t_run=time.time() - t0)
    print(f'BÁO CÁO XONG: {os.path.relpath(out)} (xuất {time.time() - t_r:.1f} s, tổng {time.time() - t0:.1f} s)', flush=True)


if __name__ == '__main__':
    main()
