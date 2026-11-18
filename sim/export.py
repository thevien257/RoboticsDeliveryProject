"""Chuỗi thời gian và chỉ tiêu tổng hợp của một lượt chạy (dùng cho sim/report_export.py).

timeseries(): mỗi chu kỳ 10 ms: t, chế độ, tư thế thật (x, y, θ), tư thế odometry (x̂, ŷ, θ̂), lệnh v, ω,
              tốc độ bánh, sai lệch đo e_y, sai lệch thật tới line gần nhất, sai số odometry.
summarize():  thiết lập (bộ điều khiển, tham số, phần cứng, tình huống) + chỉ tiêu tổng hợp.
"""
import numpy as np
import pandas as pd
from .scenarios import map_score

COLS = ['t', 'mode', 'x', 'y', 'th', 'x_odo', 'y_odo', 'th_odo', 'v_cmd', 'om_cmd', 'w_left', 'w_right', 'e_meas']


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def timeseries(agent, world, start=0):
    tr = agent.trace[start:]
    if not tr:
        return pd.DataFrame(columns=COLS)
    df = pd.DataFrame(tr, columns=COLS)
    df['e_true'] = world.dist(df[['x', 'y']].to_numpy())
    df['odo_err'] = np.hypot(df.x - df.x_odo, df.y - df.y_odo)
    df['th_err_deg'] = np.degrees(np.abs(_wrap(df.th - df.th_odo)))
    return df


def controller_info(ctrl):
    if type(ctrl).__name__ == 'PID':
        return dict(controller='PID', Kp=ctrl.Kp, Ki=ctrl.Ki, Kd=ctrl.Kd)
    d = dict(controller='LQR', model=ctrl.model, Q=[float(q) for q in ctrl.Q], R=float(ctrl.R))
    try:
        d['K'] = [float(k) for k in ctrl.gain(0.3)]
    except Exception:
        pass
    return d


def summarize(agent, world, df, task, settings):
    p, rob, st = agent.p, agent.rob, agent.stats
    fol = df[df['mode'] == 'follow'] if len(df) else df
    s = dict(task=task, time_s=float(df.t.iloc[-1] - df.t.iloc[0]) if len(df) > 1 else 0.0,
             distance_m=float(np.sum(np.abs(df.v_cmd)) * 0.01) if len(df) else 0.0)
    if len(fol):
        e = fol.e_true.to_numpy() * 1000
        s.update(track_rms_mm=float(np.sqrt(np.mean(e ** 2))), track_p95_mm=float(np.percentile(e, 95)),
                 track_max_mm=float(e.max()), om_rms=float(np.sqrt(np.mean(fol.om_cmd ** 2))),
                 om_jerk=float(np.sqrt(np.mean(np.diff(fol.om_cmd.to_numpy()) ** 2)) / 0.01) if len(fol) > 2 else 0.0)
    if len(df):
        s.update(odo_err_mean_mm=float(df.odo_err.mean() * 1000), odo_err_max_mm=float(df.odo_err.max() * 1000),
                 odo_err_end_mm=float(df.odo_err.iloc[-1] * 1000), th_err_max_deg=float(df.th_err_deg.max()))
    if agent.map.nodes:
        sc = map_score(world, agent.map)
        ne = [np.hypot(*(n.pos - np.array(world.nodes[n.truth]))) * 1000 for n in agent.map.nodes.values() if n.truth in world.nodes]
        s.update({f'map_{k}': v for k, v in sc.items()})
        if ne:
            s.update(node_err_mean_mm=float(np.mean(ne)), node_err_max_mm=float(np.max(ne)))
    dl = [d for d in agent.delivery if d['t'] >= (df.t.iloc[0] if len(df) else 0)]
    if dl:
        s.update(stops=len(dl), stop_err_mean_mm=float(np.mean([d['err_mm'] for d in dl])),
                 stop_err_max_mm=float(np.max([d['err_mm'] for d in dl])),
                 stops_correct=float(np.mean([d['correct'] for d in dl])))
        if getattr(agent, 'plan_cost', None) is not None:
            s.update(plan_cost_m=float(agent.plan_cost), plan=str(getattr(agent, 'plan_text', '')))
    s.update({f'stat_{k}': (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in st.items()},
             collisions=int(rob.collisions), b_est_mm=float(rob.b_est * 1000), b_true_mm=float(p.b * (1 + p.b_err) * 1000),
             eps_est=float(rob.eps_est), eps_true=float(p.r_err_r - p.r_err_l),
             gyro_bias_est=float(rob.gbias_est), gyro_bias_true=float(p.gyro_bias if p.gyro else 0.0))
    hw = dict(gyro=bool(p.gyro), sens_noise=p.sens_noise, latency_s=p.latency, tau_s=p.tau, x_front_m=p.x_front,
              r_err_l=p.r_err_l, r_err_r=p.r_err_r, b_err=p.b_err, slip=p.slip, w_max=p.w_max, loaded=bool(p.loaded))
    return dict(settings=settings, controller=controller_info(agent.ctrl), hardware=hw, v_ref=agent.v_ref, metrics=s)
