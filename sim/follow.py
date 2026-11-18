"""KB1: chạy một bộ điều khiển bám line trên đường thử và tính các chỉ tiêu.

`TrackRun` chạy từng chu kỳ điều khiển (dùng cho GUI); `run_track` chạy hết (dùng cho thực nghiệm).
"""
import numpy as np
from .robot import Robot, RobotParams
from .control import LineEstimator, wheel_cmd


class TrackRun:
    def __init__(self, world, ctrl, v_ref=0.3, seed=0, loaded=False, sens_noise=0.03, pose=None,
                 t_max=40.0, Ts=0.01, params=None):
        poly = world.polylines[0]
        if pose is None:
            d0 = poly[1] - poly[0]
            pose = (poly[0][0], poly[0][1], np.arctan2(d0[1], d0[0]))
        self.w, self.ctrl, self.v_ref, self.Ts, self.t_max = world, ctrl, v_ref, Ts, t_max
        self.p = params or RobotParams(loaded=loaded, sens_noise=sens_noise)
        self.rob = Robot(world, pose, self.p, seed=seed)
        self.est = LineEstimator(self.p)
        ctrl.reset()
        self.n_sub = int(round(Ts / self.rob.dt))
        self.end = poly[-1]
        self.errs, self.oms, self.ts, self.xs, self.ys, self.eys, self.wls, self.wrs = ([] for _ in range(8))
        self.lost_events, self.lost, self.lost_t, self.last_side = 0, False, 0.0, 0.0
        self.finished = self.failed = False
        self.done = False
        self.mode = 'follow'

    def tick(self):
        if self.done:
            return False
        rob = self.rob
        rf, rr = rob.read()
        ey, eth, yf, nf, nr = self.est.errors(rf, rr)
        near_end = np.hypot(rob.x - self.end[0], rob.y - self.end[1]) < 0.12
        if ey is None:
            if near_end:
                self.finished = True
                self.done = True
                return False
            if not self.lost:
                self.lost_events += 1
                self.lost = True
            self.lost_t += self.Ts
            if self.lost_t > 2.0:
                self.failed = True
                self.done = True
                return False
            om = 4.0 * (1 if self.last_side >= 0 else -1)
            v = 0.05
            self.mode = 'lost'
        else:
            self.lost, self.lost_t = False, 0.0
            self.last_side = yf if yf is not None else ey
            v = self.v_ref
            if hasattr(self.ctrl, 'load_gain'):
                self.ctrl.load_gain = self.p.tau / self.p.tau_load if self.p.loaded else 1.0
            om = self.ctrl(ey, eth, yf, v)
            self.mode = 'follow'
        wl, wr = wheel_cmd(v, om, self.p)
        rob.step_n(wl, wr, self.n_sub)
        self.errs.append(self.w.dist_to_poly(np.array([[rob.x, rob.y]]), 0)[0])
        self.oms.append(om)
        self.ts.append(rob.t)
        self.xs.append(rob.x)
        self.ys.append(rob.y)
        self.eys.append(ey if ey is not None else np.nan)
        self.wls.append(rob.wl)
        self.wrs.append(rob.wr)
        if rob.t >= self.t_max:
            self.done = True
            return False
        return True

    def metrics(self):
        errs, oms = np.array(self.errs), np.array(self.oms)
        k0 = int(0.3 / self.Ts)
        e = errs[k0:] if len(errs) > k0 else errs
        if len(e) == 0:
            e = np.array([np.nan])
        return dict(rms=float(np.sqrt(np.nanmean(e ** 2))) * 1000, emax=float(np.nanmax(e)) * 1000,
                    p95=float(np.nanpercentile(e, 95)) * 1000, time=self.rob.t, lost=self.lost_events,
                    finished=self.finished and not self.failed,
                    dom=float(np.sqrt(np.mean(np.diff(oms) ** 2))) if len(oms) > 2 else 0.0,
                    om_rms=float(np.sqrt(np.mean(oms ** 2))) if len(oms) else 0.0,
                    sat=float(np.mean(np.maximum(np.abs(self.wls), np.abs(self.wrs)) > 0.98 * self.p.w_max)) * 100 if self.wls else 0.0)


def run_track(world, ctrl, log=False, **kw):
    tr = TrackRun(world, ctrl, **kw)
    while tr.tick():
        pass
    res = tr.metrics()
    if log:
        res['log'] = dict(t=np.array(tr.ts), x=np.array(tr.xs), y=np.array(tr.ys), e=np.array(tr.errs),
                          om=np.array(tr.oms), ey=np.array(tr.eys))
    return res
