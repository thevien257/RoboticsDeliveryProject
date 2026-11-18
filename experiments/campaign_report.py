"""Xuất bảng, hình, danh mục và manifest từ dữ liệu của bộ thí nghiệm (gọi từ campaign.py)."""
import os, sys, json, time, platform, subprocess
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

COL = {'PID*': '#1f77b4', 'LQR*': '#d62728', 'PID': '#1f77b4', 'LQR': '#d62728'}


def _ci(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan
    return float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))


def _agg(d, by, cols):
    rows = []
    for k, g in d.groupby(by):
        r = dict(zip(by, k if isinstance(k, tuple) else (k,)))
        r['so_luot'] = len(g)
        for c in cols:
            r[f'{c}_tb'] = g[c].mean()
            r[f'{c}_dlc'] = g[c].std(ddof=1)
            r[f'{c}_ci95'] = _ci(g[c])
        rows.append(r)
    return pd.DataFrame(rows)


def _welch(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2 or (a.std() == 0 and b.std() == 0):
        return np.nan
    return float(stats.ttest_ind(a, b, equal_var=False).pvalue)


class Out:
    def __init__(self, root):
        self.dir = os.path.join(root, 'bao_cao')
        os.makedirs(self.dir, exist_ok=True)
        self.items = []

    def tab(self, df, fn, cap, sec, src):
        df.to_csv(os.path.join(self.dir, fn), index=False, encoding='utf-8-sig', float_format='%.4g')
        self.items.append(('Bảng', fn, cap, sec, src))

    def fig(self, f, fn, cap, sec, src):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            f.tight_layout()
        f.savefig(os.path.join(self.dir, fn), dpi=150, bbox_inches='tight')
        plt.close(f)
        self.items.append(('Hình', fn, cap, sec, src))


def _load(raw, name, keep_errors=False):
    p = os.path.join(raw, f'{name}.jsonl')
    if os.path.exists(p):
        d = pd.read_json(p, lines=True)
    elif os.path.exists(os.path.join(raw, f'{name}.csv')):
        d = pd.read_csv(os.path.join(raw, f'{name}.csv'))
    else:
        return None
    if keep_errors:
        return d
    return d[d['error'].isna()] if 'error' in d.columns else d


# ---------------------------------------------------------------- E0
def report_e0(o, root, cfg):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import campaign
    campaign.OUT, campaign.RAW = root, os.path.join(root, 'raw')
    best = {}
    for algo in ['PID', 'LQR']:
        s = campaign.e0_summary(algo)
        if s is None or s.empty:
            continue
        best[algo] = s.iloc[0]
        lab = 'Kp × Ki × Kd' if algo == 'PID' else 'e_y,max × e_θ,max × ω_max (Bryson)'
        o.tab(s, f'E0_{algo}_luoi_tham_so.csv', f'Kết quả chỉnh định {algo} trên lưới {lab} ({len(s)} bộ tham số), xếp hạng theo luật chọn',
              '3.2, 4.1', 'E0: đáp ứng bước 2 cm, đường hỗn hợp v = 0,3 và 0,4 m/s, đường tròn R = 0,3 m')
    if not best:
        return best
    rows = []
    for algo, r in best.items():
        keys = list(cfg['E0'][algo])
        rows.append(dict(bo_dieu_khien=algo + '*', tham_so=', '.join(f'{k} = {r[k]:g}' for k in keys), kha_thi=r.kha_thi,
                         rms_mm=r.rms, p95_mm=r.p95, vot_lo_pct=r.overshoot, xac_lap_s=r.settle, dao_dong_con_lai_mm=r.residual,
                         sai_lech_tinh_duong_tron_mm=r.ess_circle, do_giat=r.jerk, mat_line=r.lost))
    o.tab(pd.DataFrame(rows), 'E0_tham_so_chon.csv', 'Tham số được chọn sau chỉnh định (PID*, LQR*) và chỉ tiêu trên tập chỉnh định',
          '3.2, 4.1', 'hàng đầu của hai bảng lưới; luật chọn ở campaign_config.json')
    # vai trò từng thành phần: bộ tốt nhất của mỗi cấu trúc (P, PI, PD, PID; LQR, LQI)
    srows = []
    for algo in ['PID', 'LQR']:
        s = campaign.e0_summary(algo)
        if s is None:
            continue
        for cs, g in s.groupby('cau_truc'):
            g = g.sort_values(['so_vi_pham', 'rms'], ascending=[True, True])
            r = g.iloc[0]
            keys = list(cfg['E0'][algo])
            srows.append(dict(cau_truc=cs, so_bo_tham_so=len(g), so_bo_kha_thi=int(g.kha_thi.sum()),
                              tham_so_tot_nhat=', '.join(f'{k} = {r[k]:g}' for k in keys), rms_mm=r.rms, vot_lo_pct=r.overshoot,
                              dao_dong_con_lai_mm=r.residual, sai_lech_tinh_dong_co_lech_mm=r.ess_bias,
                              sai_lech_tinh_duong_tron_mm=r.ess_circle, do_giat=r.jerk, mat_line=r.lost, kha_thi=r.kha_thi))
    S = pd.DataFrame(srows)
    order = {'P': 0, 'PI': 1, 'PD': 2, 'PID': 3, 'LQR': 4, 'LQI': 5}
    S = S.sort_values('cau_truc', key=lambda c: c.map(order)).reset_index(drop=True)
    o.tab(S, 'E0_vai_tro_thanh_phan.csv', 'Bộ tham số tốt nhất của từng cấu trúc (P, PI, PD, PID; LQR, LQI) – vai trò của từng thành phần',
          '4.2', 'E0, chọn trong từng cấu trúc theo cùng luật')
    f, axs = plt.subplots(1, 4, figsize=(16, 3.6))
    for ax, col, lab in [(axs[0], 'vot_lo_pct', 'vọt lố bước [%]'), (axs[1], 'dao_dong_con_lai_mm', 'dao động còn lại [mm]'),
                         (axs[2], 'sai_lech_tinh_dong_co_lech_mm', '|sai lệch tĩnh| động cơ lệch [mm]'), (axs[3], 'rms_mm', 'RMS lệch line [mm]')]:
        ax.bar(S.cau_truc, S[col], color=['#90caf9', '#64b5f6', '#42a5f5', '#1565c0', '#ef9a9a', '#c62828'][:len(S)])
        ax.set_title(lab, fontsize=9); ax.grid(axis='y', alpha=0.3)
        for x, v in enumerate(S[col]):
            ax.text(x, v, f'{v:.1f}', ha='center', va='bottom', fontsize=7)
    o.fig(f, 'E0_vai_tro_thanh_phan.png', 'So sánh cấu trúc P, PI, PD, PID và LQR, LQI (bộ tốt nhất của mỗi cấu trúc)', '4.2',
          'E0_vai_tro_thanh_phan.csv')
    s = campaign.e0_summary('PID')
    if s is not None and not s.empty:
        kis = sorted(s.Ki.unique())
        for metric, title, fn, cap in [('rms', 'RMS lệch line [mm]', 'E0_PID_ban_do_RMS.png', 'RMS lệch line theo Kp và Kd ở từng mức Ki (× : không khả thi)'),
                                       ('overshoot', 'Vọt lố bước 2 cm [%]', 'E0_PID_ban_do_vot_lo.png', 'Vọt lố theo Kp và Kd ở từng mức Ki')]:
            nc = 3
            nr = int(np.ceil((len(kis) + 1) / nc))
            f, axs = plt.subplots(nr, nc, figsize=(4.3 * nc, 3.4 * nr), squeeze=False)
            vmax = np.nanpercentile(s[metric], 90)
            for ax, ki in zip(axs.ravel(), kis):
                g = s[s.Ki == ki].pivot(index='Kd', columns='Kp', values=metric).sort_index(ascending=False)
                fe = s[s.Ki == ki].pivot(index='Kd', columns='Kp', values='kha_thi').sort_index(ascending=False)
                im = ax.imshow(g.values, cmap='viridis_r', vmin=0, vmax=vmax, aspect='auto')
                ax.set_xticks(range(g.shape[1])); ax.set_xticklabels([f'{x:g}' for x in g.columns], fontsize=7)
                ax.set_yticks(range(g.shape[0])); ax.set_yticklabels([f'{x:g}' for x in g.index], fontsize=7)
                ax.set_xlabel('Kp'); ax.set_ylabel('Kd'); ax.set_title(f'Ki = {ki:g}', fontsize=9)
                for i in range(g.shape[0]):
                    for j in range(g.shape[1]):
                        v = g.values[i, j]
                        ax.text(j, i, ('×' if not fe.values[i, j] else '') + (f'{v:.0f}' if np.isfinite(v) else ''), ha='center',
                                va='center', fontsize=6, color='w' if np.isfinite(v) and v > vmax * 0.5 else 'k')
            cax = axs.ravel()[len(kis)]
            for ax in axs.ravel()[len(kis) + 1:]:
                ax.axis('off')
            cax.axis('off')
            cb = f.colorbar(im, ax=cax, fraction=0.5, aspect=12, location='left', label=title)
            cax.text(0.55, 0.5, '× : bộ tham số không khả thi\n(mất line, vọt lố > 30 %,\ndao động còn lại > 2 mm\nhoặc sai lệch tĩnh > 1 mm)',
                     transform=cax.transAxes, va='center', fontsize=8)
            o.fig(f, fn, cap, '4.1, 4.2', 'E0_PID_luoi_tham_so.csv')
        f, axs = plt.subplots(1, 3, figsize=(14, 3.8))
        k0 = s[s.Ki == 0]
        for kd in sorted(k0.Kd.unique()):
            g = k0[k0.Kd == kd].sort_values('Kp')
            axs[0].plot(g.Kp, g.overshoot, 'o-', ms=3, label=f'Kd = {kd:g}')
        axs[0].set_xscale('log'); axs[0].set_xlabel('Kp'); axs[0].set_ylabel('vọt lố [%]'); axs[0].set_title('Ki = 0: tăng Kp', fontsize=9)
        for kp in sorted(k0.Kp.unique()):
            g = k0[k0.Kp == kp].sort_values('Kd')
            axs[1].plot(g.Kd, g.overshoot, 'o-', ms=3, label=f'Kp = {kp:g}')
        axs[1].set_xlabel('Kd'); axs[1].set_ylabel('vọt lố [%]'); axs[1].set_title('Ki = 0: tăng Kd', fontsize=9)
        b = best['PID']
        g = s[(s.Kp == b.Kp) & (s.Kd == b.Kd)].sort_values('Ki')
        axs[2].plot(g.Ki, g.ess_bias, 'o-', label='|sai lệch tĩnh| động cơ lệch 20 % [mm]')
        axs[2].plot(g.Ki, g.ess_circle, 'd-', label='|sai lệch tĩnh thật| đường tròn [mm]')
        axs[2].set_ylabel('sai lệch tĩnh [mm]')
        ax2b = axs[2].twinx(); ax2b.plot(g.Ki, g.overshoot, 's--', color='gray', label='vọt lố bước [%] (trục phải)'); ax2b.set_ylabel('vọt lố [%]')
        h1, l1 = axs[2].get_legend_handles_labels(); h2, l2 = ax2b.get_legend_handles_labels()
        axs[2].legend(h1 + h2, l1 + l2, fontsize=7, loc='center right')
        axs[2].set_xlabel('Ki'); axs[2].set_title(f'Kp = {b.Kp:g}, Kd = {b.Kd:g}: tăng Ki', fontsize=9)
        for ax in axs[:2]:
            ax.grid(alpha=0.3); ax.legend(fontsize=7)
        axs[2].grid(alpha=0.3)
        o.fig(f, 'E0_PID_anh_huong_Kp_Ki_Kd.png', 'Ảnh hưởng riêng của Kp, Kd (vọt lố) và Ki (sai lệch tĩnh, vọt lố)', '4.2',
              'E0_PID_luoi_tham_so.csv')
    s = campaign.e0_summary('LQR')
    if s is not None and not s.empty:
        z_star = best['LQR'].z_max_mm_s if 'LQR' in best else s.z_max_mm_s.max()
        s = s[s.z_max_mm_s == z_star]
        us = sorted(s.omega_max.unique())
        f, axs = plt.subplots(1, len(us), figsize=(3.6 * len(us), 3.6), squeeze=False)
        vmax = np.nanpercentile(s.rms, 90)
        for ax, u in zip(axs[0], us):
            g = s[s.omega_max == u].pivot(index='e_theta_max_deg', columns='e_y_max_mm', values='rms').sort_index(ascending=False)
            fe = s[s.omega_max == u].pivot(index='e_theta_max_deg', columns='e_y_max_mm', values='kha_thi').sort_index(ascending=False)
            im = ax.imshow(g.values, cmap='viridis_r', vmin=0, vmax=vmax, aspect='auto')
            ax.set_xticks(range(g.shape[1])); ax.set_xticklabels([f'{x:g}' for x in g.columns], fontsize=7)
            ax.set_yticks(range(g.shape[0])); ax.set_yticklabels([f'{x:g}' for x in g.index], fontsize=7)
            ax.set_xlabel('e_y,max [mm]'); ax.set_ylabel('e_θ,max [độ]'); ax.set_title(f'ω_max = {u:g} rad/s', fontsize=9)
            for i in range(g.shape[0]):
                for j in range(g.shape[1]):
                    v = g.values[i, j]
                    ax.text(j, i, ('×' if not fe.values[i, j] else '') + (f'{v:.1f}' if np.isfinite(v) else ''), ha='center', va='center',
                            fontsize=6, color='w' if np.isfinite(v) and v > vmax * 0.5 else 'k')
        f.colorbar(im, ax=axs[0].tolist(), shrink=0.8, label='RMS lệch line [mm]')
        o.fig(f, 'E0_LQR_ban_do_RMS.png', f'RMS lệch line của LQI theo giới hạn Bryson ở z_max = {z_star:g} mm·s (× : không khả thi)',
              '4.1, 4.2', 'E0_LQR_luoi_tham_so.csv')
    return best


# ---------------------------------------------------------------- E1
def report_e1(o, raw):
    d = _load(raw, 'E1')
    if d is None or d.empty:
        return None
    cols = ['rms', 'p95', 'emax', 'jerk', 'lost']
    rows = []
    for (grp, tr, v, nz, bs, ld), g in d.groupby(['group', 'track', 'v', 'noise', 'bias', 'loaded']):
        a, b = g[g.ctrl == 'PID*'], g[g.ctrl == 'LQR*']
        r = dict(nhom=grp, duong=tr, v=v, nhieu=nz, dong_co_lech=bs, cho_hang=ld, n=min(len(a), len(b)))
        for c in cols:
            r[f'PID_{c}'] = a[c].mean(); r[f'PID_{c}_dlc'] = a[c].std(ddof=1)
            r[f'LQR_{c}'] = b[c].mean(); r[f'LQR_{c}_dlc'] = b[c].std(ddof=1)
        r['p_rms'] = _welch(a.rms, b.rms); r['p_jerk'] = _welch(a.jerk, b.jerk)
        r['ket_luan_rms'] = ('khác biệt có ý nghĩa (p < 0,05)' if r['p_rms'] < 0.05 else 'không khác biệt có ý nghĩa') if np.isfinite(r['p_rms']) else '–'
        rows.append(r)
    T = pd.DataFrame(rows)
    o.tab(T, 'E1_PID_vs_LQR.csv', 'So sánh PID* và LQR* theo từng điều kiện (trung bình, độ lệch chuẩn, kiểm định Welch)', '4.2',
          'E1: đường không dùng khi chỉnh định, mỗi điều kiện 5 seed')
    o.tab(_agg(d, ['ctrl', 'group'], cols), 'E1_tong_hop.csv', 'Tổng hợp E1 theo nhóm điều kiện (trung bình, độ lệch chuẩn, CI 95 %)', '4.2', 'E1')
    s = d[d.group == 'đường × tốc độ']
    trs = sorted(s.track.unique())
    f, axs = plt.subplots(2, len(trs), figsize=(4.4 * len(trs), 6.4), squeeze=False)
    for j, tr in enumerate(trs):
        for c in ['PID*', 'LQR*']:
            g = s[(s.track == tr) & (s.ctrl == c)].groupby('v')
            m, e = g.rms.mean(), g.rms.apply(_ci)
            axs[0, j].errorbar(m.index, m.values, yerr=e.values, marker='o', capsize=3, color=COL[c], label=c)
            m, e = g.jerk.mean(), g.jerk.apply(_ci)
            axs[1, j].errorbar(m.index, m.values, yerr=e.values, marker='s', capsize=3, color=COL[c], label=c)
            lo = g.lost.sum()
            for v, n in lo.items():
                if n:
                    axs[0, j].annotate(f'mất {int(n)}', (v, m.get(v, 0)), fontsize=6, color=COL[c])
        axs[0, j].set_title(tr, fontsize=9); axs[0, j].set_ylabel('RMS lệch line [mm]'); axs[1, j].set_ylabel('độ giật lệnh ω [rad/s²]')
        axs[1, j].set_xlabel('v [m/s]')
        for ax in axs[:, j]:
            ax.grid(alpha=0.3); ax.legend(fontsize=7)
    o.fig(f, 'E1_toc_do.png', 'RMS lệch line và độ giật lệnh theo tốc độ trên các đường kiểm tra (thanh lỗi: CI 95 %)', '4.2', 'E1_PID_vs_LQR.csv')
    s = d[d.group == 'nhiễu cảm biến']
    f, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    for c in ['PID*', 'LQR*']:
        g = s[s.ctrl == c].groupby('noise')
        axs[0].errorbar(g.rms.mean().index, g.rms.mean().values, yerr=g.rms.apply(_ci).values, marker='o', capsize=3, color=COL[c], label=c)
        axs[1].errorbar(g.jerk.mean().index, g.jerk.mean().values, yerr=g.jerk.apply(_ci).values, marker='s', capsize=3, color=COL[c], label=c)
    axs[0].set_ylabel('RMS lệch line [mm]'); axs[1].set_ylabel('độ giật lệnh ω [rad/s²]')
    for ax in axs:
        ax.set_xlabel('nhiễu cảm biến (chuẩn hoá)'); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    o.fig(f, 'E1_nhieu_cam_bien.png', 'Độ bền với nhiễu cảm biến (thanh lỗi: CI 95 %)', '4.2', 'E1_PID_vs_LQR.csv')
    s = d[d.group == 'nhiễu loạn'].copy()
    s['dk'] = s.apply(lambda r: f"lệch {r.bias * 100:.0f} %" + (', chở' if r.loaded else ''), axis=1)
    f, ax = plt.subplots(figsize=(9, 3.8))
    dks = list(dict.fromkeys(s.sort_values(['loaded', 'bias']).dk))
    for k, c in enumerate(['PID*', 'LQR*']):
        g = s[s.ctrl == c].groupby('dk')
        m, e = g.rms.mean().reindex(dks), g.rms.apply(_ci).reindex(dks)
        ax.bar(np.arange(len(dks)) + (k - 0.5) * 0.38, m.values, 0.38, yerr=e.values, capsize=3, color=COL[c], label=c)
    ax.set_xticks(range(len(dks))); ax.set_xticklabels(dks, fontsize=8); ax.set_ylabel('RMS lệch line [mm]'); ax.grid(axis='y', alpha=0.3)
    ax.legend()
    o.fig(f, 'E1_nhieu_loan.png', 'Ảnh hưởng của động cơ lệch và chở hàng (thanh lỗi: CI 95 %)', '4.2', 'E1_PID_vs_LQR.csv')
    return d


# ---------------------------------------------------------------- E2, E3, E4
def report_e2(o, raw):
    d = _load(raw, 'E2')
    if d is None or d.empty:
        return None
    d['correct'] = d.correct.astype(float)
    T = _agg(d, ['map', 'level', 'cfg'], ['correct', 'node_err_mean', 'node_err_max', 'odo_err_mean', 'wrong_merge', 'duplicate', 't_explore'])
    o.tab(T, 'E2_hoc_map.csv', 'Học map theo sân, mức sai số bánh và cấu hình (tỉ lệ đúng, sai số nút, sai số odometry)', '4.1, 4.2',
          'E2: mỗi ô 2 seed; robot dùng PID*')
    lv = [l for l in ['Lý tưởng', 'Thấp', 'Trung bình', 'Cao'] if l in set(d.level)]
    maps = sorted(d['map'].unique())
    cfgs = list(dict.fromkeys(d.cfg))
    cols = ['#2e7d32', '#ef6c00', '#1565c0', '#8e24aa']
    f, axs = plt.subplots(1, len(maps), figsize=(6.5 * len(maps), 4.2), squeeze=False)
    w = 0.8 / len(cfgs)
    for j, m in enumerate(maps):
        ax = axs[0, j]
        for k, cfg in enumerate(cfgs):
            g = d[(d['map'] == m) & (d.cfg == cfg)].groupby('level').node_err_mean
            mu, lo, hi = g.mean().reindex(lv), g.min().reindex(lv), g.max().reindex(lv)
            x = np.arange(len(lv)) + (k - (len(cfgs) - 1) / 2) * w
            ax.bar(x, mu.values, w, color=cols[k % 4], label=cfg, yerr=[mu.values - lo.values, hi.values - mu.values], capsize=2)
        ok = d[d['map'] == m].correct.mean() * 100
        ax.set_xticks(range(len(lv))); ax.set_xticklabels(lv)
        ax.set_title(f'sân {m} – bản đồ đúng hoàn toàn {ok:.0f} % số lượt', fontsize=9)
        ax.set_ylabel('sai số vị trí nút TB [mm]'); ax.set_xlabel('mức sai số chế tạo bánh'); ax.grid(axis='y', alpha=0.3)
        ax.legend(fontsize=7)
    o.fig(f, 'E2_hoc_map.png', 'Sai số vị trí nút theo mức sai số bánh và cấu hình (cột: trung bình, vạch: nhỏ nhất–lớn nhất các seed)', '4.2', 'E2_hoc_map.csv')
    return d


def report_e3(o, raw, cfg=None):
    d = _load(raw, 'E3')
    if d is not None and cfg is not None:
        d = d[d.scen.isin(cfg['E3']['kich_ban'])]          # chỉ các kịch bản đang có trong cấu hình
    if d is None or d.empty:
        return None
    T = _agg(d, ['scen', 'ctrl'], ['success', 'stop_err', 'collisions', 'waits', 'wait_time', 'replans', 'lost', 'track_rms', 't', 'dist'])
    o.tab(T, 'E3_tinh_huong.csv', 'Giao 3 đơn trong các kịch bản tình huống với PID* và LQR*', '4.1, 4.2', 'E3: mỗi ô 2 seed, sân kho, sai số bánh Trung bình')
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    from sim.scenarios import SCEN
    sc = [s for s in SCEN if s in set(d.scen)]
    f, axs = plt.subplots(1, 3, figsize=(15, 4.2))
    for k, c in enumerate(['PID*', 'LQR*']):
        g = d[d.ctrl == c].groupby('scen')
        for ax, col, lab in [(axs[0], 'success', 'thành công [%]'), (axs[1], 'stop_err', 'sai số dừng TB [mm]'), (axs[2], 't', 'thời gian giao [s]')]:
            v = g[col].mean().reindex(sc) * (100 if col == 'success' else 1)
            ax.barh(np.arange(len(sc)) + (k - 0.5) * 0.38, v.values, 0.38, color=COL[c], label=c)
            ax.set_xlabel(lab)
    for ax in axs:
        ax.set_yticks(range(len(sc))); ax.set_yticklabels(sc if ax is axs[0] else [], fontsize=8); ax.invert_yaxis(); ax.grid(axis='x', alpha=0.3)
    axs[0].legend(fontsize=8)
    o.fig(f, 'E3_tinh_huong.png', 'Tỉ lệ thành công, sai số dừng và thời gian giao theo kịch bản', '4.2', 'E3_tinh_huong.csv')
    return d


def report_e4(o, raw):
    d = _load(raw, 'E4')
    if d is None or d.empty:
        return None
    dp = d[d.method == 'Quy hoạch động'].set_index(['n', 'rep', 'Q']).cost
    d['vuot_QHD_pct'] = [100 * (c / dp[(n, r, q)] - 1) if dp[(n, r, q)] > 0 else 0 for c, n, r, q in zip(d.cost, d.n, d.rep, d.Q)]
    T = _agg(d, ['n', 'Q', 'method'], ['cost', 'vuot_QHD_pct', 'ms'])
    o.tab(T, 'E4_lap_lich.csv', 'Quãng đường kế hoạch, phần trăm dài hơn QHĐ và thời gian tính theo số đơn, sức chứa, phương pháp', '4.2',
          'E4: đồ thị thật của sân kho, đơn ngẫu nhiên 30 lần mỗi số đơn')
    f, axs = plt.subplots(1, 3, figsize=(15, 4))
    for m, c in [('FIFO', '#9e9e9e'), ('Tham lam', '#ff9800'), ('Quy hoạch động', '#2e7d32')]:
        g = d[(d.method == m) & (d.Q == 1)].groupby('n')
        axs[0].errorbar(g.cost.mean().index, g.cost.mean().values, yerr=g.cost.apply(_ci).values, marker='o', capsize=3, color=c, label=m)
        g = d[d.method == m].groupby('Q')
        axs[1].bar(np.array(sorted(d.Q.unique())) + {'FIFO': -0.27, 'Tham lam': 0, 'Quy hoạch động': 0.27}[m], g.vuot_QHD_pct.mean().values,
                   0.27, color=c, label=m)
        g = d[d.method == m].groupby('n')
        axs[2].plot(g.ms.mean().index, g.ms.mean().values, 'o-', color=c, label=m)
    axs[0].set_xlabel('số đơn'); axs[0].set_ylabel('quãng đường kế hoạch [m] (sức chứa 1)')
    axs[1].set_xlabel('sức chứa [kiện]'); axs[1].set_ylabel('dài hơn QHĐ [%]')
    axs[2].set_xlabel('số đơn'); axs[2].set_ylabel('thời gian tính [ms]'); axs[2].set_yscale('log')
    for ax in axs:
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    o.fig(f, 'E4_lap_lich.png', 'So sánh phương pháp lập lịch theo số đơn và sức chứa', '4.2', 'E4_lap_lich.csv')
    return d


# ---------------------------------------------------------------- đối chiếu mục tiêu, danh mục, manifest
def objectives(o, best, e1, e2, e3, e4):
    rows = []
    def add(code, what, crit, meas, ok):
        v = ok if isinstance(ok, str) else ('–' if ok is None else ('Đạt' if ok else 'Chưa đạt'))
        rows.append(dict(ma=code, muc_tieu=what, tieu_chi=crit, ket_qua=meas, danh_gia=v))
    if e1 is not None:
        s = e1[(e1.group == 'đường × tốc độ') & (e1.v <= 0.4)]
        meas = '; '.join(f"{c}: RMS TB {g.rms.mean():.2f} mm, mất line {int(g.lost.sum())}/{len(g)} lượt" for c, g in s.groupby('ctrl'))
        okc = {c: bool(g.rms.mean() < 5 and g.lost.sum() == 0) for c, g in s.groupby('ctrl')}
        ok = 'Đạt' if all(okc.values()) else ('Đạt với ' + ', '.join(c for c, v in okc.items() if v) if any(okc.values()) else 'Chưa đạt')
        add('MT1', 'Bám line ổn định (PID, LQR)', 'RMS < 5 mm, không mất line ở v ≤ 0,4 m/s trên đường kiểm tra', meas, ok)
    if e2 is not None:
        g = e2[(e2.cfg == 'encoder + gyro (đề xuất)') & (e2.level == 'Trung bình')]
        if g.empty:
            g = e2[e2.cfg == 'encoder + gyro (đề xuất)']
        add('MT2', 'Tự học bản đồ không dùng mốc', 'đúng hoàn toàn ≥ 90 % lượt (cấu hình đề xuất, sai số Trung bình)',
            f"{g.correct.mean() * 100:.0f} % ({len(g)} lượt)", g.correct.mean() >= 0.9)
        add('MT3', 'Định vị nút đủ chính xác', 'sai số nút TB < 100 mm', f"{g.node_err_mean.mean():.0f} ± {g.node_err_mean.std():.0f} mm",
            g.node_err_mean.mean() < 100)
    if e3 is not None:
        g = e3[e3.scen == 'Bình thường']
        add('MT4', 'Giao đúng trạm, dừng chính xác', 'thành công 100 %, sai số dừng TB < 10 mm (kịch bản bình thường)',
            f"thành công {g.success.mean() * 100:.0f} %, sai số dừng {g.stop_err.mean():.1f} mm", g.success.mean() == 1 and g.stop_err.mean() < 10)
        ok_s = e3.groupby('scen').apply(lambda x: x.success.mean() == 1 and x.collisions.sum() == 0)
        add('MT5', 'Xử lý tình huống', 'hoàn thành mọi kịch bản tình huống, không va chạm',
            f"{int(ok_s.sum())}/{len(ok_s)} kịch bản; va chạm {int(e3.collisions.sum())}", bool(ok_s.all()) and e3.collisions.sum() == 0)
    if e4 is not None:
        g = e4.pivot_table(index=['n', 'rep', 'Q'], columns='method', values='cost')
        ok = bool((g['Quy hoạch động'] <= g[['FIFO', 'Tham lam']].min(axis=1) + 1e-9).all())
        ms = e4[(e4.method == 'Quy hoạch động') & (e4.Q == 1)].groupby('n').ms.mean()
        ms3 = e4[(e4.method == 'Quy hoạch động') & (e4.Q == e4.Q.max())].groupby('n').ms.mean()
        add('MT6', 'Lập lịch tối ưu', 'QHĐ không dài hơn FIFO/tham lam ở mọi bộ đơn; tính < 10 ms với ≤ 7 đơn (chở 1 kiện)',
            f"QHĐ ngắn nhất ở {int((g['Quy hoạch động'] <= g[['FIFO', 'Tham lam']].min(axis=1) + 1e-9).sum())}/{len(g)} bộ đơn; "
            f"{ms.max():.2f} ms với {int(ms.idxmax())} đơn (sức chứa {int(e4.Q.max())}: {ms3.max():.1f} ms)", ok and ms.max() < 10)
    add('MT7', 'GUI cho toàn bộ quy trình', 'vẽ sân → học map → gán trạm → giao hàng, đổi thuật toán khi chạy', 'kiểm tra trên GUI', True)
    o.tab(pd.DataFrame(rows), 'doi_chieu_muc_tieu.csv', 'Đối chiếu kết quả thí nghiệm với mục tiêu MT1–MT7', '4.2, 5.1', 'E1–E4')


def _ver(mod):
    try:
        return __import__(mod).__version__
    except Exception:
        return '?'


def _part(args):
    """Xuất phần báo cáo của một giai đoạn (chạy trong tiến trình riêng để các giai đoạn vẽ hình song song)."""
    name, root, cfg = args
    o = Out(root)
    raw = os.path.join(root, 'raw')
    if name == 'E0':
        data = report_e0(o, root, cfg)
    elif name == 'E3':
        data = report_e3(o, raw, cfg)
    else:
        data = {'E1': report_e1, 'E2': report_e2, 'E4': report_e4}[name](o, raw)
    return name, o.items, data


def build_report(root, cfg, cmd='', t_run=0.0, procs=4):
    raw = os.path.join(root, 'raw')
    o = Out(root)
    from multiprocessing import Pool
    with Pool(procs) as P:
        parts = dict((n, (it, d)) for n, it, d in P.map(_part, [(n, root, cfg) for n in ['E0', 'E1', 'E2', 'E3', 'E4']]))
    for n in ['E0', 'E1', 'E2', 'E3', 'E4']:
        o.items += parts[n][0]
    best = parts['E0'][1]
    e1, e2, e3, e4 = (parts[n][1] for n in ['E1', 'E2', 'E3', 'E4'])
    objectives(o, best, e1, e2, e3, e4)
    repo = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    try:
        commit = subprocess.check_output(['git', '-C', repo, 'rev-parse', 'HEAD'], text=True).strip()
        dirty = bool(subprocess.check_output(['git', '-C', repo, 'status', '--porcelain', '--', 'sim', 'experiments'], text=True).strip())
    except Exception:
        commit, dirty = '?', None
    counts = {}
    for n in ['E0', 'E1', 'E2', 'E3', 'E4']:
        d = _load(raw, n, keep_errors=True)
        if d is not None:
            counts[n] = dict(luot=len(d), loi=int(d['error'].notna().sum()) if 'error' in d.columns else 0)
    tuned = json.load(open(os.path.join(root, 'tuned.json'), encoding='utf-8')) if os.path.exists(os.path.join(root, 'tuned.json')) else {}
    man = dict(thoi_diem=time.strftime('%Y-%m-%d %H:%M:%S'), lenh=cmd, thoi_gian_chay_s=round(t_run, 1), git_commit=commit,
               ma_nguon_co_sua_chua_commit=dirty, python=platform.python_version(),
               thu_vien={m: _ver(m) for m in ['numpy', 'scipy', 'pandas', 'matplotlib']}, so_luot=counts, tham_so_chon=tuned, cau_hinh=cfg)
    with open(os.path.join(o.dir, 'manifest.json'), 'w', encoding='utf-8') as f:
        json.dump(man, f, ensure_ascii=False, indent=1)
    nb = nh = 0
    L = ['# Danh mục bảng và hình của bộ thí nghiệm', '',
         f'Tạo lúc {man["thoi_diem"]} bằng `{cmd}` (commit `{commit[:10]}`). Số lượt: '
         + ', '.join(f'{k} {v["luot"]} ({v["loi"]} lỗi)' for k, v in counts.items()) + '.', '',
         'Toàn bộ số liệu do nhóm tự tạo bằng mô phỏng; vị trí thật chỉ dùng để chấm điểm. Khi đưa vào báo cáo, đánh số lại theo chương.', '',
         '| Loại | Số tạm | Tệp | Chú thích | Mục báo cáo | Nguồn |', '|---|---|---|---|---|---|']
    for kind, fn, cap, sec, src in o.items:
        if kind == 'Bảng':
            nb += 1; no = f'Bảng {nb}'
        else:
            nh += 1; no = f'Hình {nh}'
        L.append(f'| {kind} | {no} | `{fn}` | {cap} | {sec} | {src} |')
    L += ['', '## Quy trình', '',
          '1. E0 chỉnh định PID và LQR trên lưới tham số (tập chỉnh định: đáp ứng bước, đường hỗn hợp, đường tròn R = 0,3 m).',
          '2. Chọn PID*, LQR* theo luật trong `campaign_config.json` (không mất line, hoàn thành, vọt lố ≤ 30 %, dao động còn lại ≤ 2 mm, RMS nhỏ nhất).',
          '3. E1 so sánh PID* với LQR* trên dữ liệu kiểm tra độc lập (đường sóng, tròn R = 0,15 / 0,5 m) với 5 seed; kiểm định Welch.',
          '4. E2 học map, E3 tình huống, E4 lập lịch dùng PID*/LQR*.',
          '', 'Chạy lại: `' + cmd + '` (lượt đã có trong `results/campaign/raw` được dùng lại).']
    with open(os.path.join(o.dir, 'DANH_MUC_HINH_BANG.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n')
    return o.dir
