"""GUI mô phỏng robot vận chuyển hàng bám line trong nhà máy / kho thông minh.

Chạy:  python gui.py      (Python 3.9+ có tkinter, numpy, scipy, matplotlib)

Quy trình sử dụng:
  1. Vẽ mạng line (nhấp chuột để thêm điểm, nhấp đúp hoặc chuột phải để kết thúc nét). Có thể lưu/mở sân.
  2. Đặt robot lên line.
  3. Nhấn "Học map": robot tự bám line, khám phá các nhánh, tự lưu nút và cạnh.
  4. Chọn công cụ "Gán trạm" rồi nhấp vào nút đã học để đặt tên trạm.
  5. Thêm đơn (lấy tại / trả tại), chọn cách lập lịch, nhấn "Giao hàng". Có thể thêm đơn khi robot đang chạy.
  Trong lúc chạy có thể: đổi PID <-> LQR, chỉnh tham số, tốc độ; bật các tình huống (người đi ngang, vật cản,
  line mòn, sàn trơn, ánh sáng nhiễu, pin yếu, tải nặng).
"""
import time
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Polygon, Circle

from sim.world import build_from_strokes, save_strokes, load_strokes, Obstacle
from sim.robot import RobotParams
from sim.control import PID_DEFAULT, LQR_DEFAULT, PID, LQR, bryson
from sim.agent import Agent
from sim.scenarios import NOISE_LEVELS, map_score
from sim.planning import SCHEDULERS, STOP_METHODS

TOOLS = ['Vẽ line', 'Xoá nét gần nhất', 'Đặt robot', 'Gán trạm (nhấp vào nút đã học)',
         'Đặt vật cản đứng yên', 'Đặt người/xe đi ngang', 'Tạo đoạn line mòn/đứt', 'Đặt vùng sàn trơn', 'Xoá tình huống gần nhất']
SCENARIOS = {
    'Không có tình huống': [],
    'Người đi ngang qua tuyến': ['moving'],
    'Vật cản đứng yên chắn một đoạn line': ['static'],
    'Line bị mòn / đứt đoạn': ['gap'],
    'Sàn trơn (vết dầu)': ['slip'],
    'Ánh sáng nhiễu mạnh': ['light'],
    'Pin yếu (tốc độ động cơ tối đa giảm 40%)': ['battery'],
    'Chở tải nặng': ['heavy'],
    'Tổng hợp: người đi ngang + line mòn + sàn trơn': ['moving', 'gap', 'slip'],
}
SNAP = 0.05


class App:
    def __init__(self, root):
        self.root = root
        root.title('Mô phỏng robot vận chuyển hàng bám line – học map, gán trạm, giao hàng')
        root.geometry('1600x950')
        self.strokes, self.cur_stroke = [], []
        self.obstacles, self.gaps, self.slips = [], [], []
        self.robot_pose = None
        self.world = None
        self.agent = None
        self.running = False
        self.pending = []          # đơn đang chờ (chưa giao cho tới khi nhấn Giao hàng)
        self.session_runs = []     # các lượt đã chạy trong phiên (để xuất dữ liệu báo cáo)
        self.task_name = None
        self._build_left()
        self._build_right()
        self.rebuild_world()

    # ================================================================ giao diện
    def _section(self, parent, title):
        f = ttk.LabelFrame(parent, text=title, padding=4)
        f.pack(fill=tk.X, pady=3)
        return f

    def _row(self, parent, label, widget, w=20):
        fr = ttk.Frame(parent)
        fr.pack(fill=tk.X, pady=1)
        ttk.Label(fr, text=label, width=w).pack(side=tk.LEFT)
        widget.pack(in_=fr, side=tk.LEFT, fill=tk.X, expand=True)
        widget.lift(fr)
        return widget

    def _build_left(self):
        outer = ttk.Frame(self.root)
        outer.pack(side=tk.LEFT, fill=tk.Y)
        cv = tk.Canvas(outer, width=430, highlightthickness=0)
        sb = ttk.Scrollbar(outer, orient=tk.VERTICAL, command=cv.yview)
        L = ttk.Frame(cv, padding=4)
        L.bind('<Configure>', lambda e: cv.configure(scrollregion=cv.bbox('all')))
        cv.create_window((0, 0), window=L, anchor='nw')
        cv.configure(yscrollcommand=sb.set)
        cv.pack(side=tk.LEFT, fill=tk.Y)
        sb.pack(side=tk.LEFT, fill=tk.Y)

        # 1. sân
        f = self._section(L, '1. Sân line và công cụ chuột')
        self.v_tool = tk.StringVar(value=TOOLS[0])
        self._row(f, 'Công cụ', ttk.Combobox(f, textvariable=self.v_tool, values=TOOLS, state='readonly'), 10)
        self.v_snap = tk.BooleanVar(value=True)
        self._row(f, 'Bắt lưới 5 cm', ttk.Checkbutton(f, variable=self.v_snap), 10)
        b = ttk.Frame(f); b.pack(fill=tk.X)
        ttk.Button(b, text='Mở sân…', command=self.open_map).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(b, text='Lưu sân…', command=self.save_map).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(b, text='Xoá tất cả', command=self.clear_all).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(f, text='Đảo hướng robot', command=self.flip_robot).pack(fill=tk.X)
        ttk.Label(f, text='Vẽ: nhấp để thêm điểm, nhấp đúp / chuột phải để kết thúc nét.\n'
                          'Giao cắt và điểm chạm được tự tách thành giao lộ.', foreground='#555').pack(anchor='w')

        # 2. robot
        f = self._section(L, '2. Robot, cảm biến, phần cứng')
        self.v_v = tk.DoubleVar(value=0.30)
        self._row(f, 'Tốc độ đặt [m/s]', tk.Scale(f, variable=self.v_v, from_=0.1, to=0.6, resolution=0.05,
                                                  orient=tk.HORIZONTAL, command=lambda e: self.live_update()))
        self.v_gyro = tk.BooleanVar(value=True)
        self._row(f, 'Có IMU (gyro)', ttk.Checkbutton(f, variable=self.v_gyro))
        self.v_enc = tk.StringVar(value='Trung bình')
        self._row(f, 'Sai số chế tạo bánh', ttk.Combobox(f, textvariable=self.v_enc, values=list(NOISE_LEVELS), state='readonly'))
        self.v_noise = tk.DoubleVar(value=0.03)
        self._row(f, 'Nhiễu cảm biến line', tk.Scale(f, variable=self.v_noise, from_=0.0, to=0.2, resolution=0.01, orient=tk.HORIZONTAL))
        self.v_seed = tk.IntVar(value=1)
        self._row(f, 'Seed', ttk.Spinbox(f, from_=0, to=999, textvariable=self.v_seed, width=6))

        # 3. thuật toán
        f = self._section(L, '3. Thuật toán bám line (đổi được khi đang chạy)')
        self.v_ctrl = tk.StringVar(value='PID')
        cb = self._row(f, 'Thuật toán', ttk.Combobox(f, textvariable=self.v_ctrl, values=['PID', 'LQR'], state='readonly'))
        cb.bind('<<ComboboxSelected>>', lambda e: self.show_ctrl_params())
        box = ttk.Frame(f)
        box.pack(fill=tk.X)
        # PID: u = Kp·e + Ki·∫e + Kd·de/dt, e = sai lệch ngang tại tâm trục bánh
        self.pid_frame = ttk.Frame(box)
        self.v_kp, self.v_ki, self.v_kd = tk.DoubleVar(value=PID_DEFAULT['Kp']), tk.DoubleVar(value=PID_DEFAULT['Ki']), \
            tk.DoubleVar(value=PID_DEFAULT['Kd'])
        for lab, var in [('Kp', self.v_kp), ('Ki', self.v_ki), ('Kd', self.v_kd)]:
            self._row(self.pid_frame, lab, ttk.Entry(self.pid_frame, textvariable=var, width=10))
        # LQR/LQI: trọng số theo quy tắc Bryson, Q = diag(1/e_y,max², 1/e_θ,max², 1/ω_max², 1/z_max²), R = 1/u_max²
        self.lqr_frame = ttk.Frame(box)
        self.v_dmax, self.v_pmax, self.v_umax = tk.DoubleVar(value=LQR_DEFAULT['d_max_mm']), \
            tk.DoubleVar(value=LQR_DEFAULT['psi_max_deg']), tk.DoubleVar(value=LQR_DEFAULT['u_max'])
        self.v_zmax = tk.DoubleVar(value=LQR_DEFAULT['z_max_mm_s'])
        for lab, var in [('e_y cho phép [mm]', self.v_dmax), ('e_θ cho phép [độ]', self.v_pmax), ('ω lệnh tối đa [rad/s]', self.v_umax),
                         ('∫e_y cho phép [mm·s] (0: tắt)', self.v_zmax)]:
            self._row(self.lqr_frame, lab, ttk.Entry(self.lqr_frame, textvariable=var, width=10))
        self.pid_frame.pack(fill=tk.X)
        ttk.Button(f, text='Áp dụng tham số ngay', command=self.apply_ctrl).pack(fill=tk.X)
        self.lbl_gain = ttk.Label(f, text='', foreground='#0b5394')
        self.lbl_gain.pack(anchor='w')

        # 4. tình huống
        f = self._section(L, '4. Kịch bản tình huống')
        self.v_scn = tk.StringVar(value=list(SCENARIOS)[0])
        self._row(f, 'Kịch bản', ttk.Combobox(f, textvariable=self.v_scn, values=list(SCENARIOS), state='readonly', width=36), 8)
        ttk.Button(f, text='Áp dụng kịch bản (đặt ngẫu nhiên trên sân hiện tại)', command=self.apply_scenario).pack(fill=tk.X)
        self.v_light, self.v_batt, self.v_heavy = tk.BooleanVar(), tk.BooleanVar(), tk.BooleanVar()
        for lab, var in [('Ánh sáng nhiễu mạnh', self.v_light), ('Pin yếu', self.v_batt), ('Tải nặng (τ động cơ ↑)', self.v_heavy)]:
            self._row(f, lab, ttk.Checkbutton(f, variable=var, command=self.live_update))

        # 5. học map
        f = self._section(L, '5. Học map')
        self.v_assoc = tk.StringVar(value='Nhiều giả thuyết + kiểm chứng')
        self._row(f, 'Nhận dạng nút', ttk.Combobox(f, textvariable=self.v_assoc, values=['Nhiều giả thuyết + kiểm chứng', 'Đơn giản (gần nhất)'], state='readonly'))
        self.v_pg = tk.BooleanVar(value=True)
        self._row(f, 'Tối ưu pose graph', ttk.Checkbutton(f, variable=self.v_pg))
        ttk.Button(f, text='▶ Học map (robot tự chạy line)', command=self.learn_map).pack(fill=tk.X)

        # 6. trạm & đơn
        f = self._section(L, '6. Trạm và đơn hàng')
        self.lst_st = tk.Listbox(f, height=4)
        self.lst_st.pack(fill=tk.X)
        b = ttk.Frame(f); b.pack(fill=tk.X)
        self.v_pick, self.v_drop = tk.StringVar(), tk.StringVar()
        self.cb_pick = ttk.Combobox(b, textvariable=self.v_pick, width=8, state='readonly')
        self.cb_drop = ttk.Combobox(b, textvariable=self.v_drop, width=8, state='readonly')
        ttk.Label(b, text='Lấy').pack(side=tk.LEFT); self.cb_pick.pack(side=tk.LEFT)
        ttk.Label(b, text='Trả').pack(side=tk.LEFT); self.cb_drop.pack(side=tk.LEFT)
        ttk.Button(b, text='Thêm đơn', command=self.add_order).pack(side=tk.LEFT)
        self.lst_or = tk.Listbox(f, height=4)
        self.lst_or.pack(fill=tk.X)
        b2 = ttk.Frame(f); b2.pack(fill=tk.X)
        ttk.Button(b2, text='Xoá đơn đang chờ', command=self.clear_pending).pack(side=tk.LEFT, expand=True, fill=tk.X)
        self.v_sched = tk.StringVar(value=list(STOP_METHODS)[2])
        self._row(f, 'Lập lịch', ttk.Combobox(f, textvariable=self.v_sched, values=list(STOP_METHODS), state='readonly'))
        self.v_cap = tk.IntVar(value=1)
        self._row(f, 'Sức chứa (kiện/chuyến)', ttk.Spinbox(f, from_=1, to=6, textvariable=self.v_cap, width=5))
        ttk.Label(f, text='Đơn thêm vào chỉ nằm trong hàng chờ; robot chỉ giao\nkhi nhấn "Giao hàng" (giao toàn bộ hàng chờ một lượt).',
                  foreground='#555').pack(anchor='w')
        ttk.Button(f, text='▶ Giao hàng', command=self.deliver).pack(fill=tk.X)

        # 7. bộ thí nghiệm cho báo cáo
        f = self._section(L, '7. Bộ thí nghiệm cho báo cáo (E0–E4)')
        ttk.Label(f, text='E0 chỉnh định PID, LQR trên lưới tham số → E1 so sánh PID* với LQR*\n'
                          '→ E2 học map → E3 tình huống → E4 lập lịch; tự xuất bảng, hình (≈ 1 phút).',
                  foreground='#555').pack(anchor='w')
        b = ttk.Frame(f); b.pack(fill=tk.X)
        self.btn_camp = ttk.Button(b, text='▶ Chạy bộ thí nghiệm', command=self.run_campaign)
        self.btn_camp.pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(b, text='Xem kết quả', command=self.show_campaign).pack(side=tk.LEFT, expand=True, fill=tk.X)
        self.v_camp_fresh = tk.BooleanVar(value=False)
        self._row(f, 'Chạy lại từ đầu (bỏ đệm)', ttk.Checkbutton(f, variable=self.v_camp_fresh))
        self.pb_camp = ttk.Progressbar(f, maximum=100)
        self.pb_camp.pack(fill=tk.X, pady=2)
        self.lbl_camp = ttk.Label(f, text='Chưa chạy', foreground='#555')
        self.lbl_camp.pack(anchor='w')
        ttk.Button(f, text='Áp dụng PID*, LQR* vào bộ điều khiển', command=self.apply_tuned).pack(fill=tk.X)

        # điều khiển mô phỏng
        f = self._section(L, 'Mô phỏng')
        b = ttk.Frame(f); b.pack(fill=tk.X)
        ttk.Button(b, text='⏸ Tạm dừng', command=self.pause).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(b, text='▶ Tiếp tục', command=self.resume).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(b, text='Lưu ảnh', command=self.save_fig).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(f, text='📑 Xuất dữ liệu cho báo cáo', command=self.export_report).pack(fill=tk.X)
        self.lbl_runs = ttk.Label(f, text='Đã ghi nhận 0 lượt chạy (chưa xuất)', foreground='#555')
        self.lbl_runs.pack(anchor='w')
        self.v_speed = tk.IntVar(value=8)
        self._row(f, 'Tốc độ mô phỏng', tk.Scale(f, variable=self.v_speed, from_=1, to=60, orient=tk.HORIZONTAL))
        self.status = ttk.Label(L, text='Vẽ mạng line để bắt đầu', foreground='#0b5394', font=('Segoe UI', 10, 'bold'))
        self.status.pack(anchor='w', pady=4)

    def _build_right(self):
        R = ttk.Frame(self.root)
        R.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.fig = Figure(figsize=(11, 7.4), dpi=100)
        gs = self.fig.add_gridspec(4, 3)
        self.ax = self.fig.add_subplot(gs[:3, :])
        self.ax2 = self.fig.add_subplot(gs[3, :2])
        self.ax3 = self.fig.add_subplot(gs[3, 2])
        self.canvas = FigureCanvasTkAgg(self.fig, master=R)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.canvas.mpl_connect('motion_notify_event', self.on_move)
        bot = ttk.Frame(R)
        bot.pack(fill=tk.X)
        self.metrics = tk.Text(bot, height=8, width=90, font=('Consolas', 9), wrap='none')
        self.metrics.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.logbox = tk.Text(bot, height=8, width=70, font=('Consolas', 9), wrap='none')
        self.logbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # ================================================================ sân
    def snap(self, x, y):
        if self.v_snap.get():
            return round(x / SNAP) * SNAP, round(y / SNAP) * SNAP
        return x, y

    def rebuild_world(self):
        if self.strokes:
            self.world = build_from_strokes(self.strokes)
        else:
            self.world = None
        self.attach_env()
        self.redraw()

    def attach_env(self):
        if self.world is None:
            return
        self.world.obstacles = self.obstacles
        self.world.gaps = self.gaps
        self.world.slips = self.slips
        self.world.light_noise = 0.12 if self.v_light.get() else 0.0

    def nearest_line_point(self, x, y):
        if self.world is None:
            return None
        best = None
        for p in self.world.polylines:
            for a, b in zip(p[:-1], p[1:]):
                ab = b - a
                t = np.clip(np.dot([x, y] - a, ab) / max(np.dot(ab, ab), 1e-12), 0, 1)
                q = a + t * ab
                d = np.hypot(*(q - [x, y]))
                if best is None or d < best[0]:
                    best = (d, q, np.arctan2(ab[1], ab[0]))
        return best

    def on_move(self, ev):
        if ev.inaxes != self.ax or not self.cur_stroke or self.v_tool.get() != 'Vẽ line':
            return
        x, y = self.snap(ev.xdata, ev.ydata)
        pts = np.array(self.cur_stroke + [(x, y)])
        self.a_preview.set_data(pts[:, 0], pts[:, 1])
        self.canvas.draw_idle()

    def on_click(self, ev):
        if ev.inaxes != self.ax or ev.xdata is None:
            return
        tool = self.v_tool.get()
        x, y = ev.xdata, ev.ydata
        if tool == 'Vẽ line':
            if self.running and self.agent is not None:
                messagebox.showinfo('Đang chạy', 'Tạm dừng mô phỏng trước khi vẽ thêm line.')
                return
            if ev.button == 3 or ev.dblclick:
                if len(self.cur_stroke) >= 2:
                    self.strokes.append(self.cur_stroke)
                self.cur_stroke = []
                self.agent = None
                self.rebuild_world()
                return
            self.cur_stroke.append(self.snap(x, y))
            self.redraw(keep_preview=True)
        elif tool == 'Xoá nét gần nhất':
            if not self.strokes:
                return
            d = [min(np.hypot(px - x, py - y) for px, py in s) for s in self.strokes]
            self.strokes.pop(int(np.argmin(d)))
            self.agent = None
            self.rebuild_world()
        elif tool == 'Đặt robot':
            r = self.nearest_line_point(x, y)
            if r is None:
                return
            self.robot_pose = (r[1][0], r[1][1], r[2])
            self.agent = None
            self.redraw()
        elif tool.startswith('Gán trạm'):
            self.assign_station(x, y)
        elif tool == 'Đặt vật cản đứng yên':
            r = self.nearest_line_point(x, y)
            if r:
                self.obstacles.append(Obstacle(r[1], 0.07, 'static'))
                self.redraw()
        elif tool == 'Đặt người/xe đi ngang':
            r = self.nearest_line_point(x, y)
            if r:
                self.obstacles.append(Obstacle(r[1], 0.10, 'moving', dirn=r[2] + np.pi / 2, amp=0.45, speed=0.4, pause=1.5))
                self.redraw()
        elif tool == 'Tạo đoạn line mòn/đứt':
            r = self.nearest_line_point(x, y)
            if r:
                self.gaps.append((tuple(r[1]), 0.02))
                self.redraw()
        elif tool == 'Đặt vùng sàn trơn':
            self.slips.append(((x, y), 0.25, 0.15))
            self.redraw()
        elif tool == 'Xoá tình huống gần nhất':
            items = [('o', i, o.c) for i, o in enumerate(self.obstacles)] + [('g', i, np.array(g[0])) for i, g in enumerate(self.gaps)] + \
                    [('s', i, np.array(s[0])) for i, s in enumerate(self.slips)]
            if items:
                k, i, _ = min(items, key=lambda it: np.hypot(*(it[2] - [x, y])))
                {'o': self.obstacles, 'g': self.gaps, 's': self.slips}[k].pop(i)
                self.redraw()
        self.attach_env()

    def flip_robot(self):
        if self.robot_pose:
            x, y, th = self.robot_pose
            self.robot_pose = (x, y, th + np.pi)
            self.agent = None
            self.redraw()

    def clear_all(self):
        self.pause()
        self.strokes, self.cur_stroke = [], []
        self.obstacles, self.gaps, self.slips = [], [], []
        self.robot_pose, self.agent = None, None
        self.lst_st.delete(0, tk.END); self.lst_or.delete(0, tk.END)
        self.rebuild_world()

    def open_map(self):
        fn = filedialog.askopenfilename(filetypes=[('Sân line', '*.json')], initialdir='maps')
        if fn:
            self.clear_all()
            self.strokes = [[tuple(p) for p in s] for s in load_strokes(fn)]
            self.rebuild_world()

    def save_map(self):
        fn = filedialog.asksaveasfilename(defaultextension='.json', filetypes=[('Sân line', '*.json')], initialdir='maps')
        if fn:
            save_strokes(fn, [[list(p) for p in s] for s in self.strokes])

    def apply_scenario(self):
        if self.world is None or not self.world.polylines:
            messagebox.showinfo('Kịch bản', 'Hãy vẽ sân line trước.')
            return
        rng = np.random.default_rng()
        polys = [p for p in self.world.polylines if len(p) >= 2]
        lens = [np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)) for p in polys]

        def point_on_line():
            p = polys[rng.choice(len(polys), p=np.array(lens) / sum(lens))]
            seg = rng.integers(len(p) - 1)
            t = rng.uniform(0.3, 0.7)
            a, b = p[seg], p[seg + 1]
            return a + t * (b - a), np.arctan2(*(b - a)[::-1])
        for item in SCENARIOS[self.v_scn.get()]:
            if item == 'moving':
                q, ang = point_on_line()
                self.obstacles.append(Obstacle(q, 0.10, 'moving', dirn=ang + np.pi / 2, amp=0.45, speed=0.4, pause=1.5))
            elif item == 'static':
                q, _ = point_on_line()
                self.obstacles.append(Obstacle(q, 0.07, 'static'))
            elif item == 'gap':
                for _ in range(2):
                    q, _ = point_on_line()
                    self.gaps.append((tuple(q), 0.02))
            elif item == 'slip':
                q, _ = point_on_line()
                self.slips.append((tuple(q), 0.3, 0.15))
            elif item == 'light':
                self.v_light.set(True)
            elif item == 'battery':
                self.v_batt.set(True)
            elif item == 'heavy':
                self.v_heavy.set(True)
        self.attach_env()
        self.live_update()
        self.redraw()
        self.log(f'Áp dụng kịch bản: {self.v_scn.get()}')

    # ================================================================ thuật toán
    def make_ctrl(self):
        if self.v_ctrl.get() == 'PID':
            c = PID(Kp=self.v_kp.get(), Ki=self.v_ki.get(), Kd=self.v_kd.get())
            self.lbl_gain.config(text=f'PID: Kp={c.Kp:g}, Ki={c.Ki:g}, Kd={c.Kd:g}')
            return c
        z = self.v_zmax.get()
        c = LQR(model='dyn', tau=RobotParams().tau, d_max=self.v_dmax.get() / 1000, psi_max=np.radians(self.v_pmax.get()),
                u_max=self.v_umax.get(), integral=z > 0, **({'z_max': z / 1000} if z > 0 else {}))
        K = c.gain(self.v_v.get())
        self.lbl_gain.config(text=('LQI' if z > 0 else 'LQR') + ' K = [' + ', '.join(f'{k:.1f}' for k in K) + ']  (e_y, e_θ, ω'
                             + (', ∫e_y)' if z > 0 else ')'))
        return c

    def show_ctrl_params(self):
        """Chỉ hiện tham số của thuật toán đang chọn."""
        if self.v_ctrl.get() == 'PID':
            self.lqr_frame.pack_forget()
            self.pid_frame.pack(fill=tk.X)
        else:
            self.pid_frame.pack_forget()
            self.lqr_frame.pack(fill=tk.X)
        self.apply_ctrl()

    def apply_ctrl(self):
        c = self.make_ctrl()
        if self.agent is not None:
            self.agent.ctrl = c
            self.log(f'Đổi bộ điều khiển: {self.lbl_gain.cget("text")}')

    def live_update(self):
        self.attach_env()
        if self.agent is not None:
            self.agent.v_ref = self.v_v.get()
            p = self.agent.p
            p.w_max = 31.4 * (0.6 if self.v_batt.get() else 1.0)
            p.tau_load = p.tau * (1.8 if self.v_heavy.get() else 1.5)
            p.loaded = p.loaded or self.v_heavy.get()

    # ================================================================ học map, trạm, đơn
    def new_agent(self):
        if self.world is None or self.robot_pose is None:
            messagebox.showinfo('Thiếu dữ liệu', 'Hãy vẽ sân line và đặt robot lên line trước.')
            return None
        self.attach_env()
        kw = dict(NOISE_LEVELS[self.v_enc.get()])
        p = RobotParams(sens_noise=self.v_noise.get(), gyro=self.v_gyro.get(), **kw)
        ag = Agent(self.world, self.robot_pose, ctrl=self.make_ctrl(),
                   assoc='mht' if self.v_assoc.get().startswith('Nhiều') else 'naive',
                   params=p, seed=self.v_seed.get(), v_ref=self.v_v.get(), posegraph=self.v_pg.get())
        self.agent = ag
        self._nlog = 0
        self.logbox.delete('1.0', tk.END)
        self.live_update()
        self.redraw()
        return ag

    def learn_map(self):
        ag = self.new_agent()
        if ag is None:
            return
        self.lst_st.delete(0, tk.END)
        ag.task = ag.explore()
        self.mode = 'learn'
        self.task_name, self.trace_start = 'hoc_map', 0
        self.resume()

    def assign_station(self, x, y):
        ag = self.agent
        if ag is None or not ag.map.nodes:
            messagebox.showinfo('Gán trạm', 'Cần học map trước.')
            return
        n = min(ag.map.nodes.values(), key=lambda n: np.hypot(n.pos[0] - x, n.pos[1] - y))
        if np.hypot(n.pos[0] - x, n.pos[1] - y) > 0.2:
            return
        name = simpledialog.askstring('Gán trạm', f'Tên trạm cho nút {n.id} (bậc {n.degree}):',
                                      initialvalue=n.station or chr(ord('A') + len(ag.stations)))
        if not name:
            return
        for k in [k for k, v in ag.stations.items() if v == n.id]:
            ag.stations.pop(k)
        ag.stations[name] = n.id
        n.station = name
        self.refresh_stations()
        self.redraw_map()

    def refresh_stations(self):
        ag = self.agent
        self.lst_st.delete(0, tk.END)
        names = sorted(ag.stations) if ag else []
        for k in names:
            self.lst_st.insert(tk.END, f'Trạm {k}  →  nút {ag.stations[k]}')
        self.cb_pick['values'] = names
        self.cb_drop['values'] = names

    def add_order(self):
        a, b = self.v_pick.get(), self.v_drop.get()
        if not a or not b or a == b or self.agent is None:
            return
        self.pending.append((a, b))
        self.refresh_orders()
        busy = self.is_delivering()
        self.log(f'Thêm đơn {a} → {b} vào hàng chờ' + (' (robot đang giao lượt trước)' if busy else ' – nhấn Giao hàng để bắt đầu'))

    def clear_pending(self):
        self.pending = []
        self.refresh_orders()

    def is_delivering(self):
        ag = self.agent
        return ag is not None and ag.task is not None and getattr(ag, 'batch', None) is not None and ag.mode not in ('xong lượt giao',)

    def refresh_orders(self):
        self.lst_or.delete(0, tk.END)
        ag = self.agent
        if ag is not None and getattr(ag, 'batch', None):
            for k, (a, b) in enumerate(ag.batch):
                self.lst_or.insert(tk.END, f'[lượt hiện tại] đơn {k + 1}: {a} → {b}   ({ag.batch_status[k]})')
        for a, b in self.pending:
            self.lst_or.insert(tk.END, f'[chờ] {a} → {b}')

    def deliver(self):
        ag = self.agent
        if ag is None or not ag.map.nodes or len(ag.stations) < 2:
            messagebox.showinfo('Giao hàng', 'Cần học map và gán ít nhất 2 trạm.')
            return
        if self.running and ag.task is not None:
            messagebox.showinfo('Giao hàng', 'Robot đang chạy (học map hoặc đang giao lượt trước).\nĐơn mới đã nằm trong hàng chờ; nhấn Giao hàng khi robot xong.')
            return
        if not self.pending:
            messagebox.showinfo('Giao hàng', 'Hàng chờ chưa có đơn.')
            return
        batch, self.pending = self.pending, []
        ag.task = ag.deliver(batch, STOP_METHODS[self.v_sched.get()], int(self.v_cap.get()))
        self.task_name, self.trace_start = 'giao_hang', len(ag.trace)
        self.batch_orders = list(batch)
        self.refresh_orders()
        self.resume()

    # ================================================================ vòng lặp
    def resume(self):
        if self.agent is None or self.agent.task is None:
            return
        if not self.running:
            self.running = True
            self.loop()

    def pause(self):
        self.running = False

    def loop(self):
        if not self.running:
            return
        alive = True
        for _ in range(self.v_speed.get()):
            alive = self.agent.tick()
            if not alive:
                break
        self.refresh()
        if alive:
            self.root.after(15, self.loop)
        else:
            self.running = False
            self.refresh(final=True)
            self.record_run()
            if self.agent.map.nodes and not self.agent.stations:
                self.status.config(text='Học map xong – chọn công cụ "Gán trạm" và nhấp vào nút')
            else:
                self.status.config(text='Hoàn thành')

    # ================================================================ vẽ
    def redraw(self, keep_preview=False):
        ax = self.ax
        ax.clear()
        if self.world is not None:
            for p in self.world.polylines:
                ax.plot(p[:, 0], p[:, 1], color='#222', lw=4, solid_capstyle='butt', zorder=1)
        for c, r in self.gaps:
            ax.add_patch(Circle(c, r, color='white', zorder=2))
            ax.add_patch(Circle(c, r * 1.8, fill=False, ec='#e69138', lw=1, ls='--', zorder=2))
        for c, r, _ in self.slips:
            ax.add_patch(Circle(c, r, color='#9fc5e8', alpha=0.35, zorder=0))
        self.a_obs = []
        for ob in self.obstacles:
            col = '#cc0000' if ob.kind == 'static' else '#e69138'
            pt = Circle(ob.pos(self.agent.rob.t if self.agent else 0), ob.r, color=col, alpha=0.8, zorder=6)
            ax.add_patch(pt)
            self.a_obs.append((ob, pt))
            if ob.kind == 'moving':
                a, b = ob.c - ob.amp * ob.u, ob.c + ob.amp * ob.u
                ax.plot([a[0], b[0]], [a[1], b[1]], ':', color=col, lw=1)
        self.a_preview, = ax.plot([], [], '--', color='#1f77b4', lw=2)
        if keep_preview and self.cur_stroke:
            pts = np.array(self.cur_stroke)
            self.a_preview.set_data(pts[:, 0], pts[:, 1])
            ax.plot(pts[:, 0], pts[:, 1], 'o', color='#1f77b4', ms=4)
        self.a_traj, = ax.plot([], [], color='#1f77b4', lw=1.2, label='Quỹ đạo thật', zorder=3)
        self.a_odom, = ax.plot([], [], color='#ff7f0e', lw=1.0, ls='--', label='Tư thế ước lượng', zorder=3)
        self.a_body = Polygon(np.zeros((4, 2)), closed=True, fc='#ffd966', ec='k', zorder=7)
        ax.add_patch(self.a_body)
        self.a_sens = ax.scatter([], [], s=9, zorder=8)
        self.a_nodes = ax.scatter([], [], s=110, facecolors='none', edgecolors='#9c27b0', lw=2, zorder=8, label='Nút đã học')
        self.map_artists = []
        ax.set_aspect('equal')
        ax.grid(alpha=0.25)
        if self.world is None:
            ax.set_xlim(-0.5, 3.5); ax.set_ylim(-1.0, 2.0)
            ax.set_title('Vẽ mạng line: nhấp chuột để thêm điểm, nhấp đúp để kết thúc nét')
        else:
            ax.set_title(f'Sân: {len(self.world.polylines)} đoạn, {len(self.world.nodes)} nút thật '
                         f'(giao lộ + đầu mút) – robot chưa biết')
        if self.robot_pose is not None and self.agent is None:
            from sim.robot import Robot
            rob = Robot(self.world, self.robot_pose, RobotParams()) if self.world else None
            if rob:
                self.draw_robot(rob)
        ax.legend(loc='upper right', fontsize=8)
        if self.agent is not None:
            self.draw_robot(self.agent.rob)
            self.redraw_map()
        self.canvas.draw_idle()

    def draw_robot(self, rob):
        c, s = np.cos(rob.th), np.sin(rob.th)
        Rm = np.array([[c, -s], [s, c]])
        body = np.array([[-0.04, -0.065], [0.11, -0.05], [0.11, 0.05], [-0.04, 0.065]])
        self.a_body.set_xy(body @ Rm.T + [rob.x, rob.y])
        pts = np.vstack([rob.sensor_points(rob.p.x_front), rob.sensor_points(rob.p.x_rear)])
        val = rob.w.reflect(pts)
        self.a_sens.set_offsets(pts)
        self.a_sens.set_color(np.where(val[:, None] > 0.35, [[0.85, 0.1, 0.1]], [[0.6, 0.6, 0.6]]))

    def redraw_map(self):
        for a in self.map_artists:
            a.remove()
        self.map_artists = []
        ag = self.agent
        if ag is None or not ag.map.nodes:
            self.a_nodes.set_offsets(np.zeros((0, 2)))
            return
        nodes = ag.map.nodes
        self.a_nodes.set_offsets(np.array([n.pos for n in nodes.values()]))
        for n in nodes.values():
            lbl = f'{n.id}' + (f' [{n.station}]' if n.station else '')
            self.map_artists.append(self.ax.text(n.pos[0] + 0.04, n.pos[1] + 0.04, lbl, color='#6a1b9a', fontsize=9,
                                                 fontweight='bold' if n.station else 'normal', zorder=9))
            for p, l in enumerate(n.links):
                if l is not None and l[0] in nodes and l[0] > n.id:
                    q = nodes[l[0]].pos
                    self.map_artists += self.ax.plot([n.pos[0], q[0]], [n.pos[1], q[1]], color='#9c27b0', lw=1, ls=':', zorder=4)
            for p, a in enumerate(n.gports()):
                col = '#cc0000' if p in n.blocked else '#9c27b0'
                self.map_artists += self.ax.plot([n.pos[0], n.pos[0] + 0.08 * np.cos(a)], [n.pos[1], n.pos[1] + 0.08 * np.sin(a)],
                                                 color=col, lw=1.6, zorder=4)

    def refresh(self, final=False):
        ag = self.agent
        rob = ag.rob
        self.draw_robot(rob)
        for ob, pt in getattr(self, 'a_obs', []):
            pt.center = ob.pos(rob.t)
        tr = np.array(ag.traj[-8000:]) if ag.traj else np.zeros((0, 5))
        if len(tr):
            self.a_traj.set_data(tr[:, 1], tr[:, 2])
            self.a_odom.set_data(tr[:, 3], tr[:, 4])
        if final or time.time() - getattr(self, '_last_full', 0) > 0.8:
            self._last_full = time.time()
            self.redraw_map()
            self.refresh_stations()
            self.refresh_orders()
            self.ax2.clear()
            ey = np.array(ag.ey_log[-3000:]) * 1000
            self.ax2.plot(np.arange(len(ey)) * 0.01, ey, lw=0.8, color='#1f77b4')
            self.ax2.set_ylabel('e_y [mm]'); self.ax2.set_xlabel('3000 chu kỳ gần nhất [s]'); self.ax2.grid(alpha=0.3)
            self.ax3.clear()
            st = ag.stats
            keys = ['gộp nhầm', 'nút trùng', 'kiểm chứng', 'dừng vì vật cản', 'cạnh bị chặn', 'qua line đứt']
            vals = [st['wrong_merge'], st['duplicate'], st['active_checks'], st['obstacle_stops'], st['blocked_edges'], st['gap_crossings']]
            self.ax3.barh(keys, vals, color='#9c27b0')
            self.ax3.tick_params(labelsize=7)
            sc = map_score(self.world, ag.map) if ag.map.nodes else {}
            node_err = [np.hypot(*(n.pos - np.array(self.world.nodes[n.truth]))) for n in ag.map.nodes.values() if n.truth in self.world.nodes]
            tra = np.array(ag.traj) if ag.traj else np.zeros((1, 5))
            oe = np.hypot(tra[:, 1] - tra[:, 3], tra[:, 2] - tra[:, 4])
            txt = (f't = {rob.t:7.1f} s   chế độ: {ag.mode:16s} v_đặt = {ag.v_ref:.2f} m/s   quãng đường {rob.dist_true:6.2f} m   '
                   f'bộ điều khiển: {type(ag.ctrl).__name__}\n'
                   f'Bản đồ: {len(ag.map.nodes)} nút học / {len(self.world.nodes)} nút thật; đúng hoàn toàn = {sc.get("correct")}; '
                   f'lệch vị trí nút TB {np.mean(node_err)*1000 if node_err else 0:.0f} mm, max {np.max(node_err)*1000 if node_err else 0:.0f} mm\n'
                   f'Sai số tư thế ước lượng hiện tại {oe[-1]*1000:.0f} mm (TB {oe.mean()*1000:.0f}); b̂ = {rob.b_est*1000:.1f} mm '
                   f'(thật {rob.p.b*(1+rob.p.b_err)*1000:.1f}); ε̂ = {rob.eps_est:+.4f} (thật {rob.p.r_err_r-rob.p.r_err_l:+.4f}); '
                   f'gyro bias {rob.gbias_est:+.4f} (thật {rob.p.gyro_bias if rob.p.gyro else 0:+.4f})\n'
                   f'Số lần quét {st["scans"]}, tổng góc quay tại chỗ {np.degrees(st["rot_total"]):.0f}°, va chạm {rob.collisions}, '
                   f'chờ vật cản {st["wait_time"]:.1f} s, mất line {st["lost"]}\n')
            if getattr(ag, 'plan_text', None):
                txt += f'Kế hoạch lượt ({ag.plan_cost:.2f} m): {ag.plan_text}\n'
            for d in ag.delivery[-6:]:
                txt += f"  {d['action']:4s} tại {d['station']}: t={d['t']:7.1f} s, sai số dừng {d['err_mm']:5.1f} mm, đúng trạm={d['correct']}\n"
            self.metrics.delete('1.0', tk.END)
            self.metrics.insert(tk.END, txt)
            n = int(self.logbox.index('end-1c').split('.')[0]) - 1
            for t, msg in ag.events[max(0, getattr(self, '_nlog', 0)):]:
                self.logbox.insert(tk.END, f'[{t:7.2f}s] {msg}\n')
            self._nlog = len(ag.events)
            self.logbox.see(tk.END)
            self.status.config(text=f'{ag.mode} – t = {rob.t:.0f} s')
        self.canvas.draw_idle()

    def log(self, msg):
        self.logbox.insert(tk.END, msg + '\n')
        self.logbox.see(tk.END)

    def settings(self):
        """Thiết lập GUI của lượt chạy – ghi kèm dữ liệu để biết lượt đó chạy với điều kiện nào."""
        return dict(speed=self.v_v.get(), gyro=self.v_gyro.get(), wheel_error=self.v_enc.get(), sensor_noise=self.v_noise.get(),
                    seed=self.v_seed.get(), scenario=self.v_scn.get(), light=self.v_light.get(), low_battery=self.v_batt.get(),
                    heavy=self.v_heavy.get(), association=self.v_assoc.get(), pose_graph=self.v_pg.get(),
                    schedule=self.v_sched.get(), capacity=int(self.v_cap.get()),
                    orders=getattr(self, 'batch_orders', []) if getattr(self, 'task_name', '') == 'giao_hang' else [],
                    stations=dict(self.agent.stations) if self.agent else {},
                    n_strokes=len(self.strokes), true_nodes=len(self.world.nodes) if self.world else 0,
                    obstacles=len(self.obstacles), gaps=len(self.gaps), slips=len(self.slips))

    def record_run(self):
        """Ghi nhận lượt vừa xong vào bộ nhớ của phiên (chưa ghi ra đĩa)."""
        from sim.report_export import snapshot
        ag = self.agent
        if ag is None or not ag.trace or getattr(self, 'task_name', None) is None:
            return
        start = getattr(self, 'trace_start', 0)
        if len(ag.trace) - start < 10:
            return
        self.session_runs.append(snapshot(ag, self.world, self.task_name, self.settings(), start))
        self.task_name = None
        self.lbl_runs.config(text=f'Đã ghi nhận {len(self.session_runs)} lượt chạy (chưa xuất)')

    def export_report(self):
        """Xuất bảng, hình, danh mục cho báo cáo từ các lượt đã chạy trong phiên."""
        from sim.report_export import export_report
        runs = list(self.session_runs)
        ag = self.agent
        if ag is not None and getattr(self, 'task_name', None) and len(ag.trace) - getattr(self, 'trace_start', 0) > 10:
            from sim.report_export import snapshot     # lượt đang chạy dở: xuất kèm, ghi rõ là chưa xong
            runs.append(snapshot(ag, self.world, self.task_name, dict(self.settings(), dang_chay_do=True), self.trace_start))
        if not runs:
            messagebox.showinfo('Xuất dữ liệu cho báo cáo', 'Chưa có lượt chạy nào. Hãy Học map và/hoặc Giao hàng trước.')
            return
        root = filedialog.askdirectory(title='Chọn thư mục lưu dữ liệu báo cáo',
                                       initialdir=os.path.dirname(os.path.abspath(__file__)))
        if not root:
            return
        self.status.config(text='Đang xuất dữ liệu cho báo cáo…')
        self.root.update()
        out = export_report(runs, self.world, root, strokes=self.strokes)
        self.log(f'Đã xuất dữ liệu báo cáo ({len(runs)} lượt): {out}')
        self.status.config(text='Đã xuất dữ liệu cho báo cáo')
        messagebox.showinfo('Xuất dữ liệu cho báo cáo', f'Đã lưu {len(runs)} lượt chạy vào\n{out}\n\n'
                            'Xem DANH_MUC_HINH_BANG.md để biết bảng/hình nào dùng cho mục nào của báo cáo.')

    # ================================================================ bộ thí nghiệm (chạy tiến trình riêng, GUI không bị treo)
    def campaign_dir(self):
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'campaign')

    def run_campaign(self):
        import subprocess, threading, queue, sys as _sys, shutil
        if getattr(self, '_camp_proc', None) is not None and self._camp_proc.poll() is None:
            return
        here = os.path.dirname(os.path.abspath(__file__))
        if self.v_camp_fresh.get():
            shutil.rmtree(os.path.join(self.campaign_dir(), 'raw'), ignore_errors=True)
        self._camp_q = queue.Queue()
        self._camp_t0 = time.time()
        self._camp_stage = {}
        env = dict(os.environ, PYTHONIOENCODING='utf-8')
        self._camp_proc = subprocess.Popen([_sys.executable, '-u', os.path.join(here, 'experiments', 'campaign.py')], cwd=here,
                                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8',
                                           errors='replace', env=env)

        def reader(p, q):
            for line in p.stdout:
                q.put(line.rstrip())
            q.put(None)
        threading.Thread(target=reader, args=(self._camp_proc, self._camp_q), daemon=True).start()
        self.btn_camp.config(state='disabled')
        self.pb_camp['value'] = 0
        self.log('Bắt đầu bộ thí nghiệm E0–E4 …')
        self.root.after(200, self._poll_campaign)

    def _poll_campaign(self):
        import re
        done = False
        while not self._camp_q.empty():
            line = self._camp_q.get()
            if line is None:
                done = True
                break
            m = re.match(r'\[(E0|E1\+E2\+E3|E1|E2|E3)\] (\d+)/(\d+) lượt', line)
            if m:
                frac = int(m.group(2)) / int(m.group(3))
                self.pb_camp['value'] = 10 + 15 * frac if m.group(1) == 'E0' else 25 + 65 * frac
            if 'XONG' in line or '*' in line or 'LỖI' in line or 'Traceback' in line or line.startswith('---'):
                self.log('[thí nghiệm] ' + line)
            if line.startswith('BÁO CÁO XONG'):
                self.pb_camp['value'] = 100
            self.lbl_camp.config(text=f'{time.time() - self._camp_t0:.0f} s – {line[:70]}')
        if done or (self._camp_proc.poll() is not None and self._camp_q.empty()):
            ok = self._camp_proc.wait() == 0
            self.btn_camp.config(state='normal')
            self.lbl_camp.config(text=('Xong' if ok else 'Lỗi') + f' sau {time.time() - self._camp_t0:.0f} s – nhấn "Xem kết quả"')
            if ok:
                self.show_campaign()
            return
        self.root.after(200, self._poll_campaign)

    def apply_tuned(self):
        import json
        p = os.path.join(self.campaign_dir(), 'tuned.json')
        if not os.path.exists(p):
            messagebox.showinfo('PID*, LQR*', 'Chưa có kết quả chỉnh định – hãy chạy bộ thí nghiệm trước.')
            return
        best = json.load(open(p, encoding='utf-8'))
        if 'PID' in best:
            q = best['PID']['params']
            self.v_kp.set(q['Kp']); self.v_ki.set(q['Ki']); self.v_kd.set(q['Kd'])
        if 'LQR' in best:
            q = best['LQR']['params']
            self.v_dmax.set(q['e_y_max_mm']); self.v_pmax.set(q['e_theta_max_deg']); self.v_umax.set(q['omega_max'])
            self.v_zmax.set(q.get('z_max_mm_s', 0))
        self.apply_ctrl()
        self.log('Đã áp dụng PID*, LQR* từ bộ thí nghiệm: ' + '; '.join(f"{k}* = {v['params']}" for k, v in best.items()))

    def show_campaign(self):
        """Cửa sổ xem kết quả: mỗi giai đoạn một thẻ; chọn tệp bên trái để xem hình hoặc bảng."""
        import glob
        d = os.path.join(self.campaign_dir(), 'bao_cao')
        files = sorted(glob.glob(os.path.join(d, '*.png')) + glob.glob(os.path.join(d, '*.csv')))
        if not files:
            messagebox.showinfo('Kết quả bộ thí nghiệm', 'Chưa có kết quả – hãy chạy bộ thí nghiệm.')
            return
        top = tk.Toplevel(self.root)
        top.title('Kết quả bộ thí nghiệm cho báo cáo – ' + d)
        top.geometry('1400x860')
        nb = ttk.Notebook(top)
        nb.pack(fill=tk.BOTH, expand=True)
        groups = [('Tóm tắt', lambda n: n in ('E0_tham_so_chon.csv', 'E0_vai_tro_thanh_phan.csv', 'E0_vai_tro_thanh_phan.png',
                                              'doi_chieu_muc_tieu.csv', 'E1_tong_hop.csv')),
                  ('E0 chỉnh định', lambda n: n.startswith('E0')), ('E1 PID* vs LQR*', lambda n: n.startswith('E1')),
                  ('E2 học map', lambda n: n.startswith('E2')), ('E3 tình huống', lambda n: n.startswith('E3')),
                  ('E4 lập lịch', lambda n: n.startswith('E4'))]
        for title, pick in groups:
            fs = [f for f in files if pick(os.path.basename(f))]
            if fs:
                self._campaign_tab(nb, title, fs)

    def _campaign_tab(self, nb, title, fs):
        import pandas as pd
        fr = ttk.Frame(nb)
        nb.add(fr, text=title)
        lb = tk.Listbox(fr, width=34, exportselection=False)
        lb.pack(side=tk.LEFT, fill=tk.Y)
        for f in fs:
            lb.insert(tk.END, ('🖼 ' if f.endswith('.png') else '▦ ') + os.path.basename(f))
        view = ttk.Frame(fr)
        view.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        def show(_=None):
            sel = lb.curselection()
            if not sel:
                return
            for w in view.winfo_children():
                w.destroy()
            f = fs[sel[0]]
            if f.endswith('.png'):
                fig = Figure(figsize=(10, 7))
                ax = fig.add_axes([0, 0, 1, 1])
                import matplotlib.image as mpimg
                ax.imshow(mpimg.imread(f))
                ax.axis('off')
                cv = FigureCanvasTkAgg(fig, master=view)
                cv.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                cv.draw()
            else:
                df = pd.read_csv(f, encoding='utf-8-sig')
                cols = list(df.columns)
                tv = ttk.Treeview(view, columns=cols, show='headings')
                for c in cols:
                    tv.heading(c, text=c)
                    n = max([len(str(c))] + [len(f'{x:.4g}' if isinstance(x, float) else str(x)) for x in df[c].head(50)])
                    tv.column(c, width=max(60, min(520, 7 * n + 12)), anchor='w', stretch=False)
                for row in df.head(400).itertuples(index=False):
                    tv.insert('', tk.END, values=[f'{x:.4g}' if isinstance(x, float) else x for x in row])
                sy = ttk.Scrollbar(view, orient='vertical', command=tv.yview)
                sx = ttk.Scrollbar(view, orient='horizontal', command=tv.xview)
                tv.configure(yscrollcommand=sy.set, xscrollcommand=sx.set)
                sy.pack(side=tk.RIGHT, fill=tk.Y)
                sx.pack(side=tk.BOTTOM, fill=tk.X)
                tv.pack(fill=tk.BOTH, expand=True)
        lb.bind('<<ListboxSelect>>', show)
        lb.selection_set(0)
        show()

    def save_fig(self):
        fn = filedialog.asksaveasfilename(defaultextension='.png', filetypes=[('PNG', '*.png')])
        if fn:
            self.fig.savefig(fn, dpi=150)


if __name__ == '__main__':
    root = tk.Tk()
    try:
        ttk.Style().theme_use('clam')
    except tk.TclError:
        pass
    App(root)
    root.mainloop()
