"""Sơ đồ khối hệ thống và các lưu đồ (vẽ bằng matplotlib) cho báo cáo."""
import os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, Ellipse, FancyArrowPatch
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "figures")
os.makedirs(OUT, exist_ok=True)
plt.rcParams['font.family'] = ['Liberation Sans', 'DejaVu Sans']
NAVY = '#1f3552'
GREEN = '#e6f4ea'
BLUE = '#e8f1fb'
CREAM = '#f7f5e8'
ORANGE = '#fdf0e1'
GREY = '#f2f2f2'


def box(ax, x, y, w, h, text, fc=BLUE, fs=11, bold=False, ls='-'):
    p = FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle='square,pad=0', fc=fc, ec=NAVY, lw=1.8, ls=ls)
    ax.add_patch(p)
    ax.text(x, y, text, ha='center', va='center', fontsize=fs, color='#111', fontweight='bold' if bold else 'normal', linespacing=1.35)


def para(ax, x, y, w, h, text, fc=GREEN, fs=11):
    s = 0.18 * h
    pts = [(x - w / 2 + s, y + h / 2), (x + w / 2 + s, y + h / 2), (x + w / 2 - s, y - h / 2), (x - w / 2 - s, y - h / 2)]
    ax.add_patch(Polygon(pts, closed=True, fc=fc, ec=NAVY, lw=1.8))
    ax.text(x, y, text, ha='center', va='center', fontsize=fs, linespacing=1.35)


def term(ax, x, y, w, h, text, fs=11):
    ax.add_patch(Ellipse((x, y), w, h, fc=GREEN, ec=NAVY, lw=1.8))
    ax.text(x, y, text, ha='center', va='center', fontsize=fs)


def diamond(ax, x, y, w, h, text, fs=10.5):
    pts = [(x, y + h / 2), (x + w / 2, y), (x, y - h / 2), (x - w / 2, y)]
    ax.add_patch(Polygon(pts, closed=True, fc=CREAM, ec=NAVY, lw=1.8))
    ax.text(x, y, text, ha='center', va='center', fontsize=fs, linespacing=1.3)


def arrow(ax, x1, y1, x2, y2, label=None, lx=0, ly=0, color=NAVY, ls='-'):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=16, lw=1.8, color=color, ls=ls))
    if label:
        ax.text((x1 + x2) / 2 + lx, (y1 + y2) / 2 + ly, label, fontsize=10.5, color='#1d4e89', fontweight='bold', ha='center')


def poly(ax, pts, color=NAVY, ls='-'):
    xs, ys = zip(*pts[:-1])
    ax.plot(list(xs) + [pts[-1][0]], list(ys) + [pts[-1][1]], color=color, lw=1.8, ls=ls) if False else None
    for (a, b), (c, d) in zip(pts[:-2], pts[1:-1]):
        ax.plot([a, c], [b, d], color=color, lw=1.8, ls=ls)
    arrow(ax, pts[-2][0], pts[-2][1], pts[-1][0], pts[-1][1], color=color, ls=ls)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=170, bbox_inches='tight', facecolor='white')
    plt.close(fig)



# ---------------------------------------------------------------- sơ đồ khối hệ thống
fig, ax = plt.subplots(figsize=(13, 8.4))
ax.set_xlim(0, 13); ax.set_ylim(0, 8.4); ax.axis('off')
ax.add_patch(FancyBboxPatch((0.2, 0.2), 3.0, 8.0, boxstyle='round,pad=0.02,rounding_size=0.1', fc='#fbfcfe', ec='#8aa1bd', ls='--'))
ax.text(1.7, 7.9, 'Phần cứng (mô phỏng)', ha='center', fontweight='bold', color=NAVY)
box(ax, 1.7, 7.0, 2.6, 0.8, 'Dãy cảm biến trước\n8 mắt, cách trục 100 mm', fc=GREY, fs=9.5)
box(ax, 1.7, 5.9, 2.6, 0.8, 'Dãy cảm biến sau\n8 mắt, cách trục 30 mm', fc=GREY, fs=9.5)
box(ax, 1.7, 4.8, 2.6, 0.8, 'Encoder N20\n1400 xung/vòng bánh', fc=GREY, fs=9.5)
box(ax, 1.7, 3.6, 2.6, 0.9, '2 động cơ N20 + driver\nτ = 100 ms (150 ms có tải)\ntrễ 20 ms, ω_max = 31,4 rad/s', fc=GREY, fs=9.5)
box(ax, 1.7, 2.4, 2.6, 0.8, 'Servo càng nâng + ToF\nVL53L0X (vật cản)', fc=GREY, fs=9.5)
box(ax, 1.7, 1.2, 2.6, 0.8, 'IMU MPU6050 (gyro)\nESP32 Wi-Fi / Bluetooth', fc=GREY, fs=9.5)
ax.add_patch(FancyBboxPatch((3.6, 0.2), 9.2, 8.0, boxstyle='round,pad=0.02,rounding_size=0.1', fc='#fbfefb', ec='#87b096', ls='--'))
ax.text(8.2, 7.9, 'Phần mềm trên ESP32 (chu kỳ điều khiển Ts = 10 ms)', ha='center', fontweight='bold', color='#24563a')
box(ax, 5.3, 6.45, 2.8, 1.6, 'Xử lý cảm biến\n• trọng tâm có trọng số\n• e_y, e_θ từ hai dãy\n• tự học độ rộng line\n• phát hiện giao lộ / đầu mút', fs=9.2)
box(ax, 5.3, 4.2, 2.8, 1.6, 'Ước lượng tư thế\n• encoder + gyro\n• tự hiệu chuẩn b, ε, bias\n• neo hướng tại nút đã biết', fs=9.2)
box(ax, 8.6, 6.45, 3.0, 1.6, 'Bộ điều khiển bám line\nPID (Kp, Ki, Kd)  |  LQI (Bryson)\nv giảm khi có vật cản\n→ ω, v → tốc độ hai bánh', fc=ORANGE, fs=9.2)
box(ax, 8.6, 4.2, 3.0, 1.6, 'Hành vi (máy trạng thái)\nbám line → vào tâm nút →\nquét (nút mới) → chọn nhánh →\nchờ / quay đầu khi bị chặn', fs=9.2)
box(ax, 11.5, 6.45, 2.2, 1.6, 'Bản đồ topo\nnhận dạng nút\nnhiều giả thuyết\npose graph SE(2)', fc=GREEN, fs=9.2)
box(ax, 11.5, 4.2, 2.2, 1.6, 'Lập kế hoạch\nDFS khám phá\nDijkstra (tránh cạnh bị chặn)\nQHĐ Held–Karp', fc=GREEN, fs=9.2)
box(ax, 8.6, 1.5, 7.6, 1.2, 'GUI: vẽ sân line, học map, gán trạm cho nút đã học, thêm đơn lấy → trả,\nđổi PID / LQR và tham số khi đang chạy, bật kịch bản tình huống', fc=CREAM, fs=9.5)
arrow(ax, 3.0, 7.0, 3.9, 6.8); arrow(ax, 3.0, 5.9, 3.9, 6.1); arrow(ax, 3.0, 4.8, 3.9, 4.4)
arrow(ax, 6.7, 6.45, 7.1, 6.45); arrow(ax, 6.7, 4.2, 7.1, 4.2)
arrow(ax, 10.1, 4.6, 10.4, 6.0); arrow(ax, 10.4, 4.2, 10.1, 4.2); arrow(ax, 11.5, 5.65, 11.5, 5.0)
arrow(ax, 8.6, 5.4, 8.6, 5.65)
poly(ax, [(8.6, 7.25), (8.6, 7.6), (3.4, 7.6), (3.4, 3.6), (3.0, 3.6)])
ax.text(6.0, 7.68, 'tốc độ bánh đặt (PWM)', fontsize=8.5, color=NAVY)
poly(ax, [(7.6, 3.4), (7.6, 2.55), (3.0, 2.55)])
ax.text(4.3, 2.63, 'lệnh nâng / hạ', fontsize=8.5, color=NAVY)
arrow(ax, 8.6, 2.1, 8.6, 3.4); arrow(ax, 4.8, 1.5, 3.0, 1.2)
save(fig, 'system_block.png')

# ---------------------------------------------------------------- lưu đồ nhiệm vụ tổng thể
fig, ax = plt.subplots(figsize=(8.4, 11.6))
ax.set_xlim(0, 8.4); ax.set_ylim(0, 11.6); ax.axis('off')
X = 3.6
term(ax, X, 11.1, 2.6, 0.7, 'Bắt đầu (bản đồ rỗng)')
box(ax, X, 10.1, 4.4, 0.8, 'Bám line từ vị trí đặt robot,\ntự học số mắt đen điển hình')
box(ax, X, 9.0, 4.4, 0.85, 'Đến nút: tiến vào tâm; nút mới thì quét,\nnhận dạng nút, pose graph (Hình 3.7)')
diamond(ax, X, 7.75, 3.6, 1.1, 'Còn cổng chưa\nkhám phá?', fs=10)
box(ax, 7.3, 9.0, 2.0, 0.9, 'Đi cổng chưa khám phá\n(DFS) hoặc Dijkstra\nđến nút gần nhất', fc=GREY, fs=9)
poly(ax, [(X + 1.8, 7.75), (7.3, 7.75), (7.3, 8.55)])
ax.text(6.2, 7.9, 'Còn', color='#1d4e89', fontweight='bold')
poly(ax, [(7.3, 9.45), (7.3, 10.1), (X + 2.2, 10.1)])
box(ax, X, 6.5, 4.4, 0.8, 'Bản đồ hoàn chỉnh → hiển thị lên GUI;\nngười dùng gán trạm cho nút')
box(ax, X, 5.4, 4.4, 0.8, 'Nhập danh sách đơn (lấy > trả);\nDijkstra giữa mọi cặp trạm')
box(ax, X, 4.3, 4.4, 0.8, 'Sắp thứ tự đơn bằng QHĐ Held–Karp\n(chở 1 kiện mỗi lần)')
box(ax, X, 3.2, 4.4, 0.8, 'Với từng đơn: đi tới trạm lấy → dừng →\nnâng càng → đi tới trạm trả → hạ càng')
diamond(ax, X, 2.0, 3.6, 1.0, 'Hết đơn?', fs=10)
term(ax, X, 0.7, 2.6, 0.65, 'Kết thúc')
for y1, y2 in [(10.75, 10.5), (9.7, 9.43), (8.57, 8.3), (6.1, 5.8), (5.0, 4.7), (3.9, 3.6), (2.8, 2.5), (1.5, 1.03)]:
    arrow(ax, X, y1, X, y2)
arrow(ax, X, 7.2, X, 6.9, 'Hết', -0.5, 0)
poly(ax, [(X - 1.8, 2.0), (0.6, 2.0), (0.6, 3.2), (X - 2.2, 3.2)])
ax.text(0.8, 2.15, 'Chưa', color='#1d4e89', fontweight='bold')
ax.text(X + 0.3, 1.2, 'Hết', color='#1d4e89', fontweight='bold')
save(fig, 'flow_mission.png')

# ---------------------------------------------------------------- lưu đồ xử lý nút và nhận dạng
fig, ax = plt.subplots(figsize=(10.5, 12.4))
ax.set_xlim(0, 10.5); ax.set_ylim(0, 12.4); ax.axis('off')
X = 4.2
para(ax, X, 11.8, 4.8, 0.7, 'Sự kiện: giao lộ (≥ 2n̄+1 mắt đen) hoặc đầu mút', fs=10)
box(ax, X, 10.85, 4.8, 0.75, 'Tiến chậm x_f để trục bánh vào tâm nút, dừng', fs=10)
box(ax, X, 9.85, 4.8, 0.85, 'Cạnh đã biết & chiều dài khớp? → nút dự đoán Y,\nKHÔNG quét: nắn hướng theo cổng vào, cập nhật ε̂', fs=9.5)
diamond(ax, X, 8.7, 2.8, 0.95, 'Bậc = 2 ?', fs=10)
box(ax, 8.6, 8.7, 2.6, 0.75, 'Không phải nút:\nbám tiếp nhánh còn lại', fc=GREY, fs=9)
arrow(ax, X + 1.4, 8.7, 7.3, 8.7, 'Có', 0, 0.12)
diamond(ax, X, 7.45, 3.6, 1.05, 'Quét nhánh (cắt lại\nnhánh đầu → hiệu chuẩn b)\nkhớp nút dự đoán?', fs=8.5)
box(ax, 8.6, 7.45, 2.6, 0.95, 'Nút dự đoán Y:\nkhớp bậc, chữ ký góc,\nchiều dài → neo odometry', fc=GREEN, fs=9)
arrow(ax, X + 1.8, 7.45, 7.3, 7.45, 'Có', 0, 0.12)
box(ax, X, 6.1, 5.4, 1.05, 'Ứng viên = nút cũ cùng bậc, chữ ký góc khớp,\ncổng đi vào CHƯA khám phá (ràng buộc topo);\nP(X) ∝ N(‖p̂ − p_X‖; σ²),  σ = σ₀ + k·s;  thêm giả thuyết "nút mới"', fs=9.5)
diamond(ax, X, 4.7, 3.4, 1.0, 'max P ≥ 0,9 ?', fs=10)
box(ax, 8.6, 4.7, 2.6, 0.9, 'Tạo nút mới hoặc khép vòng\n→ tối ưu pose graph SE(2)\n→ neo tư thế', fc=GREEN, fs=9)
arrow(ax, X + 1.7, 4.7, 7.3, 4.7, 'Có', 0, 0.12)
box(ax, X, 3.3, 5.4, 1.05, 'Kiểm chứng chủ động: tạo nút tạm T; đi theo nhánh mà\ngiả thuyết X đã biết đích Y (chiều dài L_XY);\nso sánh bậc, chữ ký góc và chiều dài đo được', fs=9.5)
diamond(ax, X, 1.85, 3.0, 1.0, 'Khớp Y ?', fs=10)
box(ax, 8.6, 1.85, 2.6, 0.9, 'Gộp T vào X,\nđang ở Y (đã biết)', fc=GREEN, fs=9)
arrow(ax, X + 1.5, 1.85, 7.3, 1.85, 'Có', 0, 0.12)
box(ax, X, 0.55, 5.0, 0.75, 'T là nút mới; nút vừa tới xử lý như quan sát mới', fc=ORANGE, fs=9.5)
for y1, y2 in [(11.45, 11.23), (10.47, 10.28), (9.42, 9.18), (8.22, 7.98), (6.92, 6.63), (5.57, 5.2), (4.2, 3.83), (2.77, 2.35), (1.35, 0.93)]:
    arrow(ax, X, y1, X, y2)
ax.text(X + 0.15, 8.08, 'Không', color='#1d4e89', fontweight='bold')
ax.text(X + 0.15, 6.75, 'Không', color='#1d4e89', fontweight='bold')
ax.text(X + 0.15, 4.0, 'Không', color='#1d4e89', fontweight='bold')
ax.text(X + 0.15, 1.15, 'Không', color='#1d4e89', fontweight='bold')
save(fig, 'flow_node.png')
print('ok')


# ---------------------------------------------------------------- hình học robot, đường thử, sân
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from sim.robot import RobotParams
from sim.world import test_track, warehouse_map, grid_map
P = RobotParams()
fig, ax = plt.subplots(figsize=(7.5, 5.6))
ax.plot([-0.04, 0.12, 0.12, -0.04, -0.04], [0.065, 0.05, -0.05, -0.065, 0.065], 'k', lw=2)
for yy in (P.b / 2, -P.b / 2):
    ax.add_patch(FancyBboxPatch((-0.0215, yy - 0.006), 0.043, 0.012, boxstyle='square,pad=0', fc='#555', ec='none'))
ax.plot(np.full(8, P.x_front), P.offsets, 'o', color='#d62728', ms=8, label='Dãy trước (8 mắt, bước 12 mm)')
ax.plot(np.full(8, P.x_rear), P.offsets, 'o', color='#1f77b4', ms=8, label='Dãy sau (8 mắt)')
ax.plot(0, 0, 'k+', ms=14, mew=1.5)
ax.annotate('', (0.06, 0), (0, 0), arrowprops=dict(arrowstyle='-|>', color='green', lw=2))
ax.text(0.03, 0.006, 'hướng chạy', color='green', ha='center')
ax.annotate('', (P.x_front, -0.075), (0, -0.075), arrowprops=dict(arrowstyle='<->'))
ax.text(P.x_front / 2, -0.084, f'x_f = {P.x_front * 1000:.0f} mm', ha='center')
ax.annotate('', (P.x_rear, 0.075), (0, 0.075), arrowprops=dict(arrowstyle='<->'))
ax.text(P.x_rear / 2, 0.079, f'x_r = {P.x_rear * 1000:.0f} mm', ha='center')
ax.annotate('', (-0.055, P.b / 2), (-0.055, -P.b / 2), arrowprops=dict(arrowstyle='<->'))
ax.text(-0.062, 0, f'b = {P.b * 1000:.0f} mm', rotation=90, va='center', ha='right')
ax.set_aspect('equal'); ax.set_xlim(-0.09, 0.15); ax.set_ylim(-0.1, 0.1); ax.grid(alpha=0.3); ax.set_xlabel('[m]')
ax.legend(loc='upper right', fontsize=9)
save(fig, 'robot_geometry.png')
plt.close(fig)

fig, axs = plt.subplots(1, 4, figsize=(15, 3.8))
for ax, k, t in zip(axs, ['mixed', 'wave', 'circle_0.15', 'circle_0.5'],
                    ['(a) Hỗn hợp – chỉnh định', '(b) Sóng – kiểm tra', '(c) Tròn R = 0,15 m – kiểm tra', '(d) Tròn R = 0,5 m – kiểm tra']):
    p = test_track(k).polylines[0]
    ax.plot(p[:, 0], p[:, 1], 'k', lw=3); ax.plot(*p[0], 'go', ms=8); ax.plot(*p[-1], 'rs', ms=8)
    ax.set_aspect('equal'); ax.set_title(t, fontsize=10); ax.set_xlabel('x [m]'); ax.grid(alpha=0.3)
save(fig, 'tracks.png')
plt.close(fig)
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
for ax, w, t in zip(axs, [warehouse_map(), grid_map()], ['(a) Sân kho: 11 nút, 11 cạnh, 1 chu trình', '(b) Sân lưới 3×3: 13 nút, 16 cạnh']):
    for p in w.polylines:
        ax.plot(p[:, 0], p[:, 1], 'k', lw=3)
    for k, Pn in w.nodes.items():
        ax.plot(*Pn, 'o', color='#1f77b4' if k.startswith(('J', 'G')) else '#ff7f0e', ms=8)
        ax.text(Pn[0] + 0.05, Pn[1] + 0.05, k, fontsize=9)
    ax.set_aspect('equal'); ax.set_title(t); ax.set_xlabel('x [m]'); ax.grid(alpha=0.3)
save(fig, 'maps.png')
plt.close(fig)
