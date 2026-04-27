"""Sanity test for LQR controller."""
import numpy as np
from controllers.lqr_diff_drive import LQRDiffDriveLateral, combine_actions

lqr = LQRDiffDriveLateral()
print("K matrix:", lqr.K.flatten())
print("K[y_err] =", lqr.K[0, 0])
print("K[theta_err] =", lqr.K[0, 1])

print()
print("=== Test scenarios ===")

# Робот в (0,0), смотрит вдоль X
scenarios = [
    ("Цель впереди (0 deg)",    (3.0, 0.0),  "~= 0"),
    ("Цель справа-впереди 45°", (3.0, 3.0),  "положительная (поворот влево к цели)"),
    ("Цель справа 90°",         (0.0, 3.0),  "большая положительная"),
    ("Цель слева 90°",          (0.0, -3.0), "большая отрицательная"),
    ("Цель сзади 180°",         (-3.0, 0.0), "максимум (wrap к pi)"),
    ("Цель чуть справа 10°",    (5.0, 0.88), "маленькая положительная"),
]

for label, (gx, gy), expected in scenarios:
    c = lqr.compute_correction(0, 0, 0.0, gx, gy)
    print(f"  {label:30s} goal=({gx:+.2f},{gy:+.2f})  correction={c:+.4f}   expected: {expected}")

print()
print("=== Combine actions ===")
td3 = np.array([0.5, 0.5])
for delta in [-0.5, -0.2, 0.0, 0.2, 0.5]:
    final = combine_actions(td3, lqr_omega_correction=delta, alpha=0.5)
    print(f"  td3=[0.5,0.5] + corr={delta:+.2f} -> {final}")

print()
print("OK")
