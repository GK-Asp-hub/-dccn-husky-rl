"""
Генерирует два графика для отчёта Киселёву:
  figures/training_curves_v1_v2_v3.png  — кривые обучения Stage 2a
  figures/planner_effect_v1_vs_v3.png   — «инверсия инверсии» на 30 сидах
"""
import json, os
import numpy as np
import matplotlib.pyplot as plt

FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 180,
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ============================================================
# FIG 1. Кривые обучения Stage 2a: v1, v2, v3
# ============================================================
def load_eval(path):
    d = np.load(path)
    ts = d["timesteps"]
    res = d["results"]          # (N_evals, N_eps)
    mean = res.mean(axis=1)
    std = res.std(axis=1)
    return ts, mean, std

runs = {
    "v1 (баг лидара)":             "runs/husky_obstacle_td3_v1/eval/evaluations.npz",
    "v2 (чистый env, step=0.01)":  "runs/husky_obstacle_td3_v2/eval/evaluations.npz",
    "v3 (чистый env, step=0.36)":  "runs/husky_obstacle_td3_v3_steppen036/eval/evaluations.npz",
}
colors = {"v1": "#d97706", "v2": "#b91c1c", "v3": "#2563eb"}

fig, ax = plt.subplots(figsize=(9, 5))
for label, path in runs.items():
    ts, mean, std = load_eval(path)
    key = label.split()[0]
    c = colors[key]
    ax.plot(ts / 1000, mean, "-", color=c, label=label, linewidth=2)
    ax.fill_between(ts / 1000, mean - std, mean + std, color=c, alpha=0.15)

ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.5)
ax.set_xlabel("Шаги обучения, ×10³")
ax.set_ylabel("Eval reward (среднее по 5 эпизодам)")
ax.set_title("Stage 2a: кривые обучения TD3 в среде с препятствиями")
ax.set_ylim(-2800, 200)
ax.legend(loc="lower right", frameon=True)

# аннотации ключевых точек
ax.annotate("v3 best\n−21.3 @ 190k",
            xy=(190, -21.3), xytext=(155, -600),
            fontsize=9, color=colors["v3"],
            arrowprops=dict(arrowstyle="->", color=colors["v3"], lw=1))
ax.annotate("v1 best\n−63.5 @ 200k",
            xy=(200, -63.5), xytext=(135, -1200),
            fontsize=9, color=colors["v1"],
            arrowprops=dict(arrowstyle="->", color=colors["v1"], lw=1))
ax.annotate("v2 не сошлась\n(0/10 success)",
            xy=(180, -2050), xytext=(70, -2500),
            fontsize=9, color=colors["v2"],
            arrowprops=dict(arrowstyle="->", color=colors["v2"], lw=1))

fig.tight_layout()
out1 = os.path.join(FIG_DIR, "training_curves_v1_v2_v3.png")
fig.savefig(out1)
plt.close(fig)
print(f"wrote {out1}")

# ============================================================
# FIG 2. Planner effect: v1 vs v3 на 30 сидах
# ============================================================
def load_summary(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["summaries"]

v1 = load_summary("ablation_results_obstacle_v1_30seeds.json")
v3 = load_summary("ablation_results_obstacle_v3_30seeds.json")

def parse_rate(s):
    a, b = s.split("/")
    return int(a), int(b)

labels = ["TD3", "TD3+LQR", "+Planner N=2", "+Planner N=3", "+Planner N=5"]
v1_succ = [parse_rate(s["success_rate"])[0] / parse_rate(s["success_rate"])[1] * 100 for s in v1]
v3_succ = [parse_rate(s["success_rate"])[0] / parse_rate(s["success_rate"])[1] * 100 for s in v3]

x = np.arange(len(labels))
w = 0.38

fig, ax = plt.subplots(figsize=(10, 5.2))
bars1 = ax.bar(x - w/2, v1_succ, w, label="v1 (слабая policy, баг)", color="#d97706")
bars3 = ax.bar(x + w/2, v3_succ, w, label="v3 (сильная policy, чистый env)", color="#2563eb")

for bars, vals in ((bars1, v1_succ), (bars3, v3_succ)):
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1.2,
                f"{val:.0f}%", ha="center", fontsize=10)

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Success rate, %")
ax.set_ylim(0, 105)
ax.set_title("Эффект планировщика обратно пропорционален качеству policy (30 сидов)")
ax.legend(loc="upper left", frameon=True)

# подпись разности эффектов
ax.annotate("+13 п.п.\n(×3 к baseline)",
            xy=(3 - w/2, v1_succ[3]), xytext=(2.1, 45),
            fontsize=9.5, color="#d97706",
            arrowprops=dict(arrowstyle="->", color="#d97706", lw=1))
ax.annotate("−7 п.п.",
            xy=(3 + w/2, v3_succ[3]), xytext=(3.4, 55),
            fontsize=9.5, color="#2563eb",
            arrowprops=dict(arrowstyle="->", color="#2563eb", lw=1))

fig.tight_layout()
out2 = os.path.join(FIG_DIR, "planner_effect_v1_vs_v3.png")
fig.savefig(out2)
plt.close(fig)
print(f"wrote {out2}")

# ============================================================
# Печать подтверждающих цифр (для доклада)
# ============================================================
print("\n=== Сверка чисел ===")
print(f"  v1 success: {v1_succ}")
print(f"  v3 success: {v3_succ}")
print(f"  Planner effect v1: N=3 -> {v1_succ[3] - v1_succ[0]:+.1f} п.п.")
print(f"  Planner effect v3: N=3 -> {v3_succ[3] - v3_succ[0]:+.1f} п.п.")
