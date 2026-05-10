"""
experiments/lqr_riccati_iterations.py

Итеративное решение дифференциального уравнения Риккати (continuous RDE)
методом Эйлера для 2D LQR-системы (differential drive lateral controller).

Цель: показать, как P(t) сходится к стационарному решению P_inf, которое
выдаёт scipy.linalg.solve_continuous_are. Это деливерабл к Теме 1.5
(теория LQR, блок 1).

Система:
    dx/dt = A x + B u
    A = [[0, v_ref], [0, 0]]    — линеаризация ошибки (e_y, e_theta)
    B = [[0], [-1]]              — управление omega

Дифференциальное уравнение Риккати (RDE), интегрированное назад во времени:
    -dP/dt = A^T P + P A - P B R^{-1} B^T P + Q

Метод Эйлера, шаг dt назад от P(T) к P(0):
    P(t - dt) = P(t) + dt * (A^T P + P A - P B R^{-1} B^T P + Q)

Стартуем с P(T) = 0 (нулевой терминальный штраф) и итерируем до сходимости.
При T -> infinity dP/dt -> 0 и мы получаем стационарное P_inf,
удовлетворяющее алгебраическому уравнению Риккати (ARE):
    A^T P + P A - P B R^{-1} B^T P + Q = 0
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_continuous_are
import matplotlib.pyplot as plt


# ---- Параметры системы (как в lqr_diff_drive.py) ----
V_REF = 1.0
Q = np.diag([2.0, 1.0])
R = np.array([[5.0]])

A = np.array([
    [0.0, V_REF],
    [0.0, 0.0],
])
B = np.array([
    [0.0],
    [-1.0],
])

# ---- Параметры интегрирования RDE назад во времени ----
DT = 0.01
N_ITER = 5000


def riccati_rhs(P, A, B, Q, R_inv):
    """Правая часть RDE: A^T P + P A - P B R^{-1} B^T P + Q."""
    return A.T @ P + P @ A - P @ B @ R_inv @ B.T @ P + Q


def main():
    R_inv = np.linalg.inv(R)

    P = np.zeros((2, 2))
    history = [P.copy()]

    for _ in range(N_ITER):
        rhs = riccati_rhs(P, A, B, Q, R_inv)
        P = P + DT * rhs
        history.append(P.copy())

    history = np.array(history)

    P_inf = solve_continuous_are(A, B, Q, R)

    print("=" * 60)
    print(f"После {N_ITER} шагов RDE-Эйлер с dt={DT} (горизонт {N_ITER*DT} сек):")
    print(f"P_final =\n{history[-1]}")
    print()
    print(f"Стационарное P из scipy.solve_continuous_are:")
    print(f"P_inf =\n{P_inf}")
    print()
    print(f"Норма (P_final - P_inf): {np.linalg.norm(history[-1] - P_inf):.6e}")
    print("=" * 60)

    K_iter = R_inv @ B.T @ history[-1]
    K_scipy = R_inv @ B.T @ P_inf
    print(f"\nK от итеративного P:  {K_iter.flatten()}")
    print(f"K от scipy P_inf:     {K_scipy.flatten()}")

    t_back = np.arange(N_ITER + 1) * DT

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)

    cells = [
        (0, 0, "P[0,0]  —  вес по e_y^2"),
        (0, 1, "P[0,1]  —  кросс-член e_y * e_theta"),
        (1, 0, "P[1,0]  —  кросс-член (= P[0,1] по симметрии)"),
        (1, 1, "P[1,1]  —  вес по e_theta^2"),
    ]

    for (i, j, title), ax in zip(cells, axes.flat):
        ax.plot(t_back, history[:, i, j], lw=1.8, label="RDE-итерации (Эйлер)")
        ax.axhline(P_inf[i, j], color="red", ls="--", lw=1.2,
                   label="ARE (scipy)")
        ax.set_title(title)
        ax.set_xlabel("горизонт назад от конца, сек")
        ax.set_ylabel("значение элемента P")
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=9)

    fig.suptitle(
        f"Сходимость P(t) к стационарному решению ARE  "
        f"(dt={DT}, v_ref={V_REF}, Q=diag(2,1), R=5)",
        fontsize=11,
    )
    fig.tight_layout()

    out_path = "lqr_riccati_iterations.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"\nГрафик сохранён: {out_path}")


if __name__ == "__main__":
    main()
