"""
LQR-контроллер для differential-drive robot в 2D.

Используется как Residual Policy: корректирует action TD3
на основе отклонения от желаемой траектории (прямая линия к цели).

Динамика робота (state: x, y, theta; control: v_linear, omega_angular):
    x_dot     = v * cos(theta)
    y_dot     = v * sin(theta)
    theta_dot = omega

Линеаризация вокруг «рабочей точки» (движение вдоль оси x со скоростью v_ref):
    x_e_dot     = v_ref * (1)            (по сути постоянный drift, не управляемый)
    y_e_dot     = v_ref * theta_e
    theta_e_dot = omega_e

State error (в СК желаемой траектории):
    [y_err, theta_err, omega_err]     — cross-track + heading + angular rate
Control:
    [omega]

LQR находит K такой, что u = -K * x_err минимизирует квадратичную cost.

Простая реализация: считаем 2x2-систему (y_err, theta_err) и контроллер по omega.
Линейная скорость v оставляется от TD3 (его policy умеет её регулировать по расстоянию).
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_continuous_are


class LQRDiffDriveLateral:
    """Боковой LQR для differential drive.

    Задача: робот должен ехать вдоль линии, соединяющей текущую позицию и цель.
    LQR корректирует omega, чтобы уменьшить:
    - y_err: расстояние от траектории (cross-track error)
    - theta_err: отклонение курса от направления на цель (heading error)

    Выход: небольшая поправка к omega в [-1, 1] (нормированная).
    """

    def __init__(
        self,
        v_ref: float = 1.0,          # опорная линейная скорость (м/с)
        Q_y: float = 2.0,            # стоимость cross-track ошибки
        Q_theta: float = 1.0,        # стоимость heading ошибки
        R_omega: float = 5.0,        # стоимость управления (больше = мягче поправки)
        max_correction: float = 0.3, # ограничение поправки на omega (в норм. единицах)
    ):
        self.v_ref = v_ref
        self.max_correction = max_correction

        # Линеаризованная модель (theta_err = desired_yaw - robot_yaw,
        # поэтому d/dt theta_err = -omega при постоянном desired_yaw):
        #   d/dt [y_err]     [0  v_ref] [y_err]     [ 0]
        #   d/dt [th_err]  = [0    0  ] [th_err]  + [-1] * omega
        A = np.array([
            [0.0, v_ref],
            [0.0, 0.0],
        ])
        B = np.array([
            [0.0],
            [-1.0],
        ])
        Q = np.diag([Q_y, Q_theta])
        R = np.array([[R_omega]])

        # Solve continuous algebraic Riccati equation
        P = solve_continuous_are(A, B, Q, R)
        # Optimal gain
        self.K = np.linalg.inv(R) @ B.T @ P  # shape (1, 2)

    def compute_correction(
        self,
        robot_x: float,
        robot_y: float,
        robot_yaw: float,
        goal_x: float,
        goal_y: float,
    ) -> float:
        """Возвращает поправку к нормированной omega ([-1, 1]).

        Логика:
        1. Вектор от робота к цели определяет желаемое направление.
        2. y_err — перпендикулярное расстояние робота до прямой «текущая поза робота → цель»
           в его локальной СК. Для простого варианта это 0 (мы всегда строим линию из текущей
           точки), поэтому используем упрощённый вариант: целью контроллера становится
           heading alignment.
        3. theta_err — угол между курсом робота и направлением на цель.
        """
        # Направление на цель в мировой СК
        dx = goal_x - robot_x
        dy = goal_y - robot_y
        desired_yaw = np.arctan2(dy, dx)

        # Угол ошибки курса (свёрнут в [-pi, pi])
        theta_err = self._wrap_angle(desired_yaw - robot_yaw)

        # Cross-track error: в нашей постановке (линия из текущей точки в цель)
        # технически 0 — но мы аппроксимируем его как поперечную составляющую
        # смещения, которая накопится, если робот продолжит ехать с текущим курсом.
        # Для минимальной реализации используем только heading error.
        y_err = 0.0

        # state_err = [y_err, theta_err]
        state_err = np.array([y_err, theta_err])

        # u = -K * x (результат 1D-массив длины 1, берём элемент)
        u = -float((self.K @ state_err).item())

        # Ограничение поправки
        u = float(np.clip(u, -self.max_correction, self.max_correction))
        return u

    @staticmethod
    def _wrap_angle(a: float) -> float:
        return float((a + np.pi) % (2 * np.pi) - np.pi)


def combine_actions(
    td3_action: np.ndarray,
    lqr_omega_correction: float,
    alpha: float = 0.5,
) -> np.ndarray:
    """Комбинирует action от TD3 с поправкой от LQR.

    TD3 action: [left_wheel_vel, right_wheel_vel] в [-1, 1]
    LQR поправка: добавка к omega (разнице между колёсами) в [-1, 1]

    Схема Residual Policy:
        final_left  = td3_left  - alpha * lqr_omega_correction
        final_right = td3_right + alpha * lqr_omega_correction

    Т.е. положительная LQR-поправка крутит робота влево (лево тормозит, право ускоряется).
    """
    td3_left, td3_right = float(td3_action[0]), float(td3_action[1])
    delta = alpha * lqr_omega_correction
    final_left = td3_left - delta
    final_right = td3_right + delta
    result = np.array([final_left, final_right], dtype=np.float32)
    return np.clip(result, -1.0, 1.0)
