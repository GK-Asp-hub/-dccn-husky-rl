"""
LQR-контроллер для differential-drive robot в 2D.

Используется как Residual Policy: корректирует action TD3
на основе курсовой ошибки относительно направления на цель.

Кинематика дифпривода (мировая СК):
    x_dot     = v * cos(theta)
    y_dot     = v * sin(theta)
    theta_dot = omega

Динамика ошибки в локальной СК робота, после преобразования
координат и линеаризации вокруг рабочей точки
(см. Siciliano et al. 2009, гл. 11.6, формула 11.62):
    e_y_dot     = v_ref * e_theta
    e_theta_dot = -omega_tilde

где omega_tilde = omega - omega_ref — отклонение управления от номинала.
В нашей постановке опорная траектория — прямая, проведённая из текущего
положения робота на цель; она не вращается, поэтому omega_ref = 0,
и omega_tilde = omega.

State error:
    [e_y, e_theta]    — cross-track + heading error (двумерный)
Control:
    [omega]           — угловая скорость (поправка к omega от TD3)

Матрицы линеаризованной системы:
    A = [[0, v_ref],   B = [[ 0],
         [0,    0]]        [-1]]

LQR находит K такой, что u = -K * x_err минимизирует
    J = integral_0^inf (x_err^T Q x_err + u^T R u) dt

где Q, R — веса штрафов (см. конструктор).

Cross-track error e_y в нашей постановке всегда равен нулю:
опорная прямая по построению проходит через робота. Контроллер фактически
стабилизирует только курс (e_theta), но коэффициент усиления k_theta
обоснован двумерной задачей LQR через ARE — не подобран эвристически
как в PID-подходе. При появлении планировщика пути e_y оживёт без
переделки матриц A, B (расширение за счёт строки/колонки с omega_ref).

Линейная скорость v не управляется LQR — её формирует TD3 как
часть [left_wheel_vel, right_wheel_vel]. Это decoupled control:
продольная динамика управляется через среднее колёс (TD3),
поперечная — через разность колёс (LQR-поправка).

Литература:
    Siciliano et al. 2009. Robotics: Modelling, Planning and Control. Гл. 11.
    De Luca, Oriolo, Vendittelli 2001. Control of wheeled mobile robots.
    Kanayama et al. 1990. A stable tracking control method for an
    autonomous mobile robot.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_continuous_are


class LQRDiffDriveLateral:
    """Боковой LQR для differential drive.

    Задача: робот должен двигаться к цели по направлению, заданному
    вектором (текущая_позиция -> цель). На каждом шаге опорное
    направление пересчитывается заново (опорной траектории как
    последовательности точек во времени в этой реализации нет).

    LQR корректирует omega, чтобы уменьшить:
    - e_y: cross-track error в локальной СК робота
           (всегда 0 в текущей постановке — опорная прямая проходит
           через робота по построению)
    - e_theta: отклонение курса от направления на цель (heading error)

    Двумерное состояние сохранено намеренно: матрица K вычисляется
    из ARE двумерной задачи, что обеспечивает обоснованный коэффициент
    k_theta при e_theta. При появлении планировщика пути e_y начнёт
    принимать ненулевые значения без изменения структуры контроллера.

    Выход: поправка к нормированной omega в [-max_correction, +max_correction].
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

        # Линеаризованная модель (e_theta = desired_yaw - robot_yaw,
        # поэтому d/dt e_theta = -omega при постоянном desired_yaw):
        #   d/dt [e_y]      [0  v_ref] [e_y]      [ 0]
        #   d/dt [e_theta] = [0    0  ] [e_theta] + [-1] * omega
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

        # Решаем непрерывное уравнение Риккати (ARE) — однократно при инициализации.
        # self.K сохраняется и используется на каждом шаге без пересчёта,
        # что отличает LQR от MPC (где оптимизация — на каждом такте).
        P = solve_continuous_are(A, B, Q, R)
        # Оптимальное усиление K
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
        1. Вектор (текущая_позиция -> цель) задаёт желаемое направление
           движения в мировой СК.
        2. Опорная прямая в этой постановке проходит через робота по построению,
           поэтому cross-track error e_y = 0 точно (не приближение).
        3. Heading error e_theta = желаемый_курс - текущий_курс,
           свёрнут в [-pi, pi].
        4. u = -K @ [e_y, e_theta]^T, где K вычислена один раз в __init__
           из двумерной задачи LQR.
        5. Выход клипуется в [-max_correction, +max_correction] для гарантии,
           что LQR-поправка остаётся именно поправкой (не доминирует
           над политикой TD3).
        """
        # Направление на цель в мировой СК
        dx = goal_x - robot_x
        dy = goal_y - robot_y
        desired_yaw = np.arctan2(dy, dx)

        # Угол ошибки курса, свёрнут в [-pi, pi]
        theta_err = self._wrap_angle(desired_yaw - robot_yaw)

        # Cross-track error: в текущей постановке (опорная прямая проходит
        # через робота) e_y равен нулю по построению. Контроллер вырождается
        # в стабилизацию курса; коэффициент k_theta при этом обоснован
        # двумерной задачей LQR (см. матрицы A, B в __init__), а не подбирается
        # эвристически. При появлении планировщика пути эта переменная
        # начнёт принимать ненулевые значения без изменения архитектуры.
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
