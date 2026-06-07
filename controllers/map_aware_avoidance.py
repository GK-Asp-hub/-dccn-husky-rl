"""
MapAvoid — обход препятствий по известной карте (верхний/супервизорный уровень STRL).

В отличие от ObstacleAvoidanceController (LidarAvoid), который работает по сырым
показаниям лидара и может пропустить препятствие между лучами (при секторах 22.5°),
этот контроллер использует НАПРЯМУЮ известные положения препятствий. Это оправдано,
когда карта доступна: детерминированные эксперименты, среды с планировщиком, передача
карты со стратегического уровня.

Роль в статье (§3.6): верхний уровень. Активируется индикатором 1_map(s) ∈ {0,1} по
двупороговой логике с гистерезисом и при активации замещает действие нижнего и среднего
уровней, уводя робота от ближайшего препятствия.

Для переноса на реальную платформу список препятствий заменяется картой из SLAM или
выходом модуля восприятия.
"""

from __future__ import annotations

import math
import numpy as np


class MapAwareAvoider:
    """Контроллер обхода, которому положения препятствий известны явно.

    Активация (1_map: 0 -> 1): хотя бы одно препятствие в передней полусфере ближе
        порога `takeover_dist` (дистанция от центра робота до КРАЯ препятствия).
    Снятие (1_map: 1 -> 0): все препятствия дальше порога `release_dist`.
    Зазор release_dist - takeover_dist = 1.0 м задаёт гистерезис против дребезга.

    Действие: повернуть на 90 градусов в сторону от ближайшего препятствия передней
        полусферы и ехать вперёд со скоростью `forward_speed`.
    """

    def __init__(
        self,
        takeover_dist: float = 1.5,   # порог включения d_takeover, м (§3.6)
        release_dist: float = 2.5,    # порог выключения d_release, м (зазор 1.0 м -> гистерезис)
        forward_speed: float = 0.4,   # линейная скорость при активном уровне v_sup, м/с
        turn_gain: float = 1.5,       # усиление поворота k_turn
    ):
        self.takeover_dist = float(takeover_dist)
        self.release_dist = float(release_dist)
        self.forward_speed = float(forward_speed)
        self.turn_gain = float(turn_gain)
        self._in_avoidance = False     # текущее значение индикатора 1_map

    def reset(self):
        # Сброс состояния гистерезиса между эпизодами.
        self._in_avoidance = False

    @property
    def is_active(self) -> bool:
        # Текущее значение индикатора 1_map (True = управляет верхний уровень).
        return self._in_avoidance

    def _obstacle_distances_and_angles(
        self,
        robot_x: float, robot_y: float, robot_yaw: float,
        obstacles: list[tuple[float, float, float]],
    ):
        """Для каждого препятствия вернуть (расстояние_до_края, угол_в_СК_робота).

        Угол в [-pi, pi]: 0 = прямо по курсу, +pi/2 = слева.
        """
        results = []
        for ox, oy, oradius in obstacles:
            # Вектор от робота к центру препятствия в мировой СК.
            dx = ox - robot_x
            dy = oy - robot_y
            dist_to_center = math.hypot(dx, dy)
            # Дистанция до КРАЯ препятствия (центр минус радиус) — её сравниваем с порогами.
            dist_to_edge = dist_to_center - oradius

            # Угол на препятствие в локальной СК робота (мировой угол минус курс).
            world_angle = math.atan2(dy, dx)
            local_angle = world_angle - robot_yaw
            # Привести угол к диапазону [-pi, pi].
            while local_angle > math.pi:
                local_angle -= 2 * math.pi
            while local_angle < -math.pi:
                local_angle += 2 * math.pi
            results.append((dist_to_edge, local_angle))
        return results

    def should_take_over(self, robot_x, robot_y, robot_yaw, obstacles):
        # Обновляет индикатор 1_map (двупороговый гистерезис) и возвращает его значение.
        info = self._obstacle_distances_and_angles(robot_x, robot_y, robot_yaw, obstacles)

        # Минимальная дистанция среди препятствий ПЕРЕДНЕЙ полусферы (|угол| < pi/2 + запас).
        front_dists = [d for d, a in info if abs(a) < math.pi / 2.0 + 0.1]

        if not self._in_avoidance:
            # Выключен -> включаем, если препятствие спереди ближе d_takeover.
            if front_dists and min(front_dists) < self.takeover_dist:
                self._in_avoidance = True
        else:
            # Включён -> выключаем, только когда ВСЕ препятствия (в т.ч. сбоку) дальше d_release.
            all_dists = [d for d, _ in info]
            if not all_dists or min(all_dists) > self.release_dist:
                self._in_avoidance = False

        return self._in_avoidance

    def compute_action(self, robot_x, robot_y, robot_yaw, obstacles) -> np.ndarray:
        """Отвернуть от ближайшего препятствия и медленно ехать вперёд."""
        info = self._obstacle_distances_and_angles(robot_x, robot_y, robot_yaw, obstacles)
        # Ближайшее препятствие по дистанции до края (среди всех).
        closest = min(info, key=lambda x: x[0])
        dist_edge, obstacle_angle = closest

        # Целевое направление ухода: на 90 градусов в сторону от направления на препятствие.
        if abs(obstacle_angle) < 1e-3:
            # Препятствие точно по курсу — выбираем сторону; по умолчанию влево.
            target_angle = +math.pi / 2.0
        elif obstacle_angle > 0:
            target_angle = obstacle_angle - math.pi / 2.0
        else:
            target_angle = obstacle_angle + math.pi / 2.0

        # Поворотная команда ~ k_turn * (целевой угол), нормирована и ограничена [-1, 1].
        turn = float(np.clip(self.turn_gain * target_angle / math.pi, -1.0, 1.0))
        forward = self.forward_speed
        # Перевод (вперёд, поворот) в дифференциальные команды на левую/правую пары колёс.
        left = forward - turn
        right = forward + turn
        return np.clip(np.array([left, right], dtype=np.float32), -1.0, 1.0)
