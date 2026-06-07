"""
LidarAvoid — реактивный обход препятствий по сырому лидару.

Когда лидар робота фиксирует препятствие ближе порога, контроллер перехватывает
управление и уводит робота в сторону наибольшего просвета (отворачивает на 90 градусов
от ближайшего препятствия). Когда все релевантные лучи свободны, управление возвращается
базовой политике (TD3 + LQR).

Роль в статье (§3.6): сенсорный двойник верхнего уровня — в отличие от MapAvoid он НЕ
знает карту и опирается только на сырые показания лидара. Служит для сравнения «модельное
знание о среде (MapAvoid) vs сенсорное (LidarAvoid)». Именно LidarAvoid деградирует при
реалистичном шуме сенсора, тогда как MapAvoid — нет.

Контроллер выдаёт действие (левая_пара, правая_пара) в том же нормированном пространстве
[-1, 1], что и TD3, поэтому в режиме обхода он напрямую подменяет выход политики.

Источники:
- Сценарное переключение поведения (встреча с научруком 08.05.2026, блоки B и C журнала).
- Классический bug-подход к обходу в мобильной робототехнике
  (Borenstein & Koren, 1991; Ulrich & Borenstein VFH, 1998 — концептуально близко, но проще).

Использование:
    avoider = ObstacleAvoidanceController()
    if avoider.should_take_over(obs_lidar):
        action = avoider.compute_action(obs_lidar)
    else:
        action = td3_action  # либо td3 + остаточная коррекция lqr
"""

from __future__ import annotations

import math

import numpy as np


# Пороги по умолчанию. При необходимости переопределяются в конструкторе.
DEFAULT_TAKEOVER_DIST = 1.5       # м: включение, если любой передний луч < этого (d_takeover)
DEFAULT_RELEASE_DIST = 2.5        # м: выключение, если все лучи > этого (гистерезис, d_release)
DEFAULT_FRONT_HALF_WIDTH = 6      # сколько лучей с каждой стороны от направления вперёд (ширина дуги)
DEFAULT_FORWARD_SPEED = 0.35      # нормированная [0,1] скорость вперёд во время обхода
DEFAULT_TURN_GAIN = 1.5           # насколько агрессивно поворачивать к просвету (k_turn)


class ObstacleAvoidanceController:
    """Эвристический обход с гистерезисом между включением и выключением.

    Включение (вход в режим обхода):
        если любой из передних лучей короче `takeover_dist`.

    Выключение (возврат базовой политике):
        если ВСЕ лучи длиннее `release_dist`.

    Рулёжка во время обхода:
        находим направление ближайшего препятствия в передней полусфере;
        поворачиваем на 90 градусов в сторону от него с величиной, пропорциональной углу;
        скорость вперёд фиксирована (`forward_speed`, медленное движение).

    Состояние хранится в экземпляре, чтобы гистерезис работал на протяжении эпизода.
    Между эпизодами сбрасывается вызовом `reset()`.
    """

    def __init__(
        self,
        n_lidar_rays: int = 16,
        takeover_dist: float = DEFAULT_TAKEOVER_DIST,
        release_dist: float = DEFAULT_RELEASE_DIST,
        front_half_width: int = DEFAULT_FRONT_HALF_WIDTH,
        forward_speed: float = DEFAULT_FORWARD_SPEED,
        turn_gain: float = DEFAULT_TURN_GAIN,
    ):
        self.n_lidar_rays = int(n_lidar_rays)
        self.takeover_dist = float(takeover_dist)
        self.release_dist = float(release_dist)
        self.front_half_width = int(front_half_width)
        self.forward_speed = float(forward_speed)
        self.turn_gain = float(turn_gain)
        self._in_avoidance = False     # текущее состояние режима обхода

    def reset(self) -> None:
        # Сброс гистерезиса между эпизодами.
        self._in_avoidance = False

    @property
    def is_active(self) -> bool:
        return self._in_avoidance

    def _front_ray_indices(self) -> list[int]:
        """Индексы лучей, считающихся «передними».

        Раскладка лидара в HuskyObstacleEnv: луч i имеет угол = yaw + 2*pi*i/N,
        поэтому луч 0 смотрит по курсу робота (вперёд). Передние лучи — это
        0, 1, ..., front_half_width и (N-1, N-2, ..., N-front_half_width).
        """
        idx = list(range(0, self.front_half_width + 1))
        idx += list(range(self.n_lidar_rays - self.front_half_width,
                          self.n_lidar_rays))
        return sorted(set(idx))

    def should_take_over(self, lidar: np.ndarray) -> bool:
        """Решить, должен ли режим обхода быть активен на этом шаге.

        Реализует гистерезис:
          - Включение: любой ПЕРЕДНИЙ луч < takeover_dist.
          - Выключение: ВСЕ лучи (передние + боковые, не только передние) >
            release_dist. Так мы не выключаемся, пока препятствие ещё сбоку —
            иначе при следующем повороте курса оно снова стало бы фронтальным.
        """
        lidar_arr = np.asarray(lidar)
        front = lidar_arr[self._front_ray_indices()]

        # Для проверки выключения берём ВСЕ лучи (боковые тоже важны) — полный обзор 360°.
        all_relevant = lidar_arr

        if not self._in_avoidance:
            if float(front.min()) < self.takeover_dist:
                self._in_avoidance = True
        else:
            # Остаёмся в обходе, пока хоть что-то близко (спереди или сбоку).
            if float(all_relevant.min()) > self.release_dist:
                self._in_avoidance = False

        return self._in_avoidance

    def compute_action(self, lidar: np.ndarray) -> np.ndarray:
        """Вычислить действие (левая_пара, правая_пара), уводящее ОТ направления
        на ближайшее препятствие.

        Стратегия: не «ехать туда, где свободно» (это часто недоворачивает и робот
        чиркает по препятствию), а определить направление ближайшего препятствия в
        передней полусфере и отвернуть от него на 90 градусов. Это даёт сильный
        манёвр уклонения, пропорциональный близости препятствия.

        Если ближайшее препятствие прямо по курсу (угол ~ 0), выбираем сторону по
        тому, где суммарно больше просвета.
        """
        lidar = np.asarray(lidar, dtype=np.float32)

        # Углы лучей в локальной СК робота, приведённые к [-pi, pi].
        angles = np.array(
            [2.0 * math.pi * i / self.n_lidar_rays
             for i in range(self.n_lidar_rays)],
            dtype=np.float32,
        )
        angles = np.where(angles > math.pi, angles - 2.0 * math.pi, angles)

        # Рассматриваем только переднюю полусферу при поиске направления препятствия.
        front_mask = np.abs(angles) <= (math.pi / 2.0 + 0.1)
        # Для поиска ближайшего препятствия: непередним лучам ставим большое значение,
        # чтобы они не выиграли argmin.
        front_lidar_for_min = np.where(front_mask, lidar, 1e6)
        closest_idx = int(np.argmin(front_lidar_for_min))
        obstacle_angle = float(angles[closest_idx])

        # Отворачиваем на 90 градусов ОТ препятствия.
        # Препятствие под углом a (в [-pi, pi]) -> цель поворота a +/- pi/2 (та, что в передней полусфере).
        # Знак: положительный поворот = влево (положительная угловая скорость).
        # Препятствие справа (a < 0)  -> поворот влево (цель = a + pi/2 > 0).
        # Препятствие слева  (a > 0)  -> поворот вправо (цель = a - pi/2 < 0).
        # Препятствие по курсу (a ~ 0) -> сторона по суммарному просвету.
        if abs(obstacle_angle) < 1e-3:
            # Разрешение неоднозначности по суммарному просвету слева/справа.
            n_quarter = self.n_lidar_rays // 4
            left_clr = float(lidar[1: n_quarter + 1].mean())
            right_clr = float(lidar[self.n_lidar_rays - n_quarter:].mean())
            if left_clr >= right_clr:
                target_angle = +math.pi / 2.0
            else:
                target_angle = -math.pi / 2.0
        elif obstacle_angle > 0:
            # Препятствие слева -> уходим вправо.
            target_angle = obstacle_angle - math.pi / 2.0
        else:
            # Препятствие справа -> уходим влево.
            target_angle = obstacle_angle + math.pi / 2.0

        # Команда дифпривода: вперёд + поворот; turn > 0 — поворот влево.
        turn = float(np.clip(self.turn_gain * target_angle / math.pi, -1.0, 1.0))

        forward = self.forward_speed

        # Дифференциальная раскладка: левая пара = вперёд - поворот, правая = вперёд + поворот
        # (поворот влево => притормозить левую пару, ускорить правую).
        left = forward - turn
        right = forward + turn

        action = np.array([left, right], dtype=np.float32)
        return np.clip(action, -1.0, 1.0)
