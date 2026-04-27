"""Sanity-тесты для WaypointPlanner. Запуск: python test_waypoint_planner.py"""
import numpy as np
from planners.waypoint_planner import WaypointPlanner


def approx(a, b, tol=1e-5):
    return abs(float(a) - float(b)) < tol


def test_construction_n3():
    """n=3, start=(0,0), goal=(10,0): waypoints = (3.33, 0), (6.67, 0), (10, 0)."""
    pl = WaypointPlanner((0, 0), (10, 0), n_waypoints=3, switch_radius=0.5)
    wps = pl.waypoints
    assert wps.shape == (3, 2), f"shape {wps.shape}"
    assert approx(wps[0, 0], 10 / 3), f"wp0.x = {wps[0, 0]}"
    assert approx(wps[0, 1], 0), f"wp0.y = {wps[0, 1]}"
    assert approx(wps[1, 0], 20 / 3), f"wp1.x = {wps[1, 0]}"
    assert approx(wps[2, 0], 10), f"wp2.x = {wps[2, 0]}"
    assert approx(wps[2, 1], 0), f"wp2.y = {wps[2, 1]}"
    assert pl.current_idx == 0
    assert not pl.is_final()
    print("  OK  construction_n3")


def test_switching():
    """При приближении к waypoint'у на < switch_radius — переключение."""
    pl = WaypointPlanner((0, 0), (10, 0), n_waypoints=3, switch_radius=0.5)
    # В старте активна 1-я подцель (≈3.33, 0)
    wp = pl.current_waypoint((0.0, 0.0))
    assert pl.current_idx == 0
    assert approx(wp[0], 10 / 3)

    # Далеко от 1-й — не переключаемся
    wp = pl.current_waypoint((1.0, 0.0))
    assert pl.current_idx == 0, f"не должны были переключиться, idx={pl.current_idx}"

    # Очень близко к 1-й (d < 0.5) — переключение на 2-ю
    wp = pl.current_waypoint((3.3, 0.0))
    assert pl.current_idx == 1, f"должны были переключиться, idx={pl.current_idx}"
    assert approx(wp[0], 20 / 3)

    # Близко к 2-й → переключение на 3-ю (финальную)
    wp = pl.current_waypoint((6.5, 0.0))
    assert pl.current_idx == 2
    assert pl.is_final()
    assert approx(wp[0], 10)

    # На финальной переключений больше не происходит, даже если мы в неё попали
    wp = pl.current_waypoint((9.9, 0.0))
    assert pl.current_idx == 2, "финальная подцель не должна меняться"
    assert approx(wp[0], 10)

    print("  OK  switching")


def test_n1_is_noop():
    """n=1 — единственная подцель = финальная цель, переключений не бывает."""
    pl = WaypointPlanner((0, 0), (5, 0), n_waypoints=1, switch_radius=0.5)
    assert pl.waypoints.shape == (1, 2)
    assert approx(pl.waypoints[0, 0], 5)
    assert pl.is_final()
    wp = pl.current_waypoint((4.9, 0.0))  # в пределах switch_radius, но финальная
    assert pl.current_idx == 0
    assert approx(wp[0], 5)
    print("  OK  n1_is_noop")


def test_reset_rebuilds():
    """reset() должен пересобрать waypoint'ы и сбросить индекс."""
    pl = WaypointPlanner((0, 0), (10, 0), n_waypoints=2, switch_radius=0.5)
    # Продвинемся до 2-й подцели
    pl.current_waypoint((5.0, 0.0))
    assert pl.current_idx == 1

    pl.reset((0, 0), (0, 8))
    assert pl.current_idx == 0
    wps = pl.waypoints
    assert approx(wps[0, 0], 0)
    assert approx(wps[0, 1], 4)
    assert approx(wps[1, 0], 0)
    assert approx(wps[1, 1], 8)
    print("  OK  reset_rebuilds")


def test_diagonal_trajectory():
    """start=(0,0), goal=(6,8), n=2 → wp0=(3,4), wp1=(6,8)."""
    pl = WaypointPlanner((0, 0), (6, 8), n_waypoints=2, switch_radius=0.5)
    wps = pl.waypoints
    assert approx(wps[0, 0], 3), f"wp0.x = {wps[0, 0]}"
    assert approx(wps[0, 1], 4), f"wp0.y = {wps[0, 1]}"
    assert approx(wps[1, 0], 6)
    assert approx(wps[1, 1], 8)
    # Переключение по евклидовой близости
    wp = pl.current_waypoint((2.8, 3.8))  # d ≈ 0.28 < 0.5
    assert pl.current_idx == 1
    print("  OK  diagonal_trajectory")


def test_peek_current_no_side_effects():
    """peek_current() не должен менять индекс даже если робот рядом."""
    pl = WaypointPlanner((0, 0), (10, 0), n_waypoints=3, switch_radius=0.5)
    # Мы в старте, активна 1-я подцель (10/3, 0)
    assert pl.current_idx == 0

    # Много раз вызываем peek даже когда робот «близко» к подцели.
    # peek не смотрит на робота, он просто возвращает текущую.
    for _ in range(10):
        wp = pl.peek_current()
        assert approx(wp[0], 10 / 3)
        assert pl.current_idx == 0, "peek не должен переключать"

    # current_waypoint — должен переключить, если робот в радиусе
    pl.current_waypoint((3.3, 0.0))
    assert pl.current_idx == 1

    # peek на 2-й подцели
    wp = pl.peek_current()
    assert approx(wp[0], 20 / 3)
    assert pl.current_idx == 1
    print("  OK  peek_current_no_side_effects")


def test_invalid_params():
    """n<1 и switch_radius<=0 должны падать с ValueError."""
    try:
        WaypointPlanner((0, 0), (1, 0), n_waypoints=0, switch_radius=0.5)
    except ValueError:
        pass
    else:
        raise AssertionError("n_waypoints=0 должен был упасть")

    try:
        WaypointPlanner((0, 0), (1, 0), n_waypoints=2, switch_radius=0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("switch_radius=0 должен был упасть")

    print("  OK  invalid_params")


if __name__ == "__main__":
    print("=== WaypointPlanner unit tests ===")
    test_construction_n3()
    test_switching()
    test_n1_is_noop()
    test_reset_rebuilds()
    test_diagonal_trajectory()
    test_peek_current_no_side_effects()
    test_invalid_params()
    print("\nAll tests passed.")
