# Медиа-материалы экспериментов

Видео, GIF-превью и графики траекторий ко всем экспериментам статьи DCCN-2026
(иерархическая STRL-архитектура: **TD3** нижний реактивный уровень + **LQR-residual**
средний уровень + **MapAvoid** верхний уровень по карте).

**Как читать графики траекторий** (вид сверху): зелёная точка — старт, красная
звезда — цель, серые/коричневые круги — препятствия. На одном графике наложены
**все 5 режимов управления** разными цветами; сплошная линия — успех, пунктир —
столкновение.

**Пять режимов** (везде одни и те же):

| Режим | Что включено | Смысл |
|---|---|---|
| `TD3` | только нижний | обученная реактивная политика, действует на каждом шаге |
| `TD3+LQR` | + средний | LQR гасит колебания курса (остаточная коррекция, α=0.2) |
| `TD3+LidarAvoid` | + верхний по лидару | обход по сырому лидару, без карты |
| `TD3+MapAvoid` | + верхний по карте | обход по известным положениям препятствий |
| `TD3+MapAvoid+LQR` | полная тройка | карта решает «куда», LQR сглаживает «как» |

Для каждого ролика есть **связка: график (plot) + GIF-превью + видео (mp4)**.
GIF — лёгкое превью (листается прямо в GitHub без скачивания); mp4 при клике
открывается страницей файла с кнопкой Download/Raw (GitHub не проигрывает mp4 инлайн).

---

# Эксперимент A — препятствия известной геометрии

**Что проверяем:** робот едет из старта в цель, на пути — препятствия фиксированной
геометрии. Сравниваем, какой уровень управления решает задачу. 7 сценариев,
фиксированные старты, сиды 200–209, каждая ячейка детерминирована (повтор даёт тот же исход).
Это материал к **Таблице 1** и **рис. 2** статьи. Код: `experiments/exp_a_scenarios.py`.

**Главный вывод A:** чистый нижний уровень (TD3) едет напрямую и врезается в
препятствие; верхний уровень (MapAvoid) обходит. Какой именно верхний уровень
помогает — зависит от геометрии (см. ниже single vs barrier).

## A.1 Основной сценарий — одно препятствие на пути, по ролику на каждый режим

Один и тот же сценарий, 5 роликов — видно, как добавление уровней меняет поведение.
Общий график всех пяти траекторий: [`exp_a/videos_main/expA_final_trajectories.png`](exp_a/videos_main/expA_final_trajectories.png).

| Режим | GIF | Видео | Что происходит |
|---|---|---|---|
| TD3 | [gif](exp_a/videos_main_gif/01_TD3_only.gif) | [mp4](exp_a/videos_main/01_TD3_only.mp4) | едет прямо в препятствие — столкновение |
| TD3+LQR | [gif](exp_a/videos_main_gif/02_TD3_LQR.gif) | [mp4](exp_a/videos_main/02_TD3_LQR.mp4) | курс ровнее, но всё равно врезается (LQR не объезжает) |
| TD3+LidarAvoid | [gif](exp_a/videos_main_gif/03_TD3_LidarAvoid.gif) | [mp4](exp_a/videos_main/03_TD3_LidarAvoid.mp4) | реактивно уклоняется по лидару |
| TD3+MapAvoid | [gif](exp_a/videos_main_gif/04_TD3_MapAvoid.gif) | [mp4](exp_a/videos_main/04_TD3_MapAvoid.mp4) | обходит по карте, доезжает до цели |
| TD3+MapAvoid+LQR | [gif](exp_a/videos_main_gif/05_TD3_MapAvoid_LQR.gif) | [mp4](exp_a/videos_main/05_TD3_MapAvoid_LQR.mp4) | обход + сглаженный курс — самый чистый проход |

## A.2 Отдельные сценарии (геометрия посложнее)

Связка plot + gif + video на каждый ролик. Графики содержат все 5 режимов сразу.

| Сценарий (режим в ролике) | Plot (все режимы) | GIF | Видео | Что происходит |
|---|---|---|---|---|
| two_offset (MapAvoid) | [plot](exp_a/trajectories/two_offset_trajectories.png) | [gif](exp_a/videos_scenarios_gif/two_offset__TD3_MapAvoid.gif) | [mp4](exp_a/videos_scenarios/two_offset__TD3_MapAvoid.mp4) | два смещённых препятствия — Map обходит оба |
| three_corridor (MapAvoid) | [plot](exp_a/trajectories/three_corridor_trajectories.png) | [gif](exp_a/videos_scenarios_gif/three_corridor__TD3_MapAvoid.gif) | [mp4](exp_a/videos_scenarios/three_corridor__TD3_MapAvoid.mp4) | коридор из трёх — нужен глобальный контекст карты |
| barrier_with_gap (LQR) | [plot](exp_a/trajectories/barrier_with_gap_trajectories.png) | [gif](exp_a/videos_scenarios_gif/barrier_with_gap__TD3_LQR.gif) | [mp4](exp_a/videos_scenarios/barrier_with_gap__TD3_LQR.mp4) | барьер с щелью — нижний+LQR проскакивают в проход |
| barrier_with_gap (MapAvoid) | [plot](exp_a/trajectories/barrier_with_gap_trajectories.png) | [gif](exp_a/videos_scenarios_gif/barrier_with_gap__TD3_MapAvoid.gif) | [mp4](exp_a/videos_scenarios/barrier_with_gap__TD3_MapAvoid.mp4) | Map жадно обходит весь барьер вместо щели |
| slalom (MapAvoid) | [plot](exp_a/trajectories/slalom_trajectories.png) | [gif](exp_a/videos_scenarios_gif/slalom__TD3_MapAvoid.gif) | [mp4](exp_a/videos_scenarios/slalom__TD3_MapAvoid.mp4) | змейка — не решает никто (граница применимости) |
| diagonal_goal (MapAvoid) | [plot](exp_a/trajectories/diagonal_goal_trajectories.png) | [gif](exp_a/videos_scenarios_gif/diagonal_goal__TD3_MapAvoid.gif) | [mp4](exp_a/videos_scenarios/diagonal_goal__TD3_MapAvoid.mp4) | цель по диагонали (контроль) — решают все |
| wide_obstacle (MapAvoid) | [plot](exp_a/trajectories/wide_obstacle_trajectories.png) | [gif](exp_a/videos_scenarios_gif/wide_obstacle__TD3_MapAvoid.gif) | [mp4](exp_a/videos_scenarios/wide_obstacle__TD3_MapAvoid.mp4) | широкое препятствие — Map обходит с краю |
| wide_obstacle (LidarAvoid) | [plot](exp_a/trajectories/wide_obstacle_trajectories.png) | [gif](exp_a/videos_scenarios_gif/wide_obstacle__TD3_LidarAvoid.gif) | [mp4](exp_a/videos_scenarios/wide_obstacle__TD3_LidarAvoid.mp4) | реакция по лидару на то же препятствие |
| single_on_line | [plot](exp_a/trajectories/single_on_line_trajectories.png) | (см. A.1 — это и есть основной сценарий) | | одно препятствие строго на линии до цели |

---

# Эксперимент B — поверхности с разным трением

**Что проверяем:** меняем трение пола под колёсами и смотрим, как это влияет на
управление. Трение задаётся в коде (`envs/husky_surface_env.py`, `SURFACE_PRESETS`):
**NORMAL μ=0.8 · ICE μ=0.15 (скользко) · SAND μ=0.9 + сопротивление качению (вязко)**.
Модель TD3 та же (v3), без переобучения. Материал к **Таблице 2**. Код: `experiments/exp_b_global_surfaces.py`.

**Главный вывод B:** на льду с препятствием задачу решает **только полная тройка
TD3+MapAvoid+LQR**. Без среднего уровня (LQR) робот на льду строит верную
траекторию обхода, но проскальзывает в поворотах и врезается — сглаживание курса критично.

## B.1 Глобальная поверхность (вся арена — один тип)

3 поверхности × {пусто, +препятствие}. В роликах показан репрезентативный режим
(на пустой арене — TD3+LQR; с препятствием — полная тройка); на графиках — все 5 режимов.

| Конфигурация | Plot (все режимы) | GIF | Видео | Что происходит |
|---|---|---|---|---|
| NORMAL, пусто | [plot](exp_b/plots_global/NORMAL_empty_trajectories.png) | [gif](exp_b/videos_global_gif/NORMAL_empty__TD3_LQR.gif) | [mp4](exp_b/videos_global/NORMAL_empty__TD3_LQR.mp4) | обычный пол, прямой проезд к цели |
| NORMAL, +препятствие | [plot](exp_b/plots_global/NORMAL_with_obstacle_trajectories.png) | [gif](exp_b/videos_global_gif/NORMAL_with_obstacle__TD3_MapAvoid_LQR.gif) | [mp4](exp_b/videos_global/NORMAL_with_obstacle__TD3_MapAvoid_LQR.mp4) | обход по карте на хорошем сцеплении |
| ICE, пусто | [plot](exp_b/plots_global/ICE_empty_trajectories.png) | [gif](exp_b/videos_global_gif/ICE_empty__TD3_LQR.gif) | [mp4](exp_b/videos_global/ICE_empty__TD3_LQR.mp4) | лёд без препятствия — доезжает, но скользит |
| **ICE, +препятствие** | [plot](exp_b/plots_global/ICE_with_obstacle_trajectories.png) | [gif](exp_b/videos_global_gif/ICE_with_obstacle__TD3_MapAvoid_LQR.gif) | [mp4](exp_b/videos_global/ICE_with_obstacle__TD3_MapAvoid_LQR.mp4) | **ключевой кейс**: проходит только полная тройка |
| SAND, пусто | [plot](exp_b/plots_global/SAND_empty_trajectories.png) | [gif](exp_b/videos_global_gif/SAND_empty__TD3_LQR.gif) | [mp4](exp_b/videos_global/SAND_empty__TD3_LQR.mp4) | песок — едет медленнее, но уверенно |
| SAND, +препятствие | [plot](exp_b/plots_global/SAND_with_obstacle_trajectories.png) | [gif](exp_b/videos_global_gif/SAND_with_obstacle__TD3_MapAvoid_LQR.gif) | [mp4](exp_b/videos_global/SAND_with_obstacle__TD3_MapAvoid_LQR.mp4) | вязкий обход по карте |

## B.2 Локальные полосы другого трения (поперёк пути)

Полосы льда/песка на части пути — переход «сцепление → скольжение → сцепление».

| Конфигурация | Plot (все режимы) | GIF | Видео | Что происходит |
|---|---|---|---|---|
| ice_strip | [plot](exp_b/plots_strips/ice_strip_trajectories.png) | [gif](exp_b/videos_strips_gif/ice_strip__TD3_LQR.gif) | [mp4](exp_b/videos_strips/ice_strip__TD3_LQR.mp4) | проезд через ледяную полосу |
| ice_strip + препятствие | [plot](exp_b/plots_strips/ice_strip_with_obstacle_trajectories.png) | [gif](exp_b/videos_strips_gif/ice_strip_with_obstacle__TD3_MapAvoid.gif) | [mp4](exp_b/videos_strips/ice_strip_with_obstacle__TD3_MapAvoid.mp4) | обход на скользкой полосе |
| sand_strip + препятствие | [plot](exp_b/plots_strips/sand_strip_with_obstacle_trajectories.png) | [gif](exp_b/videos_strips_gif/sand_strip_with_obstacle__TD3_MapAvoid.gif) | [mp4](exp_b/videos_strips/sand_strip_with_obstacle__TD3_MapAvoid.mp4) | обход на вязкой полосе |
| obstacle_then_ice | [plot](exp_b/plots_strips/obstacle_then_ice_trajectories.png) | [gif](exp_b/videos_strips_gif/obstacle_then_ice__TD3_MapAvoid_LQR.gif) | [mp4](exp_b/videos_strips/obstacle_then_ice__TD3_MapAvoid_LQR.mp4) | сперва обход, затем лёд — комбинированный кейс |
| ice_and_sand | [plot](exp_b/plots_strips/ice_and_sand_trajectories.png) | [gif](exp_b/videos_strips_gif/ice_and_sand__TD3_LQR.gif) | [mp4](exp_b/videos_strips/ice_and_sand__TD3_LQR.mp4) | подряд лёд и песок — разное сцепление на пути |

> Для `sand_strip` (без препятствия) есть график [plot](exp_b/plots_strips/sand_strip_trajectories.png); отдельный ролик не записывался.

---

## Соответствие статье

| Материал | Где в статье |
|---|---|
| `exp_a/trajectories/`, `exp_a/videos_*` | рис. 2 (траектории), Таблица 1 |
| `exp_b/plots_*`, `exp_b/videos_*` | Таблица 2 |

Порождающий код: `experiments/exp_a_scenarios.py` (Таблица 1),
`experiments/exp_b_global_surfaces.py` (Таблица 2),
`experiments/exp_a_video_mp4.py` (запись роликов).

> Замечание: набор медиа полнее статьи — включены сценарии и поверхности, не вошедшие
> в поданный текст (slalom, diagonal_goal, полосы ice/sand и др.), как задел для диссертации.
