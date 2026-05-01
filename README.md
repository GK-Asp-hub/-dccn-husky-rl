# dccn-husky-rl

Воспроизводимая реализация гибридной архитектуры управления для мобильного робота **Husky** в среде **PyBullet**: реактивная политика **TD3** + слой уточнения на основе **LQR** (схема Residual Policy) + классический **goal-conditioned waypoint-планировщик**.

Сопровождает статью «Метод синтеза поведения когнитивного агента на основе обработки мультимодальных сигналов» (DCCN-2026).

## Содержание

- [Что внутри](#что-внутри)
- [Установка](#установка)
- [Быстрый запуск (eval)](#быстрый-запуск-eval)
- [Обучение с нуля](#обучение-с-нуля)
- [Структура репозитория](#структура-репозитория)
- [Воспроизведение результатов статьи](#воспроизведение-результатов-статьи)
- [Демо-видео](#демо-видео)
- [Тесты](#тесты)
- [Зависимости](#зависимости)

---

## Что внутри

Реализованы три слоя управления и среда моделирования:

| Слой | Файл | Описание |
|------|------|----------|
| Среда (пустая арена) | `envs/husky_goal_env.py` | Husky едет в случайную точку на пустой плоскости 10×10. Observation 9D: pose + velocity + goal_vector. |
| Среда (с препятствиями) | `envs/husky_obstacle_env.py` | То же + 3–5 случайных цилиндров. Observation 25D (добавлены 16 лучей лидара). |
| Реактивная policy | внешняя (TD3 из `stable-baselines3`) | Continuous control, 2D action: linear + angular velocity. |
| LQR Residual | `controllers/lqr_diff_drive.py` | Линеаризованный LQR-контроллер для удержания траектории; выход смешивается с TD3 как `a = π_TD3(s) + α · π_LQR(s)`. |
| Waypoint-планировщик | `planners/waypoint_planner.py` | Делит линию старт-цель на N подцелей; переключение по радиусу. |
| Wrapper-среда | `envs/husky_goal_planned_env.py` | Generic-обёртка, подменяющая `goal` на текущий waypoint. Работает над любой inner-средой. |

Видеодемонстрация работы — три эпизода в папке [`videos/`](videos/), см. также раздел [Демо-видео](#демо-видео) ниже.

---

## Установка

Требуется Python 3.11.

```bash
# 1. Клонировать репозиторий
git clone https://github.com/GK-Asp-hub/-dccn-husky-rl.git dccn-husky-rl
cd dccn-husky-rl

# 2. Создать виртуальное окружение
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate

# 3. Установить зависимости
pip install --upgrade pip
pip install -r requirements.txt
```

Если возникнут конфликты версий — используйте `requirements-lock.txt` (точные версии, на которых получены все результаты статьи):

```bash
pip install -r requirements-lock.txt
```

### Скачать предобученные модели

Веса моделей TD3 размещены в [Releases v1.0](../../releases/tag/v1.0). Скачайте архив `models.zip` и распакуйте в корень репозитория — должна получиться папка `models/`:

```
models/
├── husky_td3_v1_cont_best/best_model.zip       # Stage 1 (пустая арена)
├── husky_obstacle_td3_v1_best/best_model.zip   # Stage 2a v1 (слабая, до фикса лидара)
└── husky_obstacle_td3_v3_steppen036_best/best_model.zip   # Stage 2a v3 (финальная)
```

---

## Быстрый запуск (eval)

Все три команды используют CPU, занимают 1–3 минуты и выводят таблицу метрик в консоль + сохраняют JSON.

### Stage 1: пустая арена (5 режимов × 10 эпизодов)

```bash
python eval_td3_lqr_planner.py
```

Результат: `ablation_results.json`. Ожидаемые числа:

| Режим | Success | Steps | Smoothness |
|---|---|---|---|
| TD3 baseline | 10/10 | 76.6 | 0.196 |
| TD3 + LQR (α=0.2) | 10/10 | 77.4 | 0.176 |
| TD3 + LQR + Planner N=3 | 9/10 | 120.9 | 0.288 |

### Stage 2a: с препятствиями, 30 сидов, финальная модель v3

```bash
python eval_td3_lqr_planner_obstacle.py ^
    --model models/husky_obstacle_td3_v3_steppen036_best/best_model.zip ^
    --episodes 30
```

Результат: `ablation_results_obstacle.json`. Ожидаемые числа:

| Режим | Success | Collisions | Steps | Smoothness |
|---|---|---|---|---|
| TD3 baseline (v3) | 27/30 | 3/30 | 65.4 ± 39 | 0.322 |
| TD3 + LQR (α=0.2) | 27/30 | 3/30 | 66.9 ± 44 | 0.302 |
| TD3 + LQR + Planner N=3 | 25/30 | 3/30 | 111.4 ± 127 | 0.285 |
| TD3 + LQR + Planner N=5 | 25/30 | 3/30 | 119.8 ± 128 | 0.315 |

### Stage 2a v1 (контрольная модель до фикса лидара)

```bash
python eval_td3_lqr_planner_obstacle.py ^
    --model models/husky_obstacle_td3_v1_best/best_model.zip ^
    --episodes 30
```

Демонстрирует противоположный эффект planner'а на слабой policy (см. раздел 5 статьи).

### Визуализация в GUI

**Stage 1 (пустая арена):**

```bash
python visual_planned_husky.py ^
    --model models/husky_td3_v1_cont_best/best_model.zip ^
    --episodes 3
```

**Stage 2a (с препятствиями) — финальная модель v3:**

```bash
python visual_obstacle_husky.py ^
    --model models/husky_obstacle_td3_v3_steppen036_best/best_model.zip ^
    --episodes 3 --seed-base 300
```

**Stage 2a с планировщиком (главный результат статьи):**

```bash
python visual_obstacle_husky.py ^
    --model models/husky_obstacle_td3_v3_steppen036_best/best_model.zip ^
    --episodes 3 --seed-base 300 --planner --n-waypoints 3
```

В окне PyBullet будут видны:
- красный цилиндр — финальная цель;
- зелёный цилиндр — активная подцель планировщика (только в режиме `--planner`);
- синие — пройденные подцели, серые — ещё не активированные;
- жёлтые лучи из робота — лидар (только Stage 2a);
- цилиндры случайной высоты — препятствия (только Stage 2a).

#### Запись видео

Оба `visual_*.py` поддерживают флаг `--record PATH` для записи окна PyBullet в mp4. Камера выставляется автоматически (вид сверху-сбоку, центр в начале координат). Запись начинается в первом эпизоде и останавливается по его завершению.

Требование: `ffmpeg` в `PATH` (PyBullet вызывает его popen'ом для кодирования h264).

```bash
python visual_planned_husky.py ^
    --model models/husky_td3_v1_cont_best/best_model.zip ^
    --episodes 1 --seed-base 200 --slowdown 2 ^
    --record videos/01_stage1_baseline.mp4
```

Готовые ролики, использованные при подготовке статьи, лежат в `videos/` — см. раздел [Демо-видео](#демо-видео).

---

## Обучение с нуля

### Stage 1 (пустая арена), ~15 минут CPU

```bash
python train_td3_husky.py
```

Сохраняет лучшую модель в `models/husky_td3_v1_best/`.

### Stage 2a (с препятствиями), ~80 минут CPU

```bash
python train_td3_obstacle.py
```

Сохраняет лучшую модель в `models/husky_obstacle_td3_v1_best/`.

Прогресс пишется в `runs/` (tensorboard). Просмотр:

```bash
tensorboard --logdir runs
```

---

## Структура репозитория

```
dccn-husky-rl/
├── envs/                          # gymnasium-среды
│   ├── husky_goal_env.py          # пустая арена
│   ├── husky_obstacle_env.py      # с препятствиями + лидар
│   └── husky_goal_planned_env.py  # generic-обёртка с планировщиком
├── controllers/
│   └── lqr_diff_drive.py          # LQR для differential drive
├── planners/
│   └── waypoint_planner.py        # линейный waypoint-планировщик
├── train_td3_husky.py             # обучение Stage 1
├── train_td3_obstacle.py          # обучение Stage 2a
├── eval_td3_lqr.py                # eval только LQR-residual
├── eval_td3_lqr_planner.py        # full ablation Stage 1
├── eval_td3_lqr_planner_obstacle.py  # full ablation Stage 2a
├── visual_planned_husky.py        # GUI-визуализация Stage 1
├── visual_obstacle_husky.py       # GUI-визуализация Stage 2a (с лидаром и препятствиями)
├── plot_trajectories.py           # графики траекторий Stage 1
├── plot_trajectories_obstacle.py  # графики траекторий Stage 2a
├── make_report_figures.py         # сводные графики (training curves, planner effect)
├── test_*.py                      # юнит- и smoke-тесты
├── ablation_results*.json         # сохранённые таблицы метрик из статьи
├── figures/                       # сгенерированные графики (PNG, SVG)
├── videos/                        # 3 демо-эпизода (mp4, см. раздел «Демо-видео»)
├── requirements.txt               # минимальные зависимости
└── requirements-lock.txt          # точные версии (для воспроизводимости)
```

---

## Воспроизведение результатов статьи

Команды ниже воспроизводят все таблицы и графики из разделов 4–5 статьи. Полный прогон — около 5 минут на CPU.

```bash
# Таблица 5.1 (Stage 1)
python eval_td3_lqr_planner.py

# Таблица 5.2 (Stage 2a v3, 30 сидов)
python eval_td3_lqr_planner_obstacle.py --model models/husky_obstacle_td3_v3_steppen036_best/best_model.zip --episodes 30

# Контрольная таблица v1 vs v3
python eval_td3_lqr_planner_obstacle.py --model models/husky_obstacle_td3_v1_best/best_model.zip --episodes 30

# Графики траекторий (рис. 5.1 и 5.3 статьи)
python plot_trajectories.py
python plot_trajectories_obstacle.py --model models/husky_obstacle_td3_v3_steppen036_best/best_model.zip

# Сводные графики (рис. 5.4 и 5.5 статьи)
python make_report_figures.py
```

---

## Демо-видео

Папка [`videos/`](videos/) содержит три эпизода, иллюстрирующие главное эмпирическое наблюдение статьи: **полезность waypoint-планировщика обратно пропорциональна качеству underlying policy**. Все ролики записаны через флаг `--record` тех же `visual_*.py` скриптов (см. [Запись видео](#запись-видео)).

### Как посмотреть

GitHub в файловом дереве не показывает встроенный плеер для mp4 — клик по файлу открывает blame-view без воспроизведения. Есть два рабочих пути:

1. **Release v1.1-media** ([прямая ссылка](https://github.com/GK-Asp-hub/-dccn-husky-rl/releases/tag/v1.1-media)) — каждый файл скачивается одним кликом по ссылке в разделе *Assets*. Это рекомендуемый способ.
2. **Файл в репо** → клик по имени файла в таблице ниже → на открывшейся странице нажать кнопку **«View raw»** (ссылка вверху справа над содержимым). Браузер скачает файл; откройте в локальном плеере (VLC, MPC, любой системный).

Видео — стандартный H.264 / yuv420p в контейнере mp4, играют в любом современном плеере.

### Эпизоды

| Файл в репо | Скачать (Release) | Длит. | Что показано |
|------|------|-------|--------------|
| [`01_stage1_baseline.mp4`](videos/01_stage1_baseline.mp4) | [⬇](https://github.com/GK-Asp-hub/-dccn-husky-rl/releases/download/v1.1-media/01_stage1_baseline.mp4) | 3.5 с | Stage 1, пустая арена, seed=200. TD3 + LQR доходит до цели за 62 шага. Базовая ситуация. |
| [`02_stage2a_v3_baseline.mp4`](videos/02_stage2a_v3_baseline.mp4) | [⬇](https://github.com/GK-Asp-hub/-dccn-husky-rl/releases/download/v1.1-media/02_stage2a_v3_baseline.mp4) | 44 с | Stage 2a v3, арена с препятствиями, seed=304. TD3 v3 (без планировщика) доходит за 76 шагов, без коллизий. |
| [`03_stage2a_v3_planner.mp4`](videos/03_stage2a_v3_planner.mp4) | [⬇](https://github.com/GK-Asp-hub/-dccn-husky-rl/releases/download/v1.1-media/03_stage2a_v3_planner.mp4) | 4:51 | Stage 2a v3 + waypoint-планировщик N=3, **тот же seed 304, та же policy**. Эпизод заканчивается timeout'ом на 500 шагов, reward −856.74. |

Эпизоды 02 и 03 различаются только включённым планировщиком — всё остальное (модель, среда, seed) идентично. Совместный просмотр визуально воспроизводит negative effect планировщика на сильной policy (−6.7 п.п. в success rate на полном ablation 30 сидов, см. таблицу выше).

### Команды для воспроизведения с нуля

```bash
python visual_planned_husky.py --model models/husky_td3_v1_cont_best/best_model.zip --episodes 1 --seed-base 200 --slowdown 2 --record videos/01_stage1_baseline.mp4
python visual_obstacle_husky.py --model models/husky_obstacle_td3_v3_steppen036_best/best_model.zip --episodes 1 --seed-base 304 --slowdown 2 --record videos/02_stage2a_v3_baseline.mp4
python visual_obstacle_husky.py --model models/husky_obstacle_td3_v3_steppen036_best/best_model.zip --episodes 1 --seed-base 304 --slowdown 2 --planner --n-waypoints 3 --record videos/03_stage2a_v3_planner.mp4
```

---

## Тесты

Smoke- и юнит-проверки на ключевые компоненты. Каждый тест — самостоятельный скрипт, запускается напрямую и печатает диагностику в консоль:

```bash
# Юнит на LQR (быстро, без PyBullet)
python test_lqr_unit.py
python test_waypoint_planner.py

# Smoke-тесты сред (требуют PyBullet, открывают headless-симуляцию)
python test_husky_env.py
python test_husky_obstacle_smoke.py
python test_husky_planned_env_smoke.py
python test_obstacle_planned_smoke.py
```

Покрытие: проверка матрицы K в LQR, формирование waypoints, корректность observation/reward в трёх gym-средах, smoke-проход эпизода с планировщиком.

---

## Зависимости

Основные: **stable-baselines3 2.8** (TD3), **gymnasium 1.2**, **PyBullet 3.2.6**, **NumPy 2.x**, **SciPy 1.17**, **Matplotlib 3.10**. Все версии из `requirements-lock.txt` проверены в среде Python 3.11.9 на Windows. Linux/macOS должны работать без правок (абсолютные пути в коде отсутствуют).

PyTorch ставится автоматически как зависимость stable-baselines3. Достаточно CPU-версии — все эксперименты проводились на CPU (Intel x86-64).

---

## Лицензия

MIT.

---

## Ссылки

- Статья: «Метод синтеза поведения когнитивного агента на основе обработки мультимодальных сигналов», DCCN-2026 (РУДН).
- Базовая работа: Вейценфельд, Киселёв (2024).
- Реализация TD3: Fujimoto et al., ICML-2018.
- Residual Policy Learning: Silver et al. (2018), Johannink et al. (2019).
