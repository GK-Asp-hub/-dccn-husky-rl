# Демо-видео для статьи DCCN

Три эпизода, иллюстрирующие главный эмпирический результат статьи: **полезность waypoint-планировщика обратно пропорциональна качеству underlying policy**.

## Как посмотреть

GitHub **не воспроизводит mp4** в файловом дереве — клик по файлу открывает blame-view без плеера. Есть два рабочих пути:

### Вариант A — Release (рекомендуется)

Все три ролика приложены как assets к [Release v1.1-media](https://github.com/GK-Asp-hub/-dccn-husky-rl/releases/tag/v1.1-media). На странице релиза в разделе **Assets** клик по имени файла скачивает его одним кликом без захода в blame.

Прямые ссылки на скачивание:

- [01_stage1_baseline.mp4](https://github.com/GK-Asp-hub/-dccn-husky-rl/releases/download/v1.1-media/01_stage1_baseline.mp4) (0.6 МБ)
- [02_stage2a_v3_baseline.mp4](https://github.com/GK-Asp-hub/-dccn-husky-rl/releases/download/v1.1-media/02_stage2a_v3_baseline.mp4) (3.3 МБ)
- [03_stage2a_v3_planner.mp4](https://github.com/GK-Asp-hub/-dccn-husky-rl/releases/download/v1.1-media/03_stage2a_v3_planner.mp4) (19.7 МБ)

### Вариант B — файлы в этой папке

Клик по имени файла открывает страницу с blame. Сверху справа над содержимым — кнопка **«View raw»** (или просто `?raw=true` в URL). Браузер скачает файл; откройте его в локальном плеере.

## Что в роликах

| Файл | Длит. | Эпизод | Итог |
|------|-------|--------|------|
| `01_stage1_baseline.mp4` | 3.5 с | Stage 1, пустая арена, seed=200. TD3 + LQR без планировщика. | ✅ goal_reached, 62 шага |
| `02_stage2a_v3_baseline.mp4` | 44 с | Stage 2a v3, арена с препятствиями, seed=304. TD3 v3 без планировщика. | ✅ goal_reached, 76 шагов, без коллизий |
| `03_stage2a_v3_planner.mp4` | 4:51 | Stage 2a v3 + waypoint-планировщик N=3, **тот же seed 304, та же policy**. | ❌ timeout на 500 шагов, reward −856.74 |

Эпизоды 02 и 03 различаются **только включённым планировщиком** — всё остальное (модель, среда, seed) идентично. Совместный просмотр визуально воспроизводит negative effect планировщика на сильной policy (−6.7 п.п. в success rate на полном ablation 30 сидов).

## Технические детали

- Кодек: H.264 High Profile, level 3.2
- Pixel format: yuv420p
- Контейнер: mp4 (isom/avc1/mp41)
- Разрешение: 1024×768
- Частота кадров: 60 fps

Стандартный набор, играет в любом современном плеере (VLC, MPC, системные плееры Windows/macOS, браузеры).

## Воспроизведение

Все три ролика записаны встроенным mp4-рекордером PyBullet через флаг `--record` в `visual_*.py`. Команды и описание — в корневом [README.md → Демо-видео](../README.md#демо-видео).
