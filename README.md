# Генерация изображений диффузионной моделью на CIFAR-10

Курсовая работа по курсу «Нейронные сети», ВИШ РУТ (МИИТ), осенний семестр 2026.

Реализуем диффузионную модель DDPM ([Ho et al., 2020](https://arxiv.org/abs/2006.11239)):
прямой процесс постепенного зашумления картинок, U-Net, который предсказывает добавленный шум,
и обратный процесс, который генерирует новые картинки из чистого шума. Затем с помощью
классификатора исследуем, на каком шаге генерации картинка «выбирает» свой класс.

## Структура репозитория

```text
├── configs/
│   └── default.yaml          # все гиперпараметры и пути
├── src/
│   ├── utils.py              # seed, конфиги, пути, логирование
│   ├── data.py               # CIFAR-10: нормализация, аугментация, разбиение train/val/test
│   ├── diffusion.py          # расписания шума, прямой и обратный процессы, функция потерь
│   ├── unet.py               # U-Net, предсказывающий шум
│   ├── training.py           # EMA-копия весов, порядок батчей, чекпойнты
│   └── plotting.py           # единый стиль графиков, сетки картинок
├── scripts/
│   ├── data_overview.py      # статистика и примеры датасета
│   ├── exp1_forward_process.py  # эксперимент 1: прямой процесс
│   ├── profile_unet.py       # память и скорость обучения на видеокарте
│   └── train_diffusion.py    # обучение диффузионной модели
├── tests/                    # тесты (pytest)
├── results/
│   └── figures/              # графики для отчёта
├── checkpoints/              # веса моделей (не хранятся в git)
└── requirements.txt
```

Классификатор и анализ траекторий генерации добавляются по мере работы.

## Установка

Нужен Python 3.10 или новее.

```bash
git clone https://github.com/Yozario/NN_diffusion_coursework.git
cd NN_diffusion_coursework
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate
```

PyTorch с поддержкой видеокарты NVIDIA ставится отдельно, до остальных зависимостей.
На [pytorch.org](https://pytorch.org/get-started/locally/) выбрать Stable, свою ОС, Pip, Python
и версию CUDA, затем выполнить показанную там команду — она выглядит так:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cuXXX
```

Проверка, что видеокарта видна: `python -c "import torch; print(torch.cuda.is_available())"` должно напечатать `True`.
Без видеокарты этот шаг можно пропустить — следующая команда поставит обычную версию. Затем остальные зависимости:

```bash
pip install -r requirements.txt
```

Проверка установки — все тесты должны пройти:

```bash
python -m pytest
```

## Данные

CIFAR-10 скачивается автоматически при первом запуске любого скрипта в папку `data/`
(около 170 МБ, в git не хранится).

| Часть | Картинок | Откуда | Преобразования |
|---|---|---|---|
| train | 45 000 | официальная обучающая выборка | случайное горизонтальное отражение, [0, 1] → [−1, 1] |
| val | 5 000 | официальная обучающая выборка, по 500 на класс | [0, 1] → [−1, 1] |
| test | 10 000 | официальная тестовая выборка | [0, 1] → [−1, 1] |

Разбиение train/val стратифицированное и зависит только от `seed` в конфиге.

## Запуск

Все команды выполняются из корня репозитория.

| Команда | Что делает | Время | Результаты |
|---|---|---|---|
| `python -m scripts.data_overview` | статистика по классам и яркостям, примеры картинок | ~1 мин | `results/data_stats.csv`, `results/data_pixel_stats.json`, `results/figures/data_samples.png` |
| `python -m scripts.exp1_forward_process` | эксперимент 1: прямой процесс | 2–5 мин на CPU | `results/exp1_*.csv`, `results/exp1_summary.json`, `results/figures/exp1_*.png` |
| `python -m scripts.profile_unet` | замер памяти и скорости обучения U-Net | ~1 мин на GPU | `results/unet_profile.json` |
| `python -m scripts.train_diffusion` | обучение диффузионной модели | ~8 ч на RTX 3050 Laptop | `checkpoints/`, `results/train_*.csv`, `results/figures/train_*` |

Другой конфиг передаётся флагом: `python -m scripts.exp1_forward_process --config configs/my.yaml`.
Логи запусков пишутся в `results/logs/`.

## Прямой процесс

Один шаг зашумления и эквивалентная ему замкнутая формула:

$$x_t = \sqrt{1-\beta_t}\,x_{t-1} + \sqrt{\beta_t}\,\varepsilon, \qquad
x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\varepsilon, \qquad
\bar\alpha_t = \prod_{s=1}^{t}(1-\beta_s), \quad \varepsilon \sim \mathcal N(0, I).$$

Реализованы два расписания $\beta_t$, число шагов $T = 1000$:

- **линейное** — $\beta_t$ равномерно от $10^{-4}$ до $0.02$ (DDPM);
- **косинусное** — $\bar\alpha_t = \cos^2\!\big(\tfrac{t/T + s}{1 + s}\cdot\tfrac{\pi}{2}\big)$ с нормировкой, $s = 0.008$
  ([Nichol & Dhariwal, 2021](https://arxiv.org/abs/2102.09672)).

**Соглашение об индексах.** Массивы коэффициентов в `GaussianDiffusion` имеют длину $T + 1$,
индекс равен номеру шага: $\beta_0 = 0$, $\bar\alpha_0 = 1$, так что `q_sample(x0, 0)` возвращает $x_0$.
При обучении шаг выбирается из $\{1, \dots, T\}$.

## Модель

U-Net по схеме DDPM, уменьшенный под видеокарту с 4 ГБ памяти: 6.47 млн параметров.
На вход — зашумлённая картинка $x_t$ и номер шага $t$, на выходе — предсказанный шум $\varepsilon_\theta(x_t, t)$.

- Три уровня разрешения: 32×32, 16×16, 8×8 с 64, 128 и 128 каналами; по два остаточных блока на уровень.
- Номер шага кодируется синусами и косинусами разных частот и подаётся в каждый остаточный блок.
- Самовнимание (4 головы) на разрешении 16×16 и в середине сети.
- GroupNorm, активация SiLU, dropout 0.1. Последние слои блоков инициализированы нулями,
  поэтому до обучения каждый блок — тождественное отображение, а сеть выдаёт ноль.

Все размеры задаются в разделе `model` конфига.

## Обучение

Функция потерь — упрощённая функция потерь DDPM:

$$L = \mathbb E_{x_0,\,t,\,\varepsilon}\,\big\lVert \varepsilon - \varepsilon_\theta(x_t, t) \big\rVert^2, \qquad
t \sim U\{1, \dots, T\}, \quad \varepsilon \sim \mathcal N(0, I).$$

| Параметр | Значение |
|---|---|
| Шагов обучения | 100 000 (~142 эпохи) |
| Батч | 64 |
| Оптимизатор | Adam, lr = 2·10⁻⁴, линейный разогрев 1000 шагов |
| Ограничение нормы градиента | 1.0 |
| EMA-копия весов | коэффициент 0.9999 |
| Смешанная точность | bfloat16 |

Каждые 2000 шагов считается потеря на 2000 картинках из val с фиксированными $t$ и шумом,
поэтому значения на разных этапах обучения сравнимы. Каждые 10 000 шагов EMA-веса генерируют
64 картинки из одного и того же шума — так видно, как растёт качество.

```bash
python -m scripts.train_diffusion                      # обучение с нуля
python -m scripts.train_diffusion --total-steps 2000   # короткий отладочный прогон
python -m scripts.train_diffusion --resume             # продолжить с checkpoints/last.pt
python -m scripts.train_diffusion --plot-only          # перерисовать график по журналам
```

Ctrl+C останавливает обучение и сохраняет чекпойнт; продолжить можно с флагом `--resume`.

**Результаты** (RTX 3050 Laptop, 4 ГБ): 100 000 шагов за 8.2 часа, пик видеопамяти 1.91 ГБ.
Потеря на валидации с EMA-весами снизилась с 0.0341 на шаге 2000 до 0.0294 на шаге 100 000
(с текущими весами — до 0.0298). Кривые — `results/figures/train_loss.png`,
сгенерированные картинки — `results/figures/train_samples/`.

## Обратный процесс и генерация

Генерация идёт от чистого шума $x_T$ к картинке $x_0$. На каждом шаге сеть предсказывает шум,
по нему восстанавливается оценка исходной картинки $\hat x_0$ (обрезается до $[-1, 1]$), и делается шаг
по апостериорному распределению:

$$\hat x_0 = \frac{x_t - \sqrt{1-\bar\alpha_t}\,\varepsilon_\theta(x_t, t)}{\sqrt{\bar\alpha_t}}, \qquad
x_{t-1} = \frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1-\bar\alpha_t}\,\hat x_0
+ \frac{\sqrt{\alpha_t}\,(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\,x_t + \sigma_t z, \quad \sigma_t^2 = \beta_t.$$

На последнем шаге шум не добавляется. Пример генерации с записью траектории:

```python
import torch
from src.diffusion import GaussianDiffusion
from src.training import load_ema_model
from src.unet import UNet
from src.utils import get_device, load_config, make_generator

cfg = load_config()
device = get_device()
model = load_ema_model(UNet.from_config(cfg), "checkpoints/ema_final.pt", device)
diffusion = GaussianDiffusion.from_config(cfg).to(device)

out = diffusion.p_sample_loop(
    model,
    shape=(16, 3, 32, 32),
    generator=make_generator(cfg["seed"], device),
    record_steps=range(0, 1001, 10),          # сохранить x_t и x̂_0 каждые 10 шагов
    autocast_dtype=torch.bfloat16 if device.type == "cuda" else None,
)
out.images       # итоговые картинки [16, 3, 32, 32] в [−1, 1]
out.x_t[500]     # x_t на шаге 500 (на CPU)
out.x0_pred[500] # оценка x̂_0, сделанная сетью на шаге 500 (на CPU)
```

`diffusion.reconstruct(model, x0, t)` зашумляет картинки до шага $t$ и восстанавливает их обратным процессом.

## Эксперимент 1

Проверяет прямой процесс тремя способами:

1. **Расписания шума** — графики $\bar\alpha_t$ и $\mathrm{SNR}(t) = \bar\alpha_t / (1 - \bar\alpha_t)$ для двух расписаний.
2. **Деградация картинок** — восемь тестовых картинок, зашумлённых пошаговым процессом, на шагах от 0 до 1000.
3. **Эквивалентность пошагового процесса и замкнутой формулы.** При фиксированной $x_0$
   остаток $z = (x_t - \sqrt{\bar\alpha_t}\,x_0)/\sqrt{1-\bar\alpha_t}$ должен быть распределён как $\mathcal N(0, 1)$.
   Для каждого расписания, пяти шагов $t$ и 20 повторов сравниваются выборки $z$ двух способов
   двухвыборочным тестом Колмогорова — Смирнова; дополнительно дисперсия остатка вдоль всей
   траектории сравнивается с теоретической $1 - \bar\alpha_t$.

## Воспроизводимость

- Все параметры — в `configs/default.yaml`, seed по умолчанию 42.
- `set_seed` фиксирует `random`, `numpy`, `torch` (CPU и CUDA) и выключает недетерминированные алгоритмы cuDNN.
- Шум в экспериментах генерируется отдельными генераторами `torch.Generator` с явным seed,
  поэтому результат не зависит от порядка остальных вызовов.
- Разбиение train/val и порядок батчей загрузчика тоже зависят только от seed.
- Чекпойнт обучения хранит веса, состояние оптимизатора и всех генераторов случайных чисел:
  продолжение после остановки идёт так же, как непрерывный запуск (на CPU результат совпадает побайтно).

Повторный запуск на той же машине даёт те же числа и рисунки. На другом железе (CPU против GPU,
другая версия PyTorch) возможны расхождения в последних знаках из-за округления.

## Веса моделей

Веса не хранятся в git из-за размера. EMA-веса обученной модели (26 МБ) —
[ema_final.pt](https://github.com/Yozario/NN_diffusion_coursework/releases/download/model-v1/ema_final.pt)
в релизе [model-v1](https://github.com/Yozario/NN_diffusion_coursework/releases/tag/model-v1).
Файл нужно положить в папку `checkpoints/`:

```bash
mkdir checkpoints
curl -L -o checkpoints/ema_final.pt https://github.com/Yozario/NN_diffusion_coursework/releases/download/model-v1/ema_final.pt
```

В Windows PowerShell вместо `curl` нужно писать `curl.exe`.

## Литература

1. Ho J., Jain A., Abbeel P. Denoising Diffusion Probabilistic Models. NeurIPS 2020. [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)
2. Nichol A., Dhariwal P. Improved Denoising Diffusion Probabilistic Models. ICML 2021. [arXiv:2102.09672](https://arxiv.org/abs/2102.09672)
3. Krizhevsky A. Learning Multiple Layers of Features from Tiny Images. 2009 (CIFAR-10).
