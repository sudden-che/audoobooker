# Audiobooker

Audiobooker - это инструмент на Python для создания аудиокниг из текстовых файлов (`.txt`) и электронных книг (`.fb2`).

## Основные возможности
- **Два движка на выбор:**
  - **Edge TTS (Online):** использует бесплатный сервис Microsoft Edge. Высокое качество, не требует мощного железа, но нужен интернет.
  <!-- - **Silero TTS (Local):** полностью локальный синтез через PyTorch. Работает без интернета, поддерживает множество дикторов. -->
  - **Qwen3-TTS CustomVoice (Local):** локальный open-source синтез (Apache-2.0), поддерживает русский язык и набор встроенных голосов с управлением стилем (`instruct`).
- **Отдельный веб-дефолт:** веб-интерфейс стартует с `Edge TTS` по умолчанию, даже если для бота выбран другой движок.
- **Поддержка форматов:** Прямое чтение `.txt` и автоматическое извлечение текста из `.fb2`.
- **Разбивка на чанки:** Текст автоматически делится на части для стабильного синтеза.
- **Склейка:** Автоматическое объединение частей в один файл с помощью `ffmpeg`.
- **Подготовка текста под аудиокнигу:** проект автоматически нормализует реплики, заголовки глав, частые символы (`№`, `%`, `§`, валюты, дроби, градусы) и сохраняет абзацы для естественных пауз.
- **Интерфейсы:** CLI (командная строка), Telegram-бот и Web-интерфейс.

---

## Установка

### 1. Системные зависимости:
Для склейки аудио и конвертации в MP3 требуется `ffmpeg`.
<!-- Для Silero TTS (локально) требуются `libsndfile1` и `scipy` (в Linux: `sudo apt install libsndfile1`). -->

### 2. Python зависимости:
```bash
pip install -r requirements.txt
```
*Зависимости для бота и веб-интерфейса:* `pip install -r requirements-tg.txt` или `pip install -r requirements-web.txt`.

### 3. Актуальность TTS-пакетов
При новой установке ставится свежая совместимая версия `edge-tts` из ветки `7.x`, а также доступен `qwen-tts`.

Проверить/обновить вручную:
```bash
python tts_dependency_manager.py
python tts_dependency_manager.py --apply
```

Для автопроверки на старте:
- `CHECK_TTS_DEPS_ON_START=true` — проверить наличие новой версии.
- `AUTO_UPDATE_TTS_DEPS=true` — автоматически обновить `edge-tts` и `qwen-tts` перед запуском приложения.

### 4. Retry для Edge TTS
Для временных сетевых ошибок `speech.platform.bing.com:443` проект теперь повторяет синтез отдельного чанка автоматически.

Настройки:
- `EDGE_TTS_MAX_RETRIES=3` — число повторов после первой неудачной попытки.
- `EDGE_TTS_RETRY_BASE_DELAY=1.5` — базовая задержка в секундах.
- `EDGE_TTS_RETRY_MAX_DELAY=12.0` — верхний предел задержки.

---

## Использование (CLI)

Запуск с движком **Edge** (по умолчанию):
```bash
python audiobooker.py book.txt --voice ru-RU-SvetlanaNeural --rate +18%
```

<!--
Запуск с движком **Silero** (используется качественная модель **v5_ru** по умолчанию):
```bash
python audiobooker.py book.fb2 --engine silero --speaker baya --sample-rate 48000 --final-mp3
```
-->

Запуск с движком **Qwen3-TTS** (CustomVoice):
```bash
python audiobooker.py book.txt --engine qwen3 --qwen3-speaker Serena --qwen3-language Russian --final-mp3
```

**Все аргументы:**
- `--engine {edge,qwen3}`: выбор движка.
<!-- - `--engine {edge,silero,qwen3}`: выбор движка. -->
<!-- - `--silero-model-id`: версия модели Silero (по умолчанию `v5_ru`). -->
<!-- - `--speaker`: имя диктора (для Silero: `aidar`, `baya`, `kseniya`, `xenia`, `eugene`). -->
- `--qwen3-model-id`: ID/путь модели Qwen3 CustomVoice (по умолчанию `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`).
- `--qwen3-speaker`: голос Qwen3 (`Vivian`, `Serena`, `Uncle_Fu`, `Dylan`, `Eric`, `Ryan`, `Aiden`, `Ono_Anna`, `Sohee`).
- `--qwen3-language`: язык Qwen3 (`Russian`, `Auto`, и др.).
- `--qwen3-instruct`: стиль/эмоция Qwen3 естественным языком.
- `--qwen3-device`: устройство для Qwen3 (`cpu`, `cuda:0`, ...).
- `--chunk-size`: количество символов в одном фрагменте (дефолт 10000).
- `--max-concurrent-tasks`: количество параллельных задач синтеза.
- `--final-mp3`: (для Qwen3) автоматически конвертировать финальный WAV в MP3.
- `--output-dir`: папка для сохранения.
- `--skip-chunks`: не пересобирать уже существующие фрагменты.
- `--skip-merge`: не склеивать фрагменты в один файл.

### Женские голоса Qwen3 для русского текста
Примеры:
```bash
python audiobooker.py book.txt --engine qwen3 --qwen3-speaker Serena --qwen3-language Russian --qwen3-instruct "Спокойный мягкий женский голос для аудиокниги"
python audiobooker.py book.txt --engine qwen3 --qwen3-speaker Vivian --qwen3-language Russian --qwen3-instruct "Яркий молодой женский голос, чуть быстрее среднего темпа"
python audiobooker.py book.txt --engine qwen3 --qwen3-speaker Ono_Anna --qwen3-language Russian --qwen3-instruct "Лёгкий дружелюбный женский голос, ровная дикция"
python audiobooker.py book.txt --engine qwen3 --qwen3-speaker Sohee --qwen3-language Russian --qwen3-instruct "Тёплый выразительный женский голос для художественного текста"
```

---

## Telegram-бот

Бот поддерживает Edge и Qwen3. Настройки задаются через переменные окружения.

**Запуск:**
```bash
export BOT_TOKEN="your_token"
export TTS_ENGINE="edge" # или "qwen3"
python tg_audiobooker.py
```
В боте доступна команда `/settings` для интерактивной настройки всех параметров синтеза.
Для параллельной обработки можно настроить:
- `MAX_CONCURRENT_REQUESTS` — сколько запросов синтеза выполнять одновременно.
- `MAX_CONCURRENT_UPDATES` — сколько Telegram-апдейтов разбирать параллельно.
- `FORWARD_GROUP_DEBOUNCE_SECONDS` — окно группировки пересланных сообщений перед запуском обработки (по умолчанию `5.0` сек).
- Для слабого сервера с 2 CPU разумный старт: `MAX_CONCURRENT_TASKS=8`, `MAX_CONCURRENT_REQUESTS=2`, `MAX_CONCURRENT_UPDATES=4`.

---

## Веб-интерфейс

Запустите и откройте `http://localhost:8000`:
```bash
python web_audiobooker.py
```
В интерфейсе можно удобно переключать движки, выбирать дикторов и настраивать параметры.
<!-- По умолчанию веб-форма открывается с `Edge TTS`. Если нужно изменить только веб-дефолт, используйте `WEB_TTS_ENGINE=edge|silero|qwen3` без влияния на Telegram-бота. -->
По умолчанию веб-форма открывается с `Qwen3 TTS`. Если нужно изменить только веб-дефолт, используйте `WEB_TTS_ENGINE=edge|qwen3` без влияния на Telegram-бота.

---

## Docker

Проще всего запустить проект через Docker Compose:

1. Скопируйте `.env.example` в `.env` и укажите `BOT_TOKEN`.
2. Запустите нужный профиль:
```bash
# Только веб-интерфейс
docker-compose --profile web up -d

# Только бот
docker-compose --profile bot up -d

# Всё вместе
docker-compose --profile both up -d
```
Модели Qwen3 автоматически скачиваются при первом запуске и сохраняются в папке `.cache`.

---

## Очистка текста
Скрипт `clean.py` позволяет предварительно очистить текст от типичного "мусора" перед синтезом.

FB2 файлы обрабатываются автоматически внутри `audiobooker.py`, внешние конвертеры больше не требуются.

## Best Practices для аудиокниг
- Держите абзацы и пустые строки: они дают естественные паузы лучше, чем ручные повторы знаков препинания.
- Для `Edge TTS` разумный старт: `VOICE=ru-RU-SvetlanaNeural`, `SPEED=+10%` ... `+18%`.
<!-- - Для `Silero` держите `chunk_size` не выше `800`, иначе качество и стабильность падают. -->
- Если в исходнике много служебных символов, списков и формул, сначала прогоняйте текст через встроенную предобработку проекта или `clean.py`.

## Имена файлов
- Архивы, чанки и итоговые файлы теперь получают имя от исходного файла, а не от временного `uploaded`.
- Кириллица автоматически транслитерируется в ASCII, чтобы имена стабильно открывались в браузере, Docker и Telegram.
- Длина basename ограничивается переменной `OUTPUT_BASENAME_MAX_LENGTH` (по умолчанию `48`).
