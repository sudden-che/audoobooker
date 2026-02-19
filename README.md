# Audiobooker

Audiobooker - это инструмент на Python для создания аудиокниг из текстовых файлов (`.txt`) и электронных книг (`.fb2`).

## Основные возможности
- **Два движка на выбор:**
  - **Edge TTS (Online):** использует бесплатный сервис Microsoft Edge. Высокое качество, не требует мощного железа, но нужен интернет.
  - **Silero TTS (Local):** полностью локальный синтез через PyTorch. Работает без интернета, поддерживает множество дикторов.
- **Поддержка форматов:** Прямое чтение `.txt` и автоматическое извлечение текста из `.fb2`.
- **Разбивка на чанки:** Текст автоматически делится на части для стабильного синтеза.
- **Склейка:** Автоматическое объединение частей в один файл с помощью `ffmpeg`.
- **Интерфейсы:** CLI (командная строка), Telegram-бот и Web-интерфейс.

---

## Установка

### 1. Системные зависимости:
Для склейки аудио и конвертации в MP3 требуется `ffmpeg`.
Для Silero TTS (локально) требуются `libsndfile1` и `scipy` (в Linux: `sudo apt install libsndfile1`).

### 2. Python зависимости:
```bash
pip install -r requirements.txt
```
*Зависимости для бота и веб-интерфейса:* `pip install -r requirements-tg.txt` или `pip install -r requirements-web.txt`.

---

## Использование (CLI)

Запуск с движком **Edge** (по умолчанию):
```bash
python audiobooker.py book.txt --voice ru-RU-SvetlanaNeural --rate +18%
```

Запуск с движком **Silero** (используется качественная модель **v5_ru** по умолчанию):
```bash
python audiobooker.py book.fb2 --engine silero --speaker baya --sample-rate 48000 --final-mp3
```

**Все аргументы:**
- `--engine {edge,silero}`: выбор движка.
- `--silero-model-id`: версия модели Silero (по умолчанию `v5_ru`).
- `--speaker`: имя диктора (для Silero: `aidar`, `baya`, `kseniya`, `xenia`, `eugene`).
- `--chunk-size`: количество символов в одном фрагменте (дефолт 10000).
- `--max-concurrent-tasks`: количество параллельных задач синтеза.
- `--final-mp3`: (для Silero) автоматически конвертировать финальный WAV в MP3.
- `--output-dir`: папка для сохранения.
- `--skip-chunks`: не пересобирать уже существующие фрагменты.
- `--skip-merge`: не склеивать фрагменты в один файл.

---

## Telegram-бот

Бот поддерживает как Edge, так и Silero. Настройки задаются через переменные окружения.

**Запуск:**
```bash
export BOT_TOKEN="your_token"
export TTS_ENGINE="edge" # или "silero"
python tg_audiobooker.py
```

---

## Веб-интерфейс

Запустите и откройте `http://localhost:8000`:
```bash
python web_audiobooker.py
```
В интерфейсе можно удобно переключать движки, выбирать дикторов и настраивать параметры.

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
Модели Silero автоматически скачиваются при первом запуске и сохраняются в папке `.cache`.

---

## Очистка текста
Скрипт `clean.py` позволяет предварительно очистить текст от типичного "мусора" перед синтезом.

FB2 файлы обрабатываются автоматически внутри `audiobooker.py`, внешние конвертеры больше не требуются.
