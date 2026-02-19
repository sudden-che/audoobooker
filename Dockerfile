FROM python:3.12-slim

# ── системные зависимости ──────────────────────────────────────────
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python-зависимости ───────────────────────────────────────────
COPY requirements.txt requirements-web.txt requirements-tg.txt ./
RUN pip install --no-cache-dir \
        -r requirements.txt \
        -r requirements-web.txt \
        -r requirements-tg.txt

# ── исходный код ─────────────────────────────────────────────────
COPY audiobooker.py web_audiobooker.py tg_audiobooker.py entrypoint.sh ./
RUN chmod +x entrypoint.sh

# ── переменные окружения (значения по умолчанию) ─────────────────
# Режим запуска: web | bot | both
ENV MODE=web

# Выбор движка: edge или silero
ENV TTS_ENGINE=edge

# Параметры Edge
ENV VOICE=ru-RU-SvetlanaNeural
ENV SPEED=+18%

# Параметры Silero
ENV SILERO_LANGUAGE=ru
ENV SILERO_SPEAKER=baya
ENV SILERO_SAMPLE_RATE=48000
ENV SILERO_PUT_ACCENT=true
ENV SILERO_PUT_YO=true
ENV DEVICE=cpu
ENV SILERO_MODEL_ID=v5_ru

# Общие параметры
ENV CHUNK_SIZE=10000
ENV MAX_CONCURRENT_TASKS=""
ENV FFMPEG_PATH=ffmpeg
ENV MERGE_CHUNKS=true
ENV MAX_TEXT_FROM_MESSAGE=50000

# Web-сервер
ENV WEB_HOST=0.0.0.0
ENV WEB_PORT=8000

# Telegram-бот (обязательно задать при запуске)
ENV BOT_TOKEN=""

# Кэш моделей Silero (лучше монтировать как volume)
VOLUME /root/.cache

EXPOSE 8000

ENTRYPOINT ["./entrypoint.sh"]
