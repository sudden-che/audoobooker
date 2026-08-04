FROM python:3.12-slim

# ── системные зависимости ──────────────────────────────────────────
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg procps && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python-зависимости ───────────────────────────────────────────
COPY requirements.txt requirements-web.txt requirements-tg.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
        -r requirements-web.txt \
        -r requirements-tg.txt

# ── исходный код ─────────────────────────────────────────────────
COPY audiobooker.py web_audiobooker.py tg_audiobooker.py download_delivery.py tts_dependency_manager.py entrypoint.sh ./
RUN chmod +x entrypoint.sh

# ── переменные окружения (значения по умолчанию) ─────────────────
# Режим запуска: web | bot | both
ENV MODE=web

# Параметры Edge
ENV VOICE=ru-RU-SvetlanaNeural
ENV SPEED=+18%
ENV EDGE_TTS_MAX_RETRIES=3
ENV EDGE_TTS_RETRY_BASE_DELAY=1.5
ENV EDGE_TTS_RETRY_MAX_DELAY=12.0

# Общие параметры
ENV CHUNK_SIZE=10000
ENV MAX_CONCURRENT_TASKS=""
ENV FFMPEG_PATH=ffmpeg
ENV OUTPUT_BASENAME_MAX_LENGTH=48
ENV MERGE_CHUNKS=true
ENV DELIVERY_MODE=telegram
ENV DOWNLOAD_DIR=/app/data/downloads
ENV DOWNLOAD_TTL_SECONDS=900
ENV DOWNLOAD_TTL_MINUTES=""
ENV DOWNLOAD_SERVER_HOST=0.0.0.0
ENV DOWNLOAD_SERVER_PORT=43561
ENV DOWNLOAD_SERVER_IDLE_GRACE_SECONDS=5
ENV PUBLIC_DOWNLOAD_BASE_URL=""
ENV TELEGRAM_SAFE_UPLOAD_BYTES=51380224
ENV MAX_TEXT_FROM_MESSAGE=50000
ENV CHECK_TTS_DEPS_ON_START=true
ENV AUTO_UPDATE_TTS_DEPS=false

# [[PAUSE: 5000]] задаёт тишину в миллисекундах
ENV MAX_PAUSE_MS=600000
ENV MAX_TOTAL_PAUSE_MS=3600000

# Web-сервер
ENV WEB_HOST=0.0.0.0
ENV WEB_PORT=8000

# Telegram-бот (обязательно задать при запуске)
ENV BOT_TOKEN=""

EXPOSE 8000 43561

ENTRYPOINT ["./entrypoint.sh"]
