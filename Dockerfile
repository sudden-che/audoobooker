FROM python:3.12-slim

# ── системные зависимости (ffmpeg) ──────────────────────────────
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
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

# Параметры синтеза
ENV VOICE=ru-RU-SvetlanaNeural
ENV SPEED=+18%
ENV CHUNK_SIZE=10000
ENV MAX_CONCURRENT_TASKS=40
ENV FFMPEG_PATH=ffmpeg
ENV MERGE_CHUNKS=true
ENV MAX_TEXT_FROM_MESSAGE=50000

# Web-сервер
ENV WEB_HOST=0.0.0.0
ENV WEB_PORT=8000

# Telegram-бот (обязательно задать при запуске)
ENV BOT_TOKEN=""

EXPOSE 8000

ENTRYPOINT ["./entrypoint.sh"]
