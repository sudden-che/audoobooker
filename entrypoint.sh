#!/bin/sh
set -e

echo "=== Audiobooker starting in MODE=${MODE} ==="

case "$MODE" in
  web)
    echo "[web] Starting FastAPI on ${WEB_HOST}:${WEB_PORT}"
    exec python web_audiobooker.py
    ;;

  bot)
    echo "[bot] Starting Telegram bot"
    exec python tg_audiobooker.py
    ;;

  both)
    echo "[both] Starting FastAPI + Telegram bot"
    python web_audiobooker.py &
    WEB_PID=$!
    python tg_audiobooker.py &
    BOT_PID=$!
    # Завершаем оба процесса при выходе
    trap "kill $WEB_PID $BOT_PID 2>/dev/null" EXIT INT TERM
    wait $WEB_PID $BOT_PID
    ;;

  *)
    echo "ERROR: Unknown MODE='${MODE}'. Use: web | bot | both"
    exit 1
    ;;
esac
