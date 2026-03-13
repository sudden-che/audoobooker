#!/bin/sh
set -e

echo "=== Audiobooker starting in MODE=${MODE} ==="

sync_tts_deps() {
  if [ "${AUDIOBOOKER_SKIP_TTS_SYNC:-0}" = "1" ]; then
    return
  fi

  check_flag=$(printf '%s' "${CHECK_TTS_DEPS_ON_START:-true}" | tr '[:upper:]' '[:lower:]')
  update_flag=$(printf '%s' "${AUTO_UPDATE_TTS_DEPS:-false}" | tr '[:upper:]' '[:lower:]')

  if [ "$check_flag" != "true" ] && [ "$check_flag" != "1" ] && \
     [ "$update_flag" != "true" ] && [ "$update_flag" != "1" ]; then
    export AUDIOBOOKER_SKIP_TTS_SYNC=1
    return
  fi

  echo "[tts] Checking runtime TTS dependencies"
  if [ "$update_flag" = "true" ] || [ "$update_flag" = "1" ]; then
    python tts_dependency_manager.py --apply || echo "[tts] Warning: update failed, continuing"
  else
    python tts_dependency_manager.py || echo "[tts] Warning: check failed, continuing"
  fi

  export AUDIOBOOKER_SKIP_TTS_SYNC=1
}

sync_tts_deps

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
