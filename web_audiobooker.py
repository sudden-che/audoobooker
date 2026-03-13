#!/usr/bin/env python3
import asyncio
import os
import shutil
import tarfile
import tempfile
import uuid
from pathlib import Path
from typing import Any, Awaitable, Callable

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from tts_dependency_manager import sync_tts_dependencies_from_env
from audiobooker import (
    apply_audiobook_best_practices,
    build_output_basename,
    collect_valid_audio_files,
    extract_fb2_text,
    synthesize_chunk_edge,
    synthesize_chunk_silero,
    merge_audio_chunks,
    convert_to_mp3,
    split_text,
)

# ============================================================
# Настройки сервера (через env)
# ============================================================
WEB_HOST = os.environ.get("WEB_HOST", "0.0.0.0")
WEB_PORT = int(os.environ.get("WEB_PORT", "8000"))

# Дефолты формы
DEFAULT_ENGINE = os.environ.get("WEB_TTS_ENGINE", "edge").lower()
DEFAULT_VOICE = os.environ.get("WEB_VOICE", os.environ.get("VOICE", "ru-RU-SvetlanaNeural"))
DEFAULT_SPEED = os.environ.get("WEB_SPEED", os.environ.get("SPEED", "+18%"))
DEFAULT_CHUNK_SIZE = int(os.environ.get("WEB_CHUNK_SIZE", os.environ.get("CHUNK_SIZE", "10000")))
DEFAULT_MAX_TASKS = os.environ.get("WEB_MAX_CONCURRENT_TASKS", os.environ.get("MAX_CONCURRENT_TASKS", ""))
DEFAULT_FFMPEG = os.environ.get("WEB_FFMPEG_PATH", os.environ.get("FFMPEG_PATH", "ffmpeg"))

# Silero дефолты
DEFAULT_SILERO_SPEAKER = os.environ.get("WEB_SILERO_SPEAKER", os.environ.get("SILERO_SPEAKER", "baya"))
DEFAULT_SILERO_SAMPLE_RATE = int(os.environ.get("WEB_SILERO_SAMPLE_RATE", os.environ.get("SILERO_SAMPLE_RATE", "48000")))
DEFAULT_DEVICE = os.environ.get("WEB_DEVICE", os.environ.get("DEVICE", "cpu"))
DEFAULT_SILERO_MODEL_ID = os.environ.get("WEB_SILERO_MODEL_ID", os.environ.get("SILERO_MODEL_ID", "v5_ru"))
WEB_JOB_TTL_SECONDS = int(os.environ.get("WEB_JOB_TTL_SECONDS", "3600"))
# ============================================================

app = FastAPI(title="Audiobooker Web")
WEB_JOBS: dict[str, dict[str, Any]] = {}


def _get_job(job_id: str) -> dict[str, Any]:
    job = WEB_JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    return job


def _set_job_state(
    job_id: str,
    *,
    status: str | None = None,
    message: str | None = None,
    detail: str | None = None,
    completed_chunks: int | None = None,
    total_chunks: int | None = None,
    result_path: Path | None = None,
    result_kind: str | None = None,
    filename: str | None = None,
    media_type: str | None = None,
    error: str | None = None,
) -> None:
    job = _get_job(job_id)
    if status is not None:
        job["status"] = status
    if message is not None:
        job["message"] = message
    if detail is not None:
        job["detail"] = detail
    if completed_chunks is not None:
        job["completed_chunks"] = completed_chunks
    if total_chunks is not None:
        job["total_chunks"] = total_chunks
    if result_path is not None:
        job["result_path"] = result_path
    if result_kind is not None:
        job["result_kind"] = result_kind
    if filename is not None:
        job["filename"] = filename
    if media_type is not None:
        job["media_type"] = media_type
    if error is not None:
        job["error"] = error


def _serialize_job(job_id: str) -> dict[str, Any]:
    job = _get_job(job_id)
    completed = int(job.get("completed_chunks", 0) or 0)
    total = int(job.get("total_chunks", 0) or 0)
    percent = 0 if total <= 0 else min(100, int(completed * 100 / total))

    payload = {
        "job_id": job_id,
        "status": job.get("status", "queued"),
        "message": job.get("message", ""),
        "detail": job.get("detail", ""),
        "completed_chunks": completed,
        "total_chunks": total,
        "progress_percent": percent,
        "filename": job.get("filename"),
        "error": job.get("error"),
        "download_url": f"/download/{job_id}" if job.get("status") == "done" else None,
    }
    return payload


async def _call_callback(
    callback: Callable[..., Any] | None,
    *args: Any,
) -> None:
    if callback is None:
        return
    result = callback(*args)
    if asyncio.iscoroutine(result):
        await result


async def _cleanup_job_later(job_id: str, delay_seconds: int = WEB_JOB_TTL_SECONDS) -> None:
    await asyncio.sleep(max(delay_seconds, 0))
    job = WEB_JOBS.pop(job_id, None)
    if job is None:
        return
    tmp_root = job.get("tmp_root")
    if isinstance(tmp_root, Path):
        shutil.rmtree(tmp_root, ignore_errors=True)


async def process_uploaded_file(
    input_file: Path,
    output_dir: Path,
    engine: str,
    voice: str,
    speed: str,
    silero_speaker: str,
    sample_rate: int,
    put_accent: bool,
    put_yo: bool,
    device: str,
    silero_model_id: str,
    chunk_size: int,
    max_concurrent_tasks: int,
    skip_chunks: bool,
    merge_chunks: bool,
    ffmpeg_path: str,
    source_name: str | None = None,
    on_stage: Callable[..., Any] | None = None,
    on_progress: Callable[..., Any] | None = None,
) -> tuple[Path, str]:
    await _call_callback(on_stage, "preparing", "Читаю и подготавливаю текст")
    if input_file.suffix.lower() == ".fb2":
        text = extract_fb2_text(input_file)
    elif input_file.suffix.lower() == ".txt":
        text = input_file.read_text(encoding="utf-8")
    else:
        raise HTTPException(
            status_code=400, detail="Поддерживаются только .txt и .fb2 файлы"
        )

    text = apply_audiobook_best_practices(text, lang="ru")

    if not text.strip():
        raise HTTPException(status_code=400, detail="Загруженный файл пуст")

    name = build_output_basename(source_name or input_file.name)
    parts_dir = output_dir / f"{name}_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    chunks = split_text(text, chunk_size)
    total_chunks = len(chunks)
    await _call_callback(
        on_stage,
        "synthesizing",
        "Генерирую аудио",
        f"Подготовлено чанков: {total_chunks}",
    )
    await _call_callback(on_progress, 0, total_chunks)
    semaphore = asyncio.Semaphore(max_concurrent_tasks)

    ext = "mp3" if engine == "edge" else "wav"
    tasks = []
    completed_chunks = 0

    async def _tracked_task(coro: Awaitable[None]) -> None:
        nonlocal completed_chunks
        await coro
        completed_chunks += 1
        await _call_callback(on_progress, completed_chunks, total_chunks)

    for i, chunk in enumerate(chunks):
        chunk_file = parts_dir / f"{name}_chunk_{i:06}.{ext}"
        if skip_chunks:
            valid_chunks = collect_valid_audio_files(
                [chunk_file],
                expected_ext=ext,
                remove_invalid=True,
            )
            if valid_chunks:
                completed_chunks += 1
                continue

        if engine == "edge":
            tasks.append(
                asyncio.create_task(
                    _tracked_task(
                        synthesize_chunk_edge(
                            text=chunk,
                            file_path=chunk_file,
                            voice=voice,
                            rate=speed,
                            semaphore=semaphore,
                        )
                    )
                )
            )
        else:
            tasks.append(
                asyncio.create_task(
                    _tracked_task(
                        synthesize_chunk_silero(
                            text=chunk,
                            file_path=chunk_file,
                            language="ru",
                            speaker=silero_speaker,
                            sample_rate=sample_rate,
                            put_accent=put_accent,
                            put_yo=put_yo,
                            device=device,
                            model_id=silero_model_id,
                            semaphore=semaphore,
                        )
                    )
                )
            )

    await _call_callback(on_progress, completed_chunks, total_chunks)
    await asyncio.gather(*tasks)
    expected_parts = [parts_dir / f"{name}_chunk_{i:06}.{ext}" for i in range(len(chunks))]
    actual_parts = collect_valid_audio_files(
        expected_parts,
        expected_ext=ext,
        remove_invalid=True,
    )

    if not actual_parts:
        raise HTTPException(
            status_code=400,
            detail="Не удалось сгенерировать ни одного валидного аудиочанка",
        )

    if merge_chunks:
        await _call_callback(
            on_stage,
            "merging",
            "Склеиваю результат",
            f"Готово чанков: {len(actual_parts)}",
        )
        list_file = parts_dir / "list.txt"
        with list_file.open("w", encoding="utf-8") as f:
            for part_path in actual_parts:
                part_path = part_path.resolve()
                f.write(f"file '{part_path.as_posix()}'\n")

        full_file = output_dir / f"{name}.{ext}"
        await merge_audio_chunks(
            ffmpeg_path=ffmpeg_path,
            list_file=list_file,
            output_file=full_file,
        )

        if ext == "wav":
            mp3_file = output_dir / f"{name}.mp3"
            try:
                await convert_to_mp3(
                    ffmpeg_path=ffmpeg_path,
                    input_audio=full_file,
                    output_mp3=mp3_file,
                )
                return mp3_file, "single"
            except Exception:
                return full_file, "single"

        return full_file, "single"

    await _call_callback(
        on_stage,
        "packing",
        "Упаковываю чанки в TAR",
        f"Файлов: {len(actual_parts)}",
    )
    tar_path = output_dir / f"{name}_parts.tar"
    with tarfile.open(tar_path, "w") as tar:
        for audio_file in actual_parts:
            tar.add(audio_file, arcname=audio_file.name)
    return tar_path, "tar"


async def _run_web_job(
    *,
    job_id: str,
    input_path: Path,
    tmp_root: Path,
    source_name: str,
    engine: str,
    voice: str,
    speed: str,
    silero_speaker: str,
    sample_rate: int,
    put_accent: bool,
    put_yo: bool,
    device: str,
    silero_model_id: str,
    chunk_size: int,
    max_concurrent_tasks: int,
    skip_chunks: bool,
    merge_chunks: bool,
    ffmpeg_path: str,
) -> None:
    def _update_stage(status: str, message: str, detail: str = "") -> None:
        _set_job_state(
            job_id,
            status=status,
            message=message,
            detail=detail,
        )

    def _update_progress(completed: int, total: int) -> None:
        _set_job_state(
            job_id,
            completed_chunks=completed,
            total_chunks=total,
        )

    try:
        result_path, result_kind = await process_uploaded_file(
            input_file=input_path,
            output_dir=tmp_root,
            engine=engine,
            voice=voice,
            speed=speed,
            silero_speaker=silero_speaker,
            sample_rate=sample_rate,
            put_accent=put_accent,
            put_yo=put_yo,
            device=device,
            silero_model_id=silero_model_id,
            chunk_size=chunk_size,
            max_concurrent_tasks=max_concurrent_tasks,
            skip_chunks=skip_chunks,
            merge_chunks=merge_chunks,
            ffmpeg_path=ffmpeg_path,
            source_name=source_name,
            on_stage=_update_stage,
            on_progress=_update_progress,
        )
    except Exception as exc:
        _set_job_state(
            job_id,
            status="error",
            message="Ошибка обработки",
            detail=str(exc),
            error=str(exc),
        )
        asyncio.create_task(_cleanup_job_later(job_id))
        return

    if result_kind == "single":
        media_type = "audio/mpeg" if result_path.suffix == ".mp3" else "audio/wav"
    else:
        media_type = "application/x-tar"

    _set_job_state(
        job_id,
        status="done",
        message="Готово к скачиванию",
        detail=result_path.name,
        result_path=result_path,
        result_kind=result_kind,
        filename=result_path.name,
        media_type=media_type,
    )
    asyncio.create_task(_cleanup_job_later(job_id))


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return f"""
<!doctype html>
<html lang="ru">
<head>
<meta charset="UTF-8" />
<title>Audiobooker Web</title>
<style>
body {{ max-width: 860px; margin: 2rem auto; font-family: Arial, sans-serif; line-height: 1.5; }}
fieldset {{ margin-bottom: 1.5rem; padding: 1rem; border: 1px solid #ccc; border-radius: 8px; }}
legend {{ font-weight: bold; padding: 0 0.5rem; }}
label {{ display: block; margin: .8rem 0; }}
input[type='text'], input[type='number'], select {{ width: 100%; padding: .5rem; box-sizing: border-box; }}
.inline-label {{ display: flex; align-items: center; gap: 0.5rem; }}
.inline-label input {{ width: auto; }}
button {{ padding: .7rem 1.5rem; font-size: 1.1rem; cursor: pointer; background: #007bff; color: white; border: none; border-radius: 4px; }}
button:hover {{ background: #0056b3; }}
button:disabled {{ background: #7b9ac0; cursor: wait; }}
small {{ color: #666; display: block; margin-top: 0.5rem; }}
.engine-settings {{ margin-top: 1rem; padding-top: 1rem; border-top: 1px dashed #eee; }}
.progress-panel {{ display: none; margin: 1.5rem 0; padding: 1rem; border: 1px solid #cfd8e3; border-radius: 8px; background: #f8fbff; }}
.progress-panel.visible {{ display: block; }}
.progress-bar {{ width: 100%; height: 18px; border-radius: 999px; background: #e5edf6; overflow: hidden; margin: .75rem 0 .5rem; }}
.progress-fill {{ width: 0%; height: 100%; background: linear-gradient(90deg, #007bff, #35a7ff); transition: width .3s ease; }}
.status-line {{ font-weight: bold; }}
.status-detail {{ color: #555; min-height: 1.3rem; }}
.status-meta {{ color: #333; font-size: .95rem; }}
.error-text {{ color: #a12424; }}
.success-text {{ color: #126b2f; }}
.download-link {{ display: inline-block; margin-top: .75rem; }}
</style>
<script>
function toggleEngine() {{
    const engine = document.querySelector('select[name="engine"]').value;
    document.getElementById('edge-settings').style.display = (engine === 'edge') ? 'block' : 'none';
    document.getElementById('silero-settings').style.display = (engine === 'silero') ? 'block' : 'none';
}}

let currentJobId = null;
let pollTimer = null;

function setProgressPanelVisible(visible) {{
    const panel = document.getElementById('progress-panel');
    panel.classList.toggle('visible', visible);
}}

function updateProgressUi(payload) {{
    const statusLine = document.getElementById('status-line');
    const statusDetail = document.getElementById('status-detail');
    const statusMeta = document.getElementById('status-meta');
    const progressFill = document.getElementById('progress-fill');
    const downloadWrap = document.getElementById('download-wrap');

    statusLine.textContent = payload.message || 'Обработка';
    statusDetail.textContent = payload.error || payload.detail || '';
    progressFill.style.width = `${{payload.progress_percent || 0}}%`;

    if ((payload.total_chunks || 0) > 0) {{
        statusMeta.textContent = `Чанки: ${{payload.completed_chunks}} / ${{payload.total_chunks}} (${{payload.progress_percent}}%)`;
    }} else {{
        statusMeta.textContent = payload.status === 'uploading' ? 'Идёт отправка файла на сервер' : '';
    }}

    if (payload.status === 'error') {{
        statusLine.className = 'status-line error-text';
        downloadWrap.innerHTML = '';
        downloadWrap.dataset.autodownload = '';
        return;
    }}

    if (payload.status === 'done' && payload.download_url) {{
        statusLine.className = 'status-line success-text';
        downloadWrap.innerHTML = `<a class="download-link" href="${{payload.download_url}}">Скачать ${{payload.filename || 'файл'}}</a>`;
        if (downloadWrap.dataset.autodownload !== payload.download_url) {{
            downloadWrap.dataset.autodownload = payload.download_url;
            window.location.assign(payload.download_url);
        }}
        return;
    }}

    statusLine.className = 'status-line';
    downloadWrap.innerHTML = '';
    downloadWrap.dataset.autodownload = '';
}}

function setSubmitting(isSubmitting) {{
    const button = document.getElementById('submit-button');
    button.disabled = isSubmitting;
    button.textContent = isSubmitting ? 'Processing...' : 'Start';
}}

function stopPolling() {{
    if (pollTimer !== null) {{
        window.clearTimeout(pollTimer);
        pollTimer = null;
    }}
}}

async function pollJobStatus(jobId) {{
    try {{
        const response = await fetch(`/status/${{jobId}}`, {{ cache: 'no-store' }});
        const payload = await response.json();
        if (!response.ok) {{
            throw new Error(payload.detail || 'Не удалось получить статус задачи');
        }}

        updateProgressUi(payload);
        if (payload.status === 'done' || payload.status === 'error') {{
            setSubmitting(false);
            stopPolling();
            return;
        }}

        pollTimer = window.setTimeout(() => pollJobStatus(jobId), 1000);
    }} catch (error) {{
        updateProgressUi({{
            status: 'error',
            message: 'Ошибка статуса',
            error: error.message || String(error),
            progress_percent: 0,
            completed_chunks: 0,
            total_chunks: 0,
        }});
        setSubmitting(false);
        stopPolling();
    }}
}}

function submitForm(event) {{
    event.preventDefault();
    stopPolling();
    setProgressPanelVisible(true);
    setSubmitting(true);
    updateProgressUi({{
        status: 'uploading',
        message: 'Загружаю файл',
        detail: 'Файл отправляется на сервер...',
        progress_percent: 0,
        completed_chunks: 0,
        total_chunks: 0,
    }});

    const form = event.target;
    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/start');
    xhr.responseType = 'json';

    xhr.upload.onprogress = function(uploadEvent) {{
        if (!uploadEvent.lengthComputable) {{
            return;
        }}
        const percent = Math.min(100, Math.round((uploadEvent.loaded / uploadEvent.total) * 100));
        updateProgressUi({{
            status: 'uploading',
            message: 'Загружаю файл',
            detail: `Отправлено ${{percent}}%`,
            progress_percent: percent,
            completed_chunks: 0,
            total_chunks: 0,
        }});
    }};

    xhr.onload = function() {{
        const payload = xhr.response || {{}};
        if (xhr.status < 200 || xhr.status >= 300) {{
            updateProgressUi({{
                status: 'error',
                message: 'Ошибка запуска',
                error: payload.detail || 'Сервер не смог запустить задачу',
                progress_percent: 0,
                completed_chunks: 0,
                total_chunks: 0,
            }});
            setSubmitting(false);
            return;
        }}

        currentJobId = payload.job_id;
        updateProgressUi({{
            status: 'queued',
            message: 'Задача создана',
            detail: 'Начинаю обработку...',
            progress_percent: 0,
            completed_chunks: 0,
            total_chunks: 0,
        }});
        pollJobStatus(currentJobId);
    }};

    xhr.onerror = function() {{
        updateProgressUi({{
            status: 'error',
            message: 'Ошибка сети',
            error: 'Не удалось отправить файл на сервер',
            progress_percent: 0,
            completed_chunks: 0,
            total_chunks: 0,
        }});
        setSubmitting(false);
    }};

    xhr.send(new FormData(form));
}}
</script>
</head>
<body onload="toggleEngine()">
<h1>Audiobooker Web</h1>
<p>Загрузите <b>.txt</b> или <b>.fb2</b>, выберите движок и нажмите <b>Start</b>.</p>
<p><small>По умолчанию веб-версия использует Edge TTS. Перед синтезом текст автоматически нормализуется под аудиокнигу: главы, паузы, символы и диалоги.</small></p>
<div id="progress-panel" class="progress-panel" aria-live="polite">
  <div id="status-line" class="status-line">Ожидание</div>
  <div class="progress-bar"><div id="progress-fill" class="progress-fill"></div></div>
  <div id="status-detail" class="status-detail"></div>
  <div id="status-meta" class="status-meta"></div>
  <div id="download-wrap"></div>
</div>
<form method="post" action="/start" enctype="multipart/form-data" onsubmit="submitForm(event)">
  <fieldset>
    <legend>Входной файл</legend>
    <label>Файл (.txt/.fb2): <input type="file" name="book_file" accept=".txt,.fb2" required></label>
  </fieldset>
  <fieldset>
    <legend>Настройки TTS</legend>
    <label>Движок:
      <select name="engine" onchange="toggleEngine()">
        <option value="edge" {"selected" if DEFAULT_ENGINE=="edge" else ""}>Edge TTS (Online)</option>
        <option value="silero" {"selected" if DEFAULT_ENGINE=="silero" else ""}>Silero TTS (Local)</option>
      </select>
    </label>

    <div id="edge-settings" class="engine-settings">
      <label>Voice: <input type="text" name="voice" value="{DEFAULT_VOICE}"></label>
      <label>Speed: <input type="text" name="speed" value="{DEFAULT_SPEED}"></label>
    </div>

    <div id="silero-settings" class="engine-settings">
      <label>Speaker:
        <select name="silero_speaker">
          <option value="aidar">Aidar</option>
          <option value="baya" selected>Baya</option>
          <option value="kseniya">Kseniya</option>
          <option value="xenia">Xenia</option>
          <option value="eugene">Eugene</option>
        </select>
      </label>
      <label>Sample Rate:
        <select name="sample_rate">
          <option value="8000">8000</option>
          <option value="24000">24000</option>
          <option value="48000" selected>48000</option>
        </select>
      </label>
      <label class="inline-label"><input type="checkbox" name="put_accent" checked> Ставить ударения</label>
      <label class="inline-label"><input type="checkbox" name="put_yo" checked> Заменять "е" на "ё"</label>
      <label>Device: <input type="text" name="device" value="{DEFAULT_DEVICE}"></label>
      <label>Model ID: <input type="text" name="silero_model_id" value="{DEFAULT_SILERO_MODEL_ID}"></label>
    </div>

    <div class="engine-settings">
      <label>Chunk size: <input type="number" name="chunk_size" value="{DEFAULT_CHUNK_SIZE}" min="100"></label>
      <label>Max concurrent tasks: <input type="number" name="max_concurrent_tasks" value="{DEFAULT_MAX_TASKS or (40 if DEFAULT_ENGINE=='edge' else 2)}" min="1" max="100"></label>
      <label class="inline-label"><input type="checkbox" name="skip_chunks"> Skip existing chunks</label>
      <label class="inline-label"><input type="checkbox" name="merge_chunks" checked> Merge chunks (requires ffmpeg)</label>
      <label>ffmpeg path: <input type="text" name="ffmpeg_path" value="{DEFAULT_FFMPEG}"></label>
    </div>
  </fieldset>
  <button id="submit-button" type="submit">Start</button>
</form>
<p><small>Если склейка выключена, будет скачан TAR с чанками.</small></p>
</body>
</html>
"""


@app.post("/start")
async def start(
    book_file: UploadFile = File(...),
    engine: str = Form("edge"),
    voice: str = Form(None),
    speed: str = Form(None),
    silero_speaker: str = Form(None),
    sample_rate: int = Form(None),
    put_accent: bool = Form(True),
    put_yo: bool = Form(True),
    device: str = Form(None),
    silero_model_id: str = Form(None),
    chunk_size: int = Form(None),
    max_concurrent_tasks: int = Form(None),
    skip_chunks: bool = Form(False),
    merge_chunks: bool = Form(False),
    ffmpeg_path: str = Form(None),
):
    if voice is None:
        voice = DEFAULT_VOICE
    if speed is None:
        speed = DEFAULT_SPEED
    if silero_speaker is None:
        silero_speaker = DEFAULT_SILERO_SPEAKER
    if sample_rate is None:
        sample_rate = DEFAULT_SILERO_SAMPLE_RATE
    if device is None:
        device = DEFAULT_DEVICE
    if silero_model_id is None:
        silero_model_id = DEFAULT_SILERO_MODEL_ID
    if chunk_size is None:
        chunk_size = DEFAULT_CHUNK_SIZE
    if max_concurrent_tasks is None:
        max_concurrent_tasks = int(DEFAULT_MAX_TASKS) if DEFAULT_MAX_TASKS else (40 if engine == "edge" else (os.cpu_count() or 2))
    if ffmpeg_path is None:
        ffmpeg_path = DEFAULT_FFMPEG

    suffix = Path(book_file.filename or "").suffix.lower()
    if suffix not in {".txt", ".fb2"}:
        raise HTTPException(
            status_code=400, detail="Поддерживаются только .txt и .fb2 файлы"
        )

    if chunk_size < 100:
        raise HTTPException(status_code=400, detail="chunk_size должен быть >= 100")
    if max_concurrent_tasks < 1:
        raise HTTPException(
            status_code=400, detail="max_concurrent_tasks должен быть >= 1"
        )

    tmp_root = Path(tempfile.gettempdir()) / f"audiobooker_web_{uuid.uuid4().hex}"
    tmp_root.mkdir(parents=True, exist_ok=True)
    input_path = tmp_root / f"uploaded{suffix}"

    content = await book_file.read()
    input_path.write_bytes(content)

    job_id = uuid.uuid4().hex
    WEB_JOBS[job_id] = {
        "status": "queued",
        "message": "Задача создана",
        "detail": book_file.filename or input_path.name,
        "completed_chunks": 0,
        "total_chunks": 0,
        "error": None,
        "tmp_root": tmp_root,
        "result_path": None,
        "result_kind": None,
        "filename": None,
        "media_type": None,
    }

    asyncio.create_task(
        _run_web_job(
            job_id=job_id,
            input_path=input_path,
            tmp_root=tmp_root,
            source_name=book_file.filename or input_path.name,
            engine=engine,
            voice=voice,
            speed=speed,
            silero_speaker=silero_speaker,
            sample_rate=sample_rate,
            put_accent=put_accent,
            put_yo=put_yo,
            device=device,
            silero_model_id=silero_model_id,
            chunk_size=chunk_size,
            max_concurrent_tasks=max_concurrent_tasks,
            skip_chunks=skip_chunks,
            merge_chunks=merge_chunks,
            ffmpeg_path=ffmpeg_path,
        )
    )

    return {"job_id": job_id}


@app.get("/status/{job_id}")
async def status(job_id: str) -> dict[str, Any]:
    return _serialize_job(job_id)


@app.get("/download/{job_id}")
async def download(job_id: str) -> FileResponse:
    job = _get_job(job_id)
    if job.get("status") != "done":
        raise HTTPException(status_code=409, detail="Файл ещё не готов")

    result_path = job.get("result_path")
    filename = job.get("filename")
    media_type = job.get("media_type")
    if not isinstance(result_path, Path) or not result_path.is_file():
        raise HTTPException(status_code=410, detail="Результат уже очищен или недоступен")

    return FileResponse(
        path=result_path,
        filename=str(filename or result_path.name),
        media_type=str(media_type or "application/octet-stream"),
    )


if __name__ == "__main__":
    import uvicorn

    sync_tts_dependencies_from_env()
    uvicorn.run("web_audiobooker:app", host=WEB_HOST, port=WEB_PORT, reload=False)
