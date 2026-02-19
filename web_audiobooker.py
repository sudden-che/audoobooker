#!/usr/bin/env python3
import asyncio
import os
import shutil
import tarfile
import tempfile
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, BackgroundTasks
from fastapi.responses import FileResponse, HTMLResponse

from audiobooker import (
    clean_text,
    extract_fb2_text,
    synthesize_chunk_edge,
    synthesize_chunk_silero,
    merge_audio_chunks,
    convert_to_mp3,
)

# ============================================================
# Настройки сервера (через env)
# ============================================================
WEB_HOST = os.environ.get("WEB_HOST", "0.0.0.0")
WEB_PORT = int(os.environ.get("WEB_PORT", "8000"))

# Дефолты формы
DEFAULT_ENGINE = os.environ.get("TTS_ENGINE", "edge")
DEFAULT_VOICE = os.environ.get("VOICE", "ru-RU-SvetlanaNeural")
DEFAULT_SPEED = os.environ.get("SPEED", "+18%")
DEFAULT_CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "10000"))
DEFAULT_MAX_TASKS = os.environ.get("MAX_CONCURRENT_TASKS", "")
DEFAULT_FFMPEG = os.environ.get("FFMPEG_PATH", "ffmpeg")

# Silero дефолты
DEFAULT_SILERO_SPEAKER = os.environ.get("SILERO_SPEAKER", "baya")
DEFAULT_SILERO_SAMPLE_RATE = int(os.environ.get("SILERO_SAMPLE_RATE", "48000"))
DEFAULT_DEVICE = os.environ.get("DEVICE", "cpu")
DEFAULT_SILERO_MODEL_ID = os.environ.get("SILERO_MODEL_ID", "v5_ru")
# ============================================================

app = FastAPI(title="Audiobooker Web")


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
) -> tuple[Path, str]:
    if input_file.suffix.lower() == ".fb2":
        text = extract_fb2_text(input_file)
    elif input_file.suffix.lower() == ".txt":
        text = input_file.read_text(encoding="utf-8")
        text = clean_text(text)
    else:
        raise HTTPException(
            status_code=400, detail="Поддерживаются только .txt и .fb2 файлы"
        )

    if not text.strip():
        raise HTTPException(status_code=400, detail="Загруженный файл пуст")

    name = input_file.stem
    parts_dir = output_dir / f"{name}_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
    semaphore = asyncio.Semaphore(max_concurrent_tasks)

    ext = "mp3" if engine == "edge" else "wav"
    tasks = []
    for i, chunk in enumerate(chunks):
        chunk_file = parts_dir / f"{name}_chunk_{i:06}.{ext}"
        if skip_chunks and chunk_file.exists():
            continue

        if engine == "edge":
            tasks.append(
                asyncio.create_task(
                    synthesize_chunk_edge(
                        text=chunk,
                        file_path=chunk_file,
                        voice=voice,
                        rate=speed,
                        semaphore=semaphore,
                    )
                )
            )
        else:
            tasks.append(
                asyncio.create_task(
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

    await asyncio.gather(*tasks)

    if merge_chunks:
        list_file = parts_dir / "list.txt"
        with list_file.open("w", encoding="utf-8") as f:
            for i in range(len(chunks)):
                part_path = (parts_dir / f"{name}_chunk_{i:06}.{ext}").resolve()
                f.write(f"file '{part_path.as_posix()}'\n")

        full_file = output_dir / f"full_{name}.{ext}"
        await merge_audio_chunks(
            ffmpeg_path=ffmpeg_path,
            list_file=list_file,
            output_file=full_file,
        )

        if ext == "wav":
            mp3_file = output_dir / f"full_{name}.mp3"
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

    tar_path = output_dir / f"{name}_parts.tar"
    with tarfile.open(tar_path, "w") as tar:
        for audio_file in sorted(parts_dir.glob(f"*.{ext}")):
            tar.add(audio_file, arcname=audio_file.name)
    return tar_path, "tar"


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
small {{ color: #666; display: block; margin-top: 0.5rem; }}
.engine-settings {{ margin-top: 1rem; padding-top: 1rem; border-top: 1px dashed #eee; }}
</style>
<script>
function toggleEngine() {{
    const engine = document.querySelector('select[name="engine"]').value;
    document.getElementById('edge-settings').style.display = (engine === 'edge') ? 'block' : 'none';
    document.getElementById('silero-settings').style.display = (engine === 'silero') ? 'block' : 'none';
}}
</script>
</head>
<body onload="toggleEngine()">
<h1>Audiobooker Web</h1>
<p>Загрузите <b>.txt</b> или <b>.fb2</b>, выберите движок и нажмите <b>Start</b>.</p>
<form method="post" action="/start" enctype="multipart/form-data">
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
  <button type="submit">Start</button>
</form>
<p><small>Если склейка выключена, будет скачан TAR с чанками.</small></p>
</body>
</html>
"""


@app.post("/start")
async def start(
    background_tasks: BackgroundTasks,
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
        max_concurrent_tasks = int(DEFAULT_MAX_TASKS) if DEFAULT_MAX_TASKS else (40 if engine == "edge" else 2)
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
    )

    if result_kind == "single":
        media = "audio/mpeg" if result_path.suffix == ".mp3" else "audio/wav"
    else:
        media = "application/x-tar"
    filename = result_path.name

    # Очистка временной папки после отправки файла
    background_tasks.add_task(shutil.rmtree, tmp_root, ignore_errors=True)

    return FileResponse(path=result_path, filename=filename, media_type=media)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("web_audiobooker:app", host=WEB_HOST, port=WEB_PORT, reload=False)
