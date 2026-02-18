#!/usr/bin/env python3
import asyncio
import os
import shutil
import subprocess
import tarfile
import tempfile
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path
import re

from edge_tts import Communicate
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, BackgroundTasks
from fastapi.responses import FileResponse, HTMLResponse

# ============================================================
# Настройки сервера (через env)
# ============================================================
WEB_HOST = os.environ.get("WEB_HOST", "0.0.0.0")
WEB_PORT = int(os.environ.get("WEB_PORT", "8000"))

# Дефолты формы (env переопределяют значения по умолчанию в HTML)
DEFAULT_VOICE = os.environ.get("VOICE", "ru-RU-SvetlanaNeural")
DEFAULT_SPEED = os.environ.get("SPEED", "+18%")
DEFAULT_CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "10000"))
DEFAULT_MAX_TASKS = int(os.environ.get("MAX_CONCURRENT_TASKS", "40"))
DEFAULT_FFMPEG = os.environ.get("FFMPEG_PATH", "ffmpeg")
# ============================================================

app = FastAPI(title="Audiobooker Web")


def clean_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = text.replace("«", '"').replace("»", '"')
    text = "".join(c for c in text if c.isprintable() or c in "\n\t")
    text = re.sub(r"[ \t]+", " ", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_and_clean_text(fb2_path: Path) -> str:
    tree = ET.parse(fb2_path)
    root = tree.getroot()
    ns = {"fb": "http://www.gribuser.ru/xml/fictionbook/2.0"}
    paragraphs = root.findall('.//fb:body//fb:p', ns)

    def p_text(p_el):
        return "".join(p_el.itertext()).strip()

    lines = [p_text(p) for p in paragraphs if p_text(p)]
    raw_text = "\n\n".join(lines)
    return clean_text(raw_text)


async def synthesize_chunk(
    chunk_text: str,
    file_path: Path,
    voice: str,
    speed: str,
    semaphore: asyncio.Semaphore,
    skip_chunks: bool,
) -> None:
    if skip_chunks and file_path.exists():
        return
    async with semaphore:
        communicate = Communicate(text=chunk_text, voice=voice, rate=speed)
        await communicate.save(str(file_path))


async def process_uploaded_file(
    input_file: Path,
    output_dir: Path,
    voice: str,
    speed: str,
    chunk_size: int,
    max_concurrent_tasks: int,
    skip_chunks: bool,
    merge_chunks: bool,
    ffmpeg_path: str,
) -> tuple[Path, str]:
    if input_file.suffix.lower() == ".fb2":
        text = extract_and_clean_text(input_file)
    elif input_file.suffix.lower() == ".txt":
        text = input_file.read_text(encoding="utf-8")
    else:
        raise HTTPException(status_code=400, detail="Поддерживаются только .txt и .fb2 файлы")

    if not text.strip():
        raise HTTPException(status_code=400, detail="Загруженный файл пуст")

    name = input_file.stem
    parts_dir = output_dir / f"{name}_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
    semaphore = asyncio.Semaphore(max_concurrent_tasks)

    tasks = []
    for i, chunk in enumerate(chunks):
        chunk_file = parts_dir / f"{name}_chunk_{i:06}.mp3"
        tasks.append(
            asyncio.create_task(
                synthesize_chunk(
                    chunk_text=chunk,
                    file_path=chunk_file,
                    voice=voice,
                    speed=speed,
                    semaphore=semaphore,
                    skip_chunks=skip_chunks,
                )
            )
        )

    await asyncio.gather(*tasks)

    if merge_chunks:
        ffmpeg_bin = shutil.which(ffmpeg_path) if Path(ffmpeg_path).name == ffmpeg_path else ffmpeg_path
        if not ffmpeg_bin:
            raise HTTPException(status_code=400, detail=f"ffmpeg не найден: {ffmpeg_path}")

        list_file = parts_dir / "list.txt"
        with list_file.open("w", encoding="utf-8") as f:
            for i in range(len(chunks)):
                part_path = (parts_dir / f"{name}_chunk_{i:06}.mp3").resolve()
                f.write(f"file '{part_path.as_posix()}'\n")

        full_file = output_dir / f"full_{name}.mp3"
        try:
            subprocess.run(
                [
                    ffmpeg_bin,
                    "-f",
                    "concat",
                    "-safe",
                    "0",
                    "-i",
                    str(list_file),
                    "-c",
                    "copy",
                    str(full_file),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            raise HTTPException(status_code=500, detail=f"Ошибка ffmpeg: {exc.stderr}")

        return full_file, "single"

    tar_path = output_dir / f"{name}_parts.tar"
    with tarfile.open(tar_path, "w") as tar:
        for mp3 in sorted(parts_dir.glob("*.mp3")):
            tar.add(mp3, arcname=mp3.name)
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
body {{ max-width: 860px; margin: 2rem auto; font-family: Arial, sans-serif; }}
fieldset {{ margin-bottom: 1rem; }}
label {{ display: block; margin: .5rem 0; }}
input[type='text'], input[type='number'] {{ width: 100%; padding: .4rem; }}
button {{ padding: .7rem 1.2rem; font-size: 1rem; }}
small {{ color: #555; }}
</style>
</head>
<body>
<h1>Audiobooker Web</h1>
<p>Загрузите <b>.txt</b> или <b>.fb2</b>, задайте параметры и нажмите <b>Start</b>.</p>
<form method="post" action="/start" enctype="multipart/form-data">
  <fieldset>
    <legend>Входной файл</legend>
    <label>Файл (.txt/.fb2): <input type="file" name="book_file" accept=".txt,.fb2" required></label>
  </fieldset>
  <fieldset>
    <legend>Параметры синтеза</legend>
    <label>Voice: <input type="text" name="voice" value="{DEFAULT_VOICE}"></label>
    <label>Speed: <input type="text" name="speed" value="{DEFAULT_SPEED}"></label>
    <label>Chunk size: <input type="number" name="chunk_size" value="{DEFAULT_CHUNK_SIZE}" min="100"></label>
    <label>Max concurrent tasks: <input type="number" name="max_concurrent_tasks" value="{DEFAULT_MAX_TASKS}" min="1" max="100"></label>
    <label><input type="checkbox" name="skip_chunks"> Skip existing chunks</label>
    <label><input type="checkbox" name="merge_chunks" checked> Merge chunks into one MP3 (requires ffmpeg)</label>
    <label>ffmpeg path or binary name: <input type="text" name="ffmpeg_path" value="{DEFAULT_FFMPEG}"></label>
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
    voice: str = Form(None),
    speed: str = Form(None),
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
    if chunk_size is None:
        chunk_size = DEFAULT_CHUNK_SIZE
    if max_concurrent_tasks is None:
        max_concurrent_tasks = DEFAULT_MAX_TASKS
    if ffmpeg_path is None:
        ffmpeg_path = DEFAULT_FFMPEG

    suffix = Path(book_file.filename or "").suffix.lower()
    if suffix not in {".txt", ".fb2"}:
        raise HTTPException(status_code=400, detail="Поддерживаются только .txt и .fb2 файлы")

    if chunk_size < 100:
        raise HTTPException(status_code=400, detail="chunk_size должен быть >= 100")
    if max_concurrent_tasks < 1:
        raise HTTPException(status_code=400, detail="max_concurrent_tasks должен быть >= 1")

    tmp_root = Path(tempfile.gettempdir()) / f"audiobooker_web_{uuid.uuid4().hex}"
    tmp_root.mkdir(parents=True, exist_ok=True)
    input_path = tmp_root / f"uploaded{suffix}"

    content = await book_file.read()
    input_path.write_bytes(content)

    result_path, result_kind = await process_uploaded_file(
        input_file=input_path,
        output_dir=tmp_root,
        voice=voice,
        speed=speed,
        chunk_size=chunk_size,
        max_concurrent_tasks=max_concurrent_tasks,
        skip_chunks=skip_chunks,
        merge_chunks=merge_chunks,
        ffmpeg_path=ffmpeg_path,
    )

    media = "audio/mpeg" if result_kind == "single" else "application/x-tar"
    filename = result_path.name
    
    # Очистка временной папки после отправки файла
    background_tasks.add_task(shutil.rmtree, tmp_root, ignore_errors=True)
    
    return FileResponse(path=result_path, filename=filename, media_type=media)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("web_audiobooker:app", host=WEB_HOST, port=WEB_PORT, reload=False)
