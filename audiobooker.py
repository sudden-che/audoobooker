import asyncio
from edge_tts import Communicate
from pathlib import Path
import subprocess
import sys

# === Установка флагов ===
SKIP_CHUNKS = False
SKIP_MERGE = True

# === Указание пути к ffmpeg (если не в PATH) ===
FFMPEG_PATH = r"ffmpeg.exe"

# === Обработка аргументов ===
if len(sys.argv) > 1:
    INPUT_FILE = Path(sys.argv[1])
else:
    INPUT_FILE = Path("sample.txt")

# === Автоконвертация FB2 → TXT ===
if INPUT_FILE.suffix.lower() == ".fb2":
    print(f"[=] Обнаружен FB2-файл: {INPUT_FILE.name}")
    print("[=] Конвертация через fb2_to_txt.py ...")
    result = subprocess.run(
        [sys.executable, "fb2_to_txt.py", str(INPUT_FILE)],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print("X Ошибка при конвертации FB2:")
        print(result.stderr)
        sys.exit(1)
    else:
        print(result.stdout)
        INPUT_FILE = INPUT_FILE.with_suffix(".txt")

# === Настройки озвучки ===
OUTPUT_DIR = "output"
OUTPUT_NAME = INPUT_FILE.stem
VOICE = "ru-RU-SvetlanaNeural"
#VOICE = "ru-RU-DmitryNeural"

CHUNK_SIZE = 5000
MAX_CONCURRENT_TASKS = 20

# === Подготовка путей ===
OUTPUT_PATH = Path(OUTPUT_DIR) / OUTPUT_NAME
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

text = INPUT_FILE.read_text(encoding="utf-8")
chunks = [text[i:i + CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]
semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)

async def synthesize_chunk(text, file_path):
    if SKIP_CHUNKS and Path(file_path).exists():
        print(f"[~] Пропущено (уже существует): {file_path}")
        return
    async with semaphore:
        communicate = Communicate(text=text, voice=VOICE, rate="+12%")
        await communicate.save(file_path)
        print(f"[+] Сохранено: {file_path}")
async def main():
    print(f"[=] Генерация {len(chunks)} фрагментов параллельно...")

    # Очередь заданий
    tasks = []

    for i, chunk in enumerate(chunks):
        file_path = OUTPUT_PATH / f"{OUTPUT_NAME}_chunk_{i:06}.mp3"
        task = asyncio.create_task(synthesize_chunk(chunk, str(file_path)))
        tasks.append(task)

        # Ждём, если превышен лимит
        if len(tasks) >= MAX_CONCURRENT_TASKS:
            await asyncio.wait(tasks[:1])  # ждём первый, чтобы поддерживать поток
            tasks = tasks[1:]

    # Дождаться оставшихся
    if tasks:
        await asyncio.gather(*tasks)

    if SKIP_MERGE:
        print("[~] Объединение отключено (SKIP_MERGE = True)")
        return

    print("[=] Подготовка list.txt для ffmpeg ...")
    list_file = OUTPUT_PATH / "list.txt"
    with open(list_file, "w", encoding="utf-8") as f:
        for i in range(len(chunks)):
            part_path = (OUTPUT_PATH / f"{OUTPUT_NAME}_chunk_{i:06}.mp3").resolve()
            f.write(f"file '{part_path.as_posix()}'\n")

    output_file = OUTPUT_PATH / f"{OUTPUT_NAME}.mp3"
    print("[=] Склейка через ffmpeg без перекодирования ...")
    subprocess.run([
        FFMPEG_PATH,
        "-f", "concat",
        "-safe", "0",
        "-i", str(list_file),
        "-c", "copy",
        str(output_file)
    ], check=True)

    print(f"[!] Готово: {output_file}")


# === Запуск ===
asyncio.run(main())
