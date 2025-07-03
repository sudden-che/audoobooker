import asyncio
from edge_tts import Communicate
from pathlib import Path
import os
from pydub import AudioSegment

AudioSegment.converter = os.path.join( "ffmpeg.exe")
AudioSegment.ffprobe = os.path.join( "ffprobe.exe")

# === Настройки ===
INPUT_FILE = "Vafin_Govorit-Vafin.IOENYQ.559764.txt"

OUTPUT_DIR = Path(INPUT_FILE).stem

VOICE = "ru-RU-SvetlanaNeural"
CHUNK_SIZE = 5000
MAX_CONCURRENT_TASKS = 20

# === Подготовка ===
Path(OUTPUT_DIR).mkdir(exist_ok=True)
text = Path(INPUT_FILE).read_text(encoding="utf-8")
chunks = [text[i:i + CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]

semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)

async def synthesize_chunk(text, file_path):
    async with semaphore:
        communicate = Communicate(text=text, voice=VOICE, rate="+12%")
        await communicate.save(file_path)
        print(f"[+] Сохранено: {file_path}")

async def main():
    print(f"[=] Генерация {len(chunks)} фрагментов параллельно...")

    tasks = []
    for i, chunk in enumerate(chunks):
        file_path = f"{OUTPUT_DIR}/chunk_{i:03}.mp3"
        tasks.append(synthesize_chunk(chunk, file_path))

    await asyncio.gather(*tasks)

    print("[=] Склейка всех фрагментов в audiobook.mp3 ...")
    combined = AudioSegment.empty()
    for i in range(len(chunks)):
        part_file = f"{OUTPUT_DIR}/chunk_{i:03}.mp3"
        if Path(part_file).exists():
            part = AudioSegment.from_mp3(part_file)
            combined += part
        else:
            print(f"[!] Пропущен отсутствующий файл: {part_file}")

    combined.export(f"{OUTPUT_DIR}.mp3", format="mp3")
    print(f"✅ Готово: {OUTPUT_DIR}.mp3")

# === Запуск ===
asyncio.run(main())
