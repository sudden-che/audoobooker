#!/usr/bin/env python3
import asyncio
from edge_tts import Communicate
from pathlib import Path
import subprocess
import sys

# === ФЛАГИ ПОВЕДЕНИЯ ===
SKIP_CHUNKS = False  # Если True — не перезаписывать уже существующие чанки
SKIP_MERGE = False  # Если True — не склеивать чанки в один mp3

# === ПУТЬ К FFMPEG (если не в PATH) ===
FFMPEG_PATH = r"/opt/homebrew/bin/ffmpeg"

# === НАСТРОЙКИ ОЗВУЧКИ ===
OUTPUT_DIR = "output"
VOICE = "ru-RU-SvetlanaNeural"
# VOICE = "ru-RU-DmitryNeural"
SPEED = "+18%"

CHUNK_SIZE = 10000  # Увеличено для уменьшения количества запросов
MAX_CONCURRENT_TASKS = 40  # Увеличено для большей параллельности

# Глобальный семафор, чтобы не улететь в космос по числу одновременных запросов
semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)


# === УТИЛИТЫ ===


async def synthesize_chunk(text: str, file_path: Path) -> None:
    """Синтез одного фрагмента текста в mp3-файл."""
    # Проверка существования файла вынесена наружу для оптимизации
    async with semaphore:
        communicate = Communicate(text=text, voice=VOICE, rate=SPEED)
        await communicate.save(str(file_path))
        print(f"[+] Сохранено: {file_path}")


def convert_fb2_to_txt(input_file: Path) -> Path:
    """
    Конвертация FB2 → TXT через fb2_to_txt.py.
    Возвращает путь к .txt-файлу.
    """
    print(f"[=] Обнаружен FB2-файл: {input_file.name}")
    print("[=] Конвертация через fb2_to_txt.py ...")

    result = subprocess.run(
        [sys.executable, "fb2_to_txt.py", str(input_file)],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print("X Ошибка при конвертации FB2:")
        print(result.stderr)
        raise RuntimeError("Не удалось конвертировать FB2 в TXT")

    print(result.stdout)
    txt_file = input_file.with_suffix(".txt")
    if not txt_file.exists():
        raise FileNotFoundError(
            f"Ожидался TXT после конвертации, но не найден: {txt_file}"
        )
    return txt_file


async def process_single_file(input_file: Path) -> None:
    """
    Полный цикл обработки одного файла:
    - при необходимости конвертация FB2 → TXT;
    - нарезка текста на чанки;
    - генерация mp3-фрагментов;
    - опциональная склейка через ffmpeg.
    """
    original_file = input_file

    # FB2 → TXT при необходимости
    if input_file.suffix.lower() == ".fb2":
        try:
            input_file = convert_fb2_to_txt(input_file)
        except Exception as e:
            print(
                f"[X] Пропуск файла {original_file.name} из-за ошибки конвертации: {e}"
            )
            return

    if input_file.suffix.lower() != ".txt":
        print(
            f"[!] Пропуск {input_file.name}: неподдерживаемое расширение {input_file.suffix}"
        )
        return

    try:
        text = input_file.read_text(encoding="utf-8")
    except Exception as e:
        print(f"[X] Не удалось прочитать {input_file}: {e}")
        return

    if not text.strip():
        print(f"[!] Файл пустой, пропуск: {input_file}")
        return

    # Подготовка выхода
    output_name = input_file.stem
    output_path = Path(OUTPUT_DIR) / output_name
    output_path.mkdir(parents=True, exist_ok=True)

    # Нарезка на чанки
    chunks = [text[i : i + CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]
    print(f"\n[=] Файл: {original_file.name} → {len(chunks)} фрагментов")

    # Запуск задач на синтез (проверка существующих файлов до создания задач)
    tasks = []
    existing_count = 0
    for i, chunk in enumerate(chunks):
        chunk_file = output_path / f"{output_name}_chunk_{i:06}.mp3"
        if SKIP_CHUNKS and chunk_file.exists():
            existing_count += 1
            continue
        tasks.append(asyncio.create_task(synthesize_chunk(chunk, chunk_file)))

    if existing_count > 0:
        print(f"[~] Пропущено существующих фрагментов: {existing_count}")

    # Ждём окончания всех чанков
    if tasks:
        await asyncio.gather(*tasks)

    # Если склейка отключена — выходим
    if SKIP_MERGE:
        print("[~] Объединение отключено (SKIP_MERGE = True)")
        return

    # Подготовка list.txt для ffmpeg
    print("[=] Подготовка list.txt для ffmpeg ...")
    list_file = output_path / "list.txt"
    with list_file.open("w", encoding="utf-8") as f:
        for i in range(len(chunks)):
            part_path = (output_path / f"{output_name}_chunk_{i:06}.mp3").resolve()
            f.write(f"file '{part_path.as_posix()}'\n")

    # Склейка через ffmpeg без перекодирования (асинхронно, с подавлением вывода)
    output_file = output_path / f"full_{output_name}.mp3"
    print("[=] Склейка через ffmpeg без перекодирования ...")
    await asyncio.to_thread(
        subprocess.run,
        [
            FFMPEG_PATH,
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_file),
            "-c",
            "copy",
            "-loglevel",
            "error",  # Подавляем лишний вывод для ускорения
            str(output_file),
        ],
        check=True,
        capture_output=True,  # Не показываем вывод ffmpeg
    )

    print(f"[!] Готово: {output_file}")


# === ТОЧКА ВХОДА ===


async def main() -> None:
    # Аргумент командной строки:
    #   - файл (txt/fb2)   → обработать один файл
    #   - каталог          → обработать все txt/fb2 в нём
    #   - без аргумента    → попытаться обработать sample.txt
    if len(sys.argv) > 1:
        input_path = Path(sys.argv[1])
    else:
        input_path = Path("sample.txt")

    if not input_path.exists():
        print(f"[X] Путь не существует: {input_path}")
        sys.exit(1)

    # Режим: каталог
    if input_path.is_dir():
        print(f"[=] Каталог: {input_path}")
        # Берём только файлы с нужными расширениями
        files = sorted(
            f
            for f in input_path.iterdir()
            if f.is_file() and f.suffix.lower() in {".txt", ".fb2"}
        )

        if not files:
            print("[X] В каталоге нет *.txt или *.fb2 файлов для обработки")
            return

        print(f"[=] Найдено {len(files)} файлов для обработки:")
        for f in files:
            print(f"    - {f.name}")

        # Параллельная обработка файлов (но с ограничением через семафор)
        # Создаём отдельный семафор для файлов, чтобы не перегружать систему
        file_semaphore = asyncio.Semaphore(3)  # Максимум 3 файла одновременно

        async def process_with_semaphore(f: Path) -> None:
            async with file_semaphore:
                print(f"\n[=] === Обработка файла: {f.name} ===")
                await process_single_file(f)

        # Обрабатываем все файлы параллельно
        await asyncio.gather(*[process_with_semaphore(f) for f in files])

    # Режим: одиночный файл
    else:
        print(f"[=] Одиночный файл: {input_path.name}")
        await process_single_file(input_path)


if __name__ == "__main__":
    asyncio.run(main())
