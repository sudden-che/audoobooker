#!/usr/bin/env python3
import asyncio
import os
import shutil
from pathlib import Path

# Список доступных дикторов для моделей v4_ru и v5_ru
SPEAKERS = ["aidar", "baya", "kseniya", "xenia", "eugene"]
MODELS = ["v4_ru", "v5_ru"]

TEST_TEXT = "Привет! Это проверка голоса в системе синтеза речи Силеро. Как я звучу?"

async def run_test():
    output_root = Path("output_tests")
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(exist_ok=True)

    test_file = Path("test_sample.txt")
    test_file.write_text(TEST_TEXT, encoding="utf-8")

    print(f"--- Запуск тестирования голосов Silero ---")
    print(f"Текст: {TEST_TEXT}\n")

    for model_id in MODELS:
        for speaker in SPEAKERS:
            out_dir = output_root / f"{model_id}_{speaker}"
            cmd = [
                "./.venv/bin/python", "audiobooker.py",
                str(test_file),
                "--engine", "silero",
                "--silero-model-id", model_id,
                "--speaker", speaker,
                "--output-dir", str(output_root),
                "--skip-merge"  # пропускаем мерж, так как нет ffmpeg
            ]
            
            print(f"[*] Генерация: Модель={model_id}, Диктор={speaker}...")
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                # Переименовываем файл для удобства
                # audiobooker.py создает структуру output_dir / stem / stem_chunk_000000.wav
                src_file = output_root / "test_sample" / "test_sample_chunk_000000.wav"
                dst_file = output_root / f"sample_{model_id}_{speaker}.wav"
                if src_file.exists():
                    src_file.rename(dst_file)
                    print(f"  [+] Готово: {dst_file.name}")
            else:
                print(f"  [X] Ошибка для {model_id}_{speaker}:")
                print(stderr.decode())

    # Очистка временной папки, созданной audiobooker.py
    temp_folder = output_root / "test_sample"
    if temp_folder.exists():
        shutil.rmtree(temp_folder)
        
    if test_file.exists():
        test_file.unlink()

    print(f"\n--- Тестирование завершено ---")
    print(f"Все файлы сохранены в папке: {output_root.absolute()}")

if __name__ == "__main__":
    asyncio.run(run_test())
