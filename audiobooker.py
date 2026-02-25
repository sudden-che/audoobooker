#!/usr/bin/env python3
"""
fb2/txt -> TTS chunks -> merge

Engines:
  - edge: uses edge-tts (network), outputs MP3 chunks, merges MP3 by concat copy
  - silero: uses torch hub Silero TTS (local), outputs WAV chunks, merges WAV, optional final MP3

Requirements:
  - edge: pip install edge-tts
  - silero: pip install torch soundfile
  - ffmpeg: installed and accessible (or pass --ffmpeg)

Usage examples:
  python tts_book.py mybook.fb2 --engine edge --voice ru-RU-SvetlanaNeural --rate +18%
  python tts_book.py mybook.txt --engine silero --speaker baya --sample-rate 48000 --final-mp3

Notes:
  - Silero is usually CPU/GPU-bound; don't set huge concurrency.
  - Edge is network-bound; concurrency helps (within reason).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, Tuple, Callable, Any, cast

# -----------------------------
# Silero lazy globals
# -----------------------------
_silero_apply_tts: Optional[Callable] = None
_silero_inited: bool = False


def _require(cmd_name: str, hint: str) -> None:
    raise RuntimeError(f"Missing dependency for '{cmd_name}'. {hint}")


def clean_text(text: str) -> str:
    """Нормализация и очистка текста."""
    text = text.replace("\xa0", " ")
    text = text.replace("«", '"').replace("»", '"')
    text = "".join(c for c in text if c.isprintable() or c in "\n\t")
    text = re.sub(r"[ \t]+", " ", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def sanitize_for_tts(text: str) -> str:
    """
    Sanitizes text for Silero by removing characters that are known to cause issues.
    """
    text = text.replace("\xa0", " ").replace("\u200b", "")
    text = text.replace("«", '"').replace("»", '"').replace("“", '"').replace("”", '"')
    text = text.replace("—", "-").replace("–", "-")
    
    # Allow Cyrillic, Latin, numbers, and a wider range of punctuation/symbols.
    # Included %, $, +, =, @, #, &, *, <, >.
    allowed_chars_pattern = re.compile(r"[^а-яА-ЯёЁa-zA-Z0-9\s.,!?\-:;'\";%@#&$*+=<>]")
    text = allowed_chars_pattern.sub(' ', text)
    
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def normalize_numbers(text: str, lang: str = "ru") -> str:
    """Заменяет цифры на слова в тексте."""
    try:
        from num2words import num2words
    except ImportError:
        # Если библиотеки нет, возвращаем как есть
        return text

    def _replace(match: re.Match) -> str:
        number_str = match.group(0)
        try:
            # Превращаем "123" в "сто двадцать три"
            return num2words(int(number_str), lang=lang)
        except Exception:
            return number_str

    # Ищем последовательности цифр
    return re.sub(r"\d+", _replace, text)


def extract_fb2_text(fb2_path: Path) -> str:
    """Извлечение текста из FB2 файла."""
    tree = ET.parse(fb2_path)
    root = tree.getroot()
    ns = {"fb": "http://www.gribuser.ru/xml/fictionbook/2.0"}
    paragraphs = root.findall(".//fb:body//fb:p", ns)

    def p_text(p_el):
        return "".join(p_el.itertext()).strip()

    lines = [p_text(p) for p in paragraphs if p_text(p)]
    return clean_text("\n\n".join(lines))


def init_silero(language: str, model_id: str, device: str) -> None:
    """
    Lazy init Silero once per process.

    Loads Silero TTS via torch hub:
      torch.hub.load(repo_or_dir="snakers4/silero-models", model="silero_tts", language=..., speaker=...)
    """
    global _silero_apply_tts, _silero_inited

    if _silero_inited and _silero_apply_tts is not None:
        return

    try:
        import torch  # type: ignore
    except Exception as e:
        _require("silero", f"Install torch: pip install torch. Original error: {e}")

    dev = torch.device(device)

    # torch.hub will download models the first time; ensure you have network access then.
    res = torch.hub.load(
        repo_or_dir="snakers4/silero-models",
        model="silero_tts",
        language=language,
        speaker=model_id,
    )

    if isinstance(res, tuple) and len(res) == 2:
        model, _ = res
        model.to(dev)
        # In newer version, the model has apply_tts method
        _silero_apply_tts = model.apply_tts
    else:
        # Fallback for older versions that return 5 values
        model, symbols, sample_rate, example_text, apply_tts = res
        model.to(dev)
        _silero_apply_tts = apply_tts

    _silero_inited = True


async def synthesize_chunk_edge(
    text: str,
    file_path: Path,
    voice: str,
    rate: str,
    semaphore: asyncio.Semaphore,
) -> None:
    """Edge TTS chunk -> MP3."""
    # Нормализуем числа и очищаем текст перед синтезом (универсально для всех движков)
    text = normalize_numbers(text, lang="ru")
    text = sanitize_for_tts(text)

    try:
        from edge_tts import Communicate  # type: ignore
        from edge_tts.exceptions import NoAudioReceived  # type: ignore
    except Exception as e:
        _require("edge-tts", f"Install: pip install edge-tts. Original error: {e}")

    async with semaphore:
        # Проверяем, есть ли в тексте буквы или цифры. 
        # Если там только знаки препинания, edge-tts выдает "No audio received".
        if not any(c.isalpha() for c in text):
            print(f"[!] Skipping chunk (no alpha chars): {text[:20]}...")
            return

        try:
            communicate = Communicate(text=text, voice=voice, rate=rate)
            await communicate.save(str(file_path))
            print(f"[+] Saved: {file_path}")
        except NoAudioReceived:
            print(f"[!] Edge TTS: No audio received for chunk: {text[:30]}... skipping.")
        except Exception as e:
            print(f"[X] Edge TTS error for chunk {file_path}: {e}")
            raise


async def synthesize_chunk_silero(
    text: str,
    file_path: Path,
    language: str,
    speaker: str,
    sample_rate: int,
    put_accent: bool,
    put_yo: bool,
    device: str,
    model_id: str,
    semaphore: asyncio.Semaphore,
) -> None:
    """Silero TTS chunk -> WAV (saved with soundfile)."""
    # Нормализуем числа и очищаем текст перед синтезом (универсально для всех движков)
    text = normalize_numbers(text, lang=language)
    text = sanitize_for_tts(text)

    if not any(c.isalpha() for c in text):
        print(f"[!] Skipping chunk (no alpha chars): {text[:20]}...")
        return

    # Silero обычно имеет лимит около 1000 символов. 
    # Если текст слишком длинный, это может вызвать ошибку в движке.
    if len(text) > 1000:
        print(f"[!] Warning: Text chunk may be too long for Silero ({len(text)} chars). Recommended < 800-1000.")

    async with semaphore:
        try:
            init_silero(language=language, model_id=model_id, device=device)
        except Exception as e:
            print(f"[X] Silero init failed: {e}")
            raise

        def _run() -> None:
            # Local imports so edge-mode doesn't require these packages
            import numpy as np  # type: ignore
            import soundfile as sf  # type: ignore

            if _silero_apply_tts is None:
                raise RuntimeError("Silero not initialized")

            # Silero обычно имеет лимит около 1000 символов.
            # Если после нормализации текст все еще слишком длинный, 
            # мы разбиваем его на под-чанки и склеиваем аудио в памяти.
            MAX_SILERO_CHARS = 800
            
            try:
                if len(text) <= MAX_SILERO_CHARS:
                    try:
                        audio = _silero_apply_tts(
                            text=text,
                            speaker=speaker,
                            sample_rate=sample_rate,
                            put_accent=put_accent,
                            put_yo=put_yo,
                        )
                    except ValueError as ve:
                        print(f"[!] Silero ValueError (unsupported chars?): {ve}")
                        return

                    # audio can be torch.Tensor or array-like
                    if hasattr(audio, "cpu"):
                        audio_np = audio.cpu().numpy()
                    else:
                        audio_np = np.asarray(audio)
                else:
                    print(f"[~] Silero: Text too long ({len(text)}), splitting into sub-chunks...")
                    sub_chunks = split_text(text, MAX_SILERO_CHARS)
                    audio_list = []
                    for sub in sub_chunks:
                        if not sub.strip():
                            continue
                        try:
                            sub_audio = _silero_apply_tts(
                                text=sub,
                                speaker=speaker,
                                sample_rate=sample_rate,
                                put_accent=put_accent,
                                put_yo=put_yo,
                            )
                        except ValueError as ve:
                            print(f"[!] Silero ValueError on sub-chunk: {ve}")
                            continue
                        
                        if hasattr(sub_audio, "cpu"):
                            audio_list.append(sub_audio.cpu().numpy())
                        else:
                            audio_list.append(np.asarray(sub_audio))
                    
                    if not audio_list:
                        print("[!] No audio generated for any sub-chunks. Skipping.")
                        return
                    audio_np = np.concatenate(audio_list)

                sf.write(str(file_path), audio_np, sample_rate)
            except Exception as e:
                print(f"[X] Silero synthesis error: {e}")
                raise

        try:
            await asyncio.to_thread(_run)
            print(f"[+] Saved: {file_path}")
        except Exception as e:
            print(f"[X] Failed to synthesize chunk {file_path}: {e}")
            raise


def convert_fb2_to_txt(input_file: Path) -> Path:
    """
    Convert FB2 -> TXT internally using extract_fb2_text.
    Returns path to produced .txt.
    """
    print(f"[=] FB2 detected: {input_file.name}")
    print("[=] Extracting text...")

    txt_content = extract_fb2_text(input_file)
    txt_file = input_file.with_suffix(".txt")
    txt_file.write_text(txt_content, encoding="utf-8")

    print(f"[!] Done. Saved as: {txt_file.name}")
    return txt_file


def split_text(text: str, chunk_size: int) -> list[str]:
    """Smart split that tries to break at paragraphs, then sentences, then spaces."""
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    while text:
        if len(text) <= chunk_size:
            chunks.append(text)
            break

        # 1. Try to split by paragraph (\n\n)
        lookback = int(chunk_size * 0.3)  # Don't look back too far
        split_idx = text.rfind("\n\n", chunk_size - lookback, chunk_size)
        if split_idx != -1:
            chunks.append(text[:split_idx].strip())
            text = text[split_idx:].strip()
            continue

        # 2. Try to split by newline (\n)
        split_idx = text.rfind("\n", chunk_size - lookback, chunk_size)
        if split_idx != -1:
            chunks.append(text[:split_idx].strip())
            text = text[split_idx:].strip()
            continue

        # 3. Try to split by sentence (. ! ?)
        # Using regex to find last sentence end within range
        chunk_slice = text[:chunk_size]
        # Look for punctuation followed by space or end of slice
        match = list(re.finditer(r"[.!?]\s", chunk_slice))
        if match:
            # Get last match position
            last_pos = match[-1].end()
            if last_pos > chunk_size - lookback:
                chunks.append(text[:last_pos].strip())
                text = text[last_pos:].strip()
                continue

        # 4. Try to split by space
        split_idx = text.rfind(" ", chunk_size - lookback, chunk_size)
        if split_idx != -1:
            chunks.append(text[:split_idx].strip())
            text = text[split_idx:].strip()
            continue

        # 5. Worst case: hard cut
        chunks.append(text[:chunk_size].strip())
        text = text[chunk_size:].strip()

    return [c for c in chunks if c.strip()]


def ensure_ffmpeg(ffmpeg_path: str) -> str:
    """Return ffmpeg binary path if valid, otherwise raise."""
    p = Path(ffmpeg_path)
    if p.is_file():
        return ffmpeg_path
    # If it's not a file, assume it's in PATH
    return ffmpeg_path


def _run_ffmpeg_concat(ffmpeg_bin: str, list_file: Path, output_file: Path, loglevel: str) -> None:
    subprocess.run(
        [
            ffmpeg_bin,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_file),
            "-c",
            "copy",
            "-loglevel",
            loglevel,
            str(output_file),
        ],
        check=True,
        capture_output=True,
    )


def _run_ffmpeg_convert(ffmpeg_bin: str, input_audio: Path, output_mp3: Path, qscale: str, loglevel: str) -> None:
    subprocess.run(
        [
            ffmpeg_bin,
            "-y",
            "-i",
            str(input_audio),
            "-codec:a",
            "libmp3lame",
            "-q:a",
            qscale,
            "-loglevel",
            loglevel,
            str(output_mp3),
        ],
        check=True,
        capture_output=True,
    )


async def merge_audio_chunks(
    ffmpeg_path: str,
    list_file: Path,
    output_file: Path,
    loglevel: str = "error",
) -> None:
    """FFmpeg concat demuxer merge."""
    ffmpeg_bin = ensure_ffmpeg(ffmpeg_path)
    await asyncio.to_thread(_run_ffmpeg_concat, ffmpeg_bin, list_file, output_file, loglevel)


async def convert_to_mp3(
    ffmpeg_path: str,
    input_audio: Path,
    output_mp3: Path,
    qscale: str = "2",
    loglevel: str = "error",
) -> None:
    """Convert any input audio to MP3 (libmp3lame)."""
    ffmpeg_bin = ensure_ffmpeg(ffmpeg_path)
    await asyncio.to_thread(_run_ffmpeg_convert, ffmpeg_bin, input_audio, output_mp3, qscale, loglevel)


async def process_single_file(
    input_file: Path,
    args: argparse.Namespace,
) -> None:
    """
    Full pipeline for one file:
      - optional FB2->TXT
      - read text
      - split to chunks
      - synthesize chunks (engine-specific)
      - optional merge
      - optional final mp3 (silero only)
    """
    original_file = input_file

    # FB2 -> TXT
    if input_file.suffix.lower() == ".fb2":
        try:
            input_file = convert_fb2_to_txt(input_file)
        except Exception as e:
            print(f"[X] Skip {original_file.name}: FB2 convert error: {e}")
            return

    if input_file.suffix.lower() != ".txt":
        print(f"[!] Skip {input_file.name}: unsupported extension {input_file.suffix}")
        return

    try:
        text = input_file.read_text(encoding="utf-8")
    except Exception as e:
        print(f"[X] Read failed {input_file}: {e}")
        return

    if not text.strip():
        print(f"[!] Empty file, skip: {input_file}")
        return

    # Output dirs
    output_name = input_file.stem
    output_path = Path(args.output_dir) / output_name
    output_path.mkdir(parents=True, exist_ok=True)

    chunks = split_text(text, args.chunk_size)
    print(f"\n[=] File: {original_file.name} -> {len(chunks)} chunks")

    # Engine-specific extension and concurrency
    ext = "mp3" if args.engine == "edge" else "wav"
    max_tasks = getattr(args, "max_concurrent_tasks", None)
    if max_tasks is None:
        max_tasks = 40 if args.engine == "edge" else (os.cpu_count() or 2)

    chunk_semaphore = asyncio.Semaphore(max_tasks)

    tasks: list[asyncio.Task] = []
    existing_count: int = 0

    for i, chunk in enumerate(chunks):
        chunk_file = output_path / f"chunk_{i:06}.{ext}"

        if getattr(args, "skip_chunks", False) and chunk_file.exists():
            existing_count += 1
            continue

        if args.engine == "edge":
            tasks.append(
                asyncio.create_task(
                    synthesize_chunk_edge(
                        text=chunk,
                        file_path=chunk_file,
                        voice=args.voice,
                        rate=args.rate,
                        semaphore=chunk_semaphore,
                    )
                )
            )
        else:
            tasks.append(
                asyncio.create_task(
                    synthesize_chunk_silero(
                        text=chunk,
                        file_path=chunk_file,
                        language=args.silero_language,
                        speaker=args.speaker,
                        sample_rate=args.sample_rate,
                        put_accent=args.put_accent,
                        put_yo=args.put_yo,
                        device=args.device,
                        model_id=args.silero_model_id,
                        semaphore=chunk_semaphore,
                    )
                )
            )

    if existing_count > 0:
        print(f"[~] Skipped existing chunks: {existing_count}")

    if tasks:
        await asyncio.gather(*tasks)

    if getattr(args, "skip_merge", False):
        print("[~] Merge disabled (--skip-merge)")
        return

    # Prepare list.txt
    print("[=] Preparing list.txt for ffmpeg ...")
    list_file = output_path / "list.txt"
    with list_file.open("w", encoding="utf-8") as f:
        for i in range(len(chunks)):
            part_path = output_path / f"chunk_{i:06}.{ext}"
            f.write(f"file '{part_path.name}'\n")

    merged_file = output_path / f"full_{output_name}.{ext}"
    print("[=] Merging via ffmpeg concat ...")
    try:
        await merge_audio_chunks(
            ffmpeg_path=args.ffmpeg,
            list_file=list_file,
            output_file=merged_file,
            loglevel="error",
        )
    except subprocess.CalledProcessError as e:
        print("[X] ffmpeg merge failed.")
        # Show stderr if available
        if e.stderr:
            try:
                print(e.stderr.decode("utf-8", errors="replace"))
            except Exception:
                print(e.stderr)
        raise

    # Optional final mp3 for Silero
    if args.engine == "silero" and getattr(args, "final_mp3", False):
        mp3_file = output_path / f"full_{output_name}.mp3"
        print("[=] Converting merged WAV -> MP3 ...")
        await convert_to_mp3(
            ffmpeg_path=args.ffmpeg,
            input_audio=merged_file,
            output_mp3=mp3_file,
            qscale=args.mp3_quality,
            loglevel="error",
        )
        print(f"[!] Done: {mp3_file}")
    else:
        print(f"[!] Done: {merged_file}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="FB2/TXT -> TTS chunks -> merge (edge-tts or silero)")
    p.add_argument("path", nargs="?", default="sample.txt", help="Input file (.txt/.fb2) or directory")

    # Behavior flags
    p.add_argument("--skip-chunks", action="store_true", help="Do not re-generate existing chunk files")
    p.add_argument("--skip-merge", action="store_true", help="Do not merge chunks")
    p.add_argument("--output-dir", default="output", help="Output directory root")

    # Chunking and concurrency
    p.add_argument("--chunk-size", type=int, default=10000, help="Characters per chunk")
    p.add_argument("--max-concurrent-tasks", type=int, default=None, help="Chunk synthesis concurrency")

    # Engine selection
    p.add_argument("--engine", choices=["edge", "silero"], default="edge", help="TTS engine")

    # ffmpeg
    p.add_argument("--ffmpeg", default="ffmpeg", help="Path to ffmpeg binary or 'ffmpeg' in PATH")

    # Edge settings
    p.add_argument("--voice", default="ru-RU-SvetlanaNeural", help="Edge voice name")
    p.add_argument("--rate", default="+18%", help="Edge rate, e.g. '+10%%' or '-10%%'")

    # Silero settings
    p.add_argument("--silero-language", default="ru", help="Silero language code, e.g. ru")
    p.add_argument("--silero-model-id", default="v5_ru", help="Silero model version, e.g. v5_ru or v3_1_ru")
    p.add_argument("--speaker", default="baya", help="Silero speaker, e.g. aidar/baya/kseniya/xenia/eugene")
    p.add_argument("--sample-rate", type=int, default=48000, help="Silero sample rate: 8000/24000/48000")
    p.add_argument("--put-accent", action="store_true", default=True, help="Silero: try to add accents")
    p.add_argument("--no-put-accent", action="store_false", dest="put_accent", help="Silero: disable accents")
    p.add_argument("--put-yo", action="store_true", default=True, help="Silero: replace 'е' with 'ё' where appropriate")
    p.add_argument("--no-put-yo", action="store_false", dest="put_yo", help="Silero: disable 'yo' replacement")
    p.add_argument("--device", default="cpu", help="Silero device: cpu or cuda")

    # Silero output option
    p.add_argument("--final-mp3", action="store_true", help="For silero: also produce final MP3 from merged WAV")
    p.add_argument("--mp3-quality", default="2", help="FFmpeg LAME VBR qscale for final mp3 (lower is better)")

    # File-level parallelism (directories)
    p.add_argument("--max-concurrent-files", type=int, default=3, help="Max files processed concurrently in dir mode")

    return p


async def main() -> None:
    _args = build_arg_parser().parse_args()
    args: Any = cast(Any, _args)

    # Concurrency defaults:
    # - If user did not set, choose based on engine.
    if getattr(args, "max_concurrent_tasks", None) is None:
        args.max_concurrent_tasks = 40 if args.engine == "edge" else 2

    input_path = Path(args.path)
    if not input_path.exists():
        print(f"[X] Path does not exist: {input_path}")
        sys.exit(1)

    # Directory mode
    if input_path.is_dir():
        print(f"[=] Directory: {input_path}")
        files = sorted(
            f
            for f in input_path.iterdir()
            if f.is_file() and f.suffix.lower() in {".txt", ".fb2"}
        )

        if not files:
            print("[X] No *.txt or *.fb2 files found")
            return

        print(f"[=] Found {len(files)} files:")
        for f in files:
            print(f"    - {f.name}")

        file_semaphore = asyncio.Semaphore(args.max_concurrent_files)

        async def process_with_semaphore(f: Path) -> None:
            async with file_semaphore:
                print(f"\n[=] === Processing: {f.name} ===")
                await process_single_file(f, args)

        await asyncio.gather(*[process_with_semaphore(f) for f in files])
        return

    # Single file mode
    print(f"[=] Single file: {input_path.name}")
    await process_single_file(input_path, args)


if __name__ == "__main__":
    asyncio.run(main())
