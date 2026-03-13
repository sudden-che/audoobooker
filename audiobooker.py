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
import hashlib
import os
import re
import shutil
import subprocess
import sys
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, Tuple, Callable, Any, Iterable, cast

# -----------------------------
# Silero lazy globals
# -----------------------------
_silero_apply_tts: Optional[Callable] = None
_silero_inited: bool = False
LATIN_TOKEN_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9]*(?:[-'][A-Za-z0-9]+)*\b")
CYRILLIC_CHAR_RE = re.compile(r"[А-Яа-яЁё]")
ZERO_WIDTH_CHAR_RE = re.compile(r"[\u200b\ufeff\u2060]")
LEADING_DIALOGUE_RE = re.compile(r"(?m)^\s*[-–—]\s*")
NON_SPEECH_NOTE_RE = re.compile(
    r"\[(?:иллюстрац(?:ия|ии)|рис\.?|картинка|image|illustration|footnote|сноска)[^\]]*\]",
    re.IGNORECASE,
)
AUDIOBOOK_HEADING_RE = re.compile(
    r"(?ix)^"
    r"(?:"
    r"(?:глава|часть|книга|том|пролог|эпилог|послесловие|сцена|акт|chapter|part|book|volume|prologue|epilogue)\b.*"
    r"|(?:[IVXLCDM]+|\d+)[.)](?:\s+\S.*)?"
    r")$"
)
FRACTION_RE = re.compile(r"(?<!\w)(\d+)\s*/\s*(\d+)(?!\w)")
DEGREE_RE = re.compile(r"(?<=\d)\s*°\s*([CFcf])\b")
INLINE_AMPERSAND_RE = re.compile(r"(?<=\w)\s*&\s*(?=\w)")
INLINE_PLUS_RE = re.compile(r"(?<=\d)\s*\+\s*(?=\d)")
INLINE_EQUALS_RE = re.compile(r"(?<=\d)\s*=\s*(?=\d)")
EDGE_TTS_MAX_RETRIES = max(0, int(os.environ.get("EDGE_TTS_MAX_RETRIES", "3")))
EDGE_TTS_RETRY_BASE_DELAY = max(
    0.0,
    float(os.environ.get("EDGE_TTS_RETRY_BASE_DELAY", "1.5")),
)
EDGE_TTS_RETRY_MAX_DELAY = max(
    0.0,
    float(os.environ.get("EDGE_TTS_RETRY_MAX_DELAY", "12.0")),
)
OUTPUT_BASENAME_MAX_LENGTH = max(
    16,
    int(os.environ.get("OUTPUT_BASENAME_MAX_LENGTH", "48")),
)


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


def build_output_basename(
    source_name: str | Path,
    *,
    max_length: int = OUTPUT_BASENAME_MAX_LENGTH,
) -> str:
    """
    Строит безопасное короткое ASCII-имя для выходных файлов.

    - берёт stem исходного файла;
    - транслитерирует кириллицу в латиницу;
    - чистит служебные символы;
    - ограничивает длину и добавляет короткий hash при усечении.
    """
    source_text = str(source_name).strip()
    stem = Path(source_text).stem.strip() or Path(source_text).name.strip() or "book"

    transliterated = stem
    if CYRILLIC_CHAR_RE.search(transliterated):
        try:
            from transliterate import translit  # type: ignore

            transliterated = translit(transliterated, "ru", reversed=True)
        except Exception:
            pass

    normalized = unicodedata.normalize("NFKD", transliterated)
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    safe_name = ascii_only.lower()
    safe_name = re.sub(r"[\"'`]+", "", safe_name)
    safe_name = re.sub(r"[^a-z0-9]+", "-", safe_name)
    safe_name = re.sub(r"-{2,}", "-", safe_name).strip("-")
    if not safe_name:
        safe_name = "book"

    if len(safe_name) <= max_length:
        return safe_name

    digest = hashlib.sha1(stem.encode("utf-8")).hexdigest()[:8]
    head_length = max(8, max_length - len(digest) - 1)
    head = safe_name[:head_length].rstrip("-")
    return f"{head}-{digest}" if head else f"book-{digest}"


def verbalize_common_symbols(text: str, lang: str = "ru") -> str:
    """Расшифровывает частые символы в более естественную для TTS форму."""
    text = NON_SPEECH_NOTE_RE.sub(" ", text)
    text = re.sub(r"№\s*(\d+)", r"номер \1", text)
    text = re.sub(r"§\s*(\d+)", r"параграф \1", text)
    text = re.sub(r"(?<=\d)\s*%", " процентов", text)
    text = re.sub(r"(?<=\d)\s*‰", " промилле", text)
    text = re.sub(r"(?<=\d)\s*[€]", " евро", text)
    text = re.sub(r"(?<=\d)\s*[$]", " долларов", text)
    text = re.sub(r"(?<=\d)\s*[£]", " фунтов", text)
    text = re.sub(r"(?<=\d)\s*[₽]", " рублей", text)
    text = FRACTION_RE.sub(r"\1 дробь \2", text)
    text = DEGREE_RE.sub(
        lambda match: (
            " градусов Цельсия"
            if match.group(1).lower() == "c"
            else " градусов Фаренгейта"
        ),
        text,
    )
    text = re.sub(r"(?<=\d)\s*°(?!\w)", " градусов", text)
    text = INLINE_AMPERSAND_RE.sub(" и ", text)
    text = INLINE_PLUS_RE.sub(" плюс ", text)
    text = INLINE_EQUALS_RE.sub(" равно ", text)
    if lang.lower().startswith("ru"):
        text = re.sub(r"\bи/или\b", "и или", text, flags=re.IGNORECASE)
    return text


def apply_audiobook_best_practices(text: str, lang: str = "ru") -> str:
    """
    Подготавливает текст к синтезу аудиокниги:
      - сохраняет абзацы для естественных пауз;
      - нормализует символы и реплики;
      - добавляет финальную пунктуацию заголовкам глав.
    """
    text = clean_text(text)
    text = ZERO_WIDTH_CHAR_RE.sub("", text)
    text = text.replace("…", "...")
    text = LEADING_DIALOGUE_RE.sub("— ", text)
    text = verbalize_common_symbols(text, lang=lang)
    text = re.sub(r"\.\.\.+", "...", text)
    text = re.sub(r"([!?])\1{2,}", r"\1", text)
    text = re.sub(r"([,:;])\1{1,}", r"\1", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([,;:!?])(?![\s\n\"')\]»])", r"\1 ", text)
    text = re.sub(r"([.])(?![\s\n\"')\]»])", r"\1 ", text)

    processed_lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            if processed_lines and processed_lines[-1] != "":
                processed_lines.append("")
            continue
        if AUDIOBOOK_HEADING_RE.match(line) and line[-1] not in ".!?":
            line = f"{line}."
        processed_lines.append(line)

    text = "\n".join(processed_lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()


def sanitize_for_tts(text: str) -> str:
    """
    Minimal text normalization before TTS.

    Keeps Latin words and most printable punctuation intact so we don't
    accidentally delete meaningful content up front.
    """
    text = text.replace("\xa0", " ")
    text = ZERO_WIDTH_CHAR_RE.sub("", text)
    text = text.replace("«", '"').replace("»", '"').replace("“", '"').replace("”", '"')
    text = "".join(c for c in text if c.isprintable() or c in "\n\t")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n+ *", "\n", text)
    return text.strip()


def sanitize_for_tts_fallback(text: str) -> str:
    """
    Stricter sanitizer for engines that reject some symbols.

    Latin words, Cyrillic, digits, and common punctuation are preserved.
    """
    text = sanitize_for_tts(text)
    allowed_chars_pattern = re.compile(
        r"[^а-яА-ЯёЁa-zA-Z0-9\s.,!?\-—–…:;'\"()%@#&$*+=<>/\[\]{}\\]"
    )
    text = allowed_chars_pattern.sub(" ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def transliterate_latin_for_ru_tts(text: str) -> str:
    """
    Converts Latin words to Cyrillic approximations for Russian TTS models.

    Silero RU models may skip or badly pronounce raw Latin fragments. This helper
    transliterates only standalone Latin tokens and leaves the rest untouched.
    """
    if not LATIN_TOKEN_RE.search(text):
        return text

    try:
        from transliterate import translit  # type: ignore
    except Exception:
        return text

    def _replace(match: re.Match[str]) -> str:
        token = match.group(0)
        try:
            converted = translit(token, "ru")
        except Exception:
            return token
        return converted if CYRILLIC_CHAR_RE.search(converted) else token

    return LATIN_TOKEN_RE.sub(_replace, text)


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


def has_tts_content(text: str) -> bool:
    """Проверяет, остались ли после очистки буквы или цифры для озвучки."""
    return any(char.isalnum() for char in text)


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
    temp_file = file_path.with_name(f"{file_path.name}.part")

    _cleanup_file(temp_file)
    _cleanup_file(file_path)

    if not has_tts_content(text):
        print(f"[!] Skipping chunk (no speech content): {text[:20]}...")
        return

    try:
        from edge_tts import Communicate  # type: ignore
        from edge_tts.exceptions import NoAudioReceived  # type: ignore
    except Exception as e:
        _require("edge-tts", f"Install: pip install edge-tts. Original error: {e}")

    total_attempts = EDGE_TTS_MAX_RETRIES + 1
    for attempt_index in range(total_attempts):
        _cleanup_file(temp_file)
        try:
            async with semaphore:
                communicate = Communicate(text=text, voice=voice, rate=rate)
                await communicate.save(str(temp_file))
            if not is_valid_audio_file(temp_file, expected_ext="mp3"):
                raise RuntimeError(
                    f"Edge TTS produced an invalid MP3 chunk: {file_path.name}"
                )

            temp_file.replace(file_path)
            print(f"[+] Saved: {file_path}")
            return
        except NoAudioReceived:
            _cleanup_file(temp_file)
            print(f"[!] Edge TTS: No audio received for chunk: {text[:30]}... skipping.")
            return
        except Exception as e:
            _cleanup_file(temp_file)
            if attempt_index < EDGE_TTS_MAX_RETRIES and is_retryable_edge_error(e):
                delay_seconds = min(
                    EDGE_TTS_RETRY_BASE_DELAY * (2**attempt_index),
                    EDGE_TTS_RETRY_MAX_DELAY,
                )
                print(
                    f"[!] Edge TTS transient error for chunk {file_path.name} "
                    f"(attempt {attempt_index + 1}/{total_attempts}): {e}. "
                    f"Retry in {delay_seconds:.1f}s."
                )
                await asyncio.sleep(delay_seconds)
                continue
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
    if language.lower().startswith("ru"):
        text = transliterate_latin_for_ru_tts(text)
    temp_file = file_path.with_name(f"{file_path.name}.part")

    _cleanup_file(temp_file)
    _cleanup_file(file_path)

    if not has_tts_content(text):
        print(f"[!] Skipping chunk (no speech content): {text[:20]}...")
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

            def _apply_tts_with_fallback(chunk_text: str):
                try:
                    return _silero_apply_tts(
                        text=chunk_text,
                        speaker=speaker,
                        sample_rate=sample_rate,
                        put_accent=put_accent,
                        put_yo=put_yo,
                    )
                except ValueError as ve:
                    fallback_text = sanitize_for_tts_fallback(chunk_text)
                    if not fallback_text or fallback_text == chunk_text:
                        print(f"[!] Silero ValueError (unsupported chars?): {ve}")
                        return None

                    print(
                        "[!] Silero rejected some symbols, retrying with stricter sanitization."
                    )
                    try:
                        return _silero_apply_tts(
                            text=fallback_text,
                            speaker=speaker,
                            sample_rate=sample_rate,
                            put_accent=put_accent,
                            put_yo=put_yo,
                        )
                    except ValueError as strict_ve:
                        print(
                            f"[!] Silero ValueError after stricter sanitization: {strict_ve}"
                        )
                        return None

            # Silero обычно имеет лимит около 1000 символов.
            # Если после нормализации текст все еще слишком длинный, 
            # мы разбиваем его на под-чанки и склеиваем аудио в памяти.
            MAX_SILERO_CHARS = 800
            
            try:
                if len(text) <= MAX_SILERO_CHARS:
                    audio = _apply_tts_with_fallback(text)
                    if audio is None:
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
                        sub_audio = _apply_tts_with_fallback(sub)
                        if sub_audio is None:
                            continue
                        
                        if hasattr(sub_audio, "cpu"):
                            audio_list.append(sub_audio.cpu().numpy())
                        else:
                            audio_list.append(np.asarray(sub_audio))
                    
                    if not audio_list:
                        print("[!] No audio generated for any sub-chunks. Skipping.")
                        return
                    audio_np = np.concatenate(audio_list)

                sf.write(str(temp_file), audio_np, sample_rate, format="WAV")
            except Exception as e:
                print(f"[X] Silero synthesis error: {e}")
                raise

        try:
            await asyncio.to_thread(_run)
            if not is_valid_audio_file(temp_file, expected_ext="wav"):
                raise RuntimeError(
                    f"Silero produced an invalid WAV chunk: {file_path.name}"
                )

            temp_file.replace(file_path)
            print(f"[+] Saved: {file_path}")
        except Exception as e:
            _cleanup_file(temp_file)
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

        # 3. Try to split by sentence (. ! ? … ; :)
        # Using regex to find last sentence end within range
        chunk_slice = text[:chunk_size]
        # Look for punctuation followed by quotes/brackets and a space or end of slice
        match = list(re.finditer(r"[.!?…;:](?:[\"')\]»]+)?\s", chunk_slice))
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


def _cleanup_file(file_path: Path) -> None:
    try:
        file_path.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        pass


def is_retryable_edge_error(exc: Exception) -> bool:
    """Определяет временные сетевые ошибки Edge TTS, для которых нужен retry."""
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError, ConnectionError)):
        return True

    try:
        import aiohttp  # type: ignore

        if isinstance(exc, aiohttp.ClientError):
            return True
    except Exception:
        pass

    try:
        from websockets.exceptions import WebSocketException  # type: ignore

        if isinstance(exc, WebSocketException):
            return True
    except Exception:
        pass

    if isinstance(exc, OSError):
        return True

    message = str(exc).lower()
    retryable_markers = (
        "cannot connect to host",
        "server disconnected",
        "connection reset",
        "temporarily unavailable",
        "timed out",
        "timeout",
        "clientconnectorerror",
        "connection aborted",
        "broken pipe",
    )
    return any(marker in message for marker in retryable_markers)


def _looks_like_mp3(data: bytes) -> bool:
    if data.startswith(b"ID3"):
        return True

    for i in range(len(data) - 1):
        if data[i] != 0xFF:
            continue
        if data[i + 1] & 0xE0 == 0xE0:
            return True
    return False


def is_valid_audio_file(file_path: Path, expected_ext: str | None = None) -> bool:
    if not file_path.is_file():
        return False

    try:
        if file_path.stat().st_size < 16:
            return False
        with file_path.open("rb") as f:
            header = f.read(4096)
    except OSError:
        return False

    ext = (expected_ext or file_path.suffix.lstrip(".")).lower()
    if ext == "wav":
        return len(header) >= 12 and header[:4] == b"RIFF" and header[8:12] == b"WAVE"
    if ext == "mp3":
        return _looks_like_mp3(header)
    return True


def collect_valid_audio_files(
    paths: Iterable[Path],
    *,
    expected_ext: str | None = None,
    remove_invalid: bool = False,
) -> list[Path]:
    valid_files: list[Path] = []
    for file_path in paths:
        if is_valid_audio_file(file_path, expected_ext=expected_ext):
            valid_files.append(file_path)
            continue

        if remove_invalid and file_path.exists():
            _cleanup_file(file_path)

    return valid_files


def _run_ffmpeg_concat(ffmpeg_bin: str, list_file: Path, output_file: Path, loglevel: str) -> None:
    try:
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
    except subprocess.CalledProcessError as e:
        err_msg = e.stderr.decode(errors="replace") if e.stderr else "No stderr"
        raise RuntimeError(f"FFmpeg concat failed (exit {e.returncode}): {err_msg}") from e


def _run_ffmpeg_convert(ffmpeg_bin: str, input_audio: Path, output_mp3: Path, qscale: str, loglevel: str) -> None:
    try:
        subprocess.run(
            [
                ffmpeg_bin,
                "-y",
                "-i",
                str(input_audio),
                "-codec:a",
                "libmp3lame",
                "-qscale:a",
                qscale,
                "-loglevel",
                loglevel,
                str(output_mp3),
            ],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        err_msg = e.stderr.decode(errors="replace") if e.stderr else "No stderr"
        raise RuntimeError(f"FFmpeg convert failed (exit {e.returncode}): {err_msg}") from e


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

    text = apply_audiobook_best_practices(text, lang="ru")

    if not text.strip():
        print(f"[!] Empty file, skip: {input_file}")
        return

    # Output dirs
    output_name = build_output_basename(original_file.name)
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
        chunk_file = output_path / f"{output_name}_chunk_{i:06}.{ext}"

        if getattr(args, "skip_chunks", False) and is_valid_audio_file(
            chunk_file, expected_ext=ext
        ):
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

    expected_parts = [
        output_path / f"{output_name}_chunk_{i:06}.{ext}"
        for i in range(len(chunks))
    ]
    actual_parts = collect_valid_audio_files(
        expected_parts,
        expected_ext=ext,
        remove_invalid=True,
    )

    if not actual_parts:
        raise FileNotFoundError("No valid audio chunks were created.")

    # Prepare list.txt
    print("[=] Preparing list.txt for ffmpeg ...")
    list_file = output_path / "list.txt"
    with list_file.open("w", encoding="utf-8") as f:
        for part_path in actual_parts:
            f.write(f"file '{part_path.name}'\n")

    merged_file = output_path / f"{output_name}.{ext}"
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
        mp3_file = output_path / f"{output_name}.mp3"
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
