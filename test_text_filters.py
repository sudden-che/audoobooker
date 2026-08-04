import asyncio
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


def _install_tg_stubs() -> None:
    if "dotenv" not in sys.modules:
        dotenv = types.ModuleType("dotenv")
        dotenv.load_dotenv = lambda: None
        sys.modules["dotenv"] = dotenv

    if "telegram" not in sys.modules:
        telegram = types.ModuleType("telegram")
        telegram.Update = type("Update", (), {})
        telegram.InlineKeyboardButton = type("InlineKeyboardButton", (), {})
        telegram.InlineKeyboardMarkup = type("InlineKeyboardMarkup", (), {})
        telegram.error = types.SimpleNamespace(NetworkError=Exception)
        sys.modules["telegram"] = telegram

    if "telegram.ext" not in sys.modules:
        telegram_ext = types.ModuleType("telegram.ext")
        telegram_ext.Application = type("Application", (), {})
        telegram_ext.CommandHandler = type("CommandHandler", (), {})
        telegram_ext.MessageHandler = type("MessageHandler", (), {})
        telegram_ext.CallbackQueryHandler = type("CallbackQueryHandler", (), {})
        telegram_ext.PicklePersistence = type("PicklePersistence", (), {})
        telegram_ext.filters = types.SimpleNamespace()
        telegram_ext.ContextTypes = types.SimpleNamespace(DEFAULT_TYPE=object)
        sys.modules["telegram.ext"] = telegram_ext


_install_tg_stubs()

from audiobooker import (  # noqa: E402
    apply_audiobook_best_practices,
    build_output_basename,
    sanitize_for_tts,
)
from tg_audiobooker import (  # noqa: E402
    EDGE_VOICES,
    clean_tg_post,
    extract_telegraph_page_text,
    extract_telegraph_url,
    generate_audio,
    get_text_preview,
    is_slash_command,
    normalize_user_settings,
)


class TextFilterTests(unittest.TestCase):
    def test_sanitize_for_tts_keeps_english_words(self) -> None:
        text = "Google Gemini 2.5 Pro и OpenAI GPT-4.1"
        self.assertEqual(sanitize_for_tts(text), text)

    def test_sanitize_for_tts_strips_angle_brackets(self) -> None:
        text = "Цена < 10 > 5"
        self.assertEqual(sanitize_for_tts(text), "Цена 10 5")

    def test_sanitize_for_tts_strips_bullet_emoji(self) -> None:
        text = "Важно ▪️ проверить"
        self.assertEqual(sanitize_for_tts(text), "Важно проверить")

    def test_clean_tg_post_keeps_regular_google_sentence(self) -> None:
        text = "Сегодня мы в Google обсуждаем Gemini. Подробности позже."
        self.assertEqual(clean_tg_post(text), text)

    def test_clean_tg_post_preserves_pause_tag(self) -> None:
        text = "До паузы. [[PAUSE: 5000]] После паузы."
        self.assertEqual(clean_tg_post(text), text)

    def test_extract_telegraph_url_strips_trailing_punctuation(self) -> None:
        text = "Смотри статью: https://telegra.ph/Test-03-25)."
        self.assertEqual(
            extract_telegraph_url(text),
            "https://telegra.ph/Test-03-25",
        )

    def test_extract_telegraph_page_text_reads_title_and_body(self) -> None:
        html = """
        <html>
          <body>
            <main class="tl_article">
              <header><h1>Заголовок статьи</h1></header>
              <article>
                <p>Первый абзац.</p>
                <p>Второй абзац с <a href="https://example.com">ссылкой</a>.</p>
                <blockquote>Цитата.</blockquote>
              </article>
            </main>
          </body>
        </html>
        """
        self.assertEqual(
            extract_telegraph_page_text(html),
            "Заголовок статьи\n\nПервый абзац.\n\nВторой абзац с ссылкой.\n\nЦитата.",
        )

    def test_apply_audiobook_best_practices_verbalizes_common_symbols(self) -> None:
        text = "Глава 5\nЦена 25% и №7, температура 21°C, 3/4."
        self.assertEqual(
            apply_audiobook_best_practices(text),
            "Глава 5.\nЦена 25 процентов и номер 7, температура 21 градусов Цельсия, 3 дробь 4.",
        )

    def test_apply_audiobook_best_practices_formats_dialogue_pause(self) -> None:
        text = "- Привет.\n- Пока."
        self.assertEqual(
            apply_audiobook_best_practices(text),
            "— Привет.\n— Пока.",
        )

    def test_build_output_basename_transliterates_cyrillic(self) -> None:
        basename = build_output_basename("Преступление и наказание.fb2")
        self.assertEqual(basename, "prestuplenie-i-nakazanie")

    def test_build_output_basename_truncates_long_names(self) -> None:
        basename = build_output_basename(
            "Очень длинное название файла для проверки ограничения длины.txt",
            max_length=24,
        )
        self.assertTrue(basename.isascii())
        self.assertLessEqual(len(basename), 24)
        self.assertRegex(basename, r"^[a-z0-9-]+$")

    def test_clean_tg_post_strips_only_subscription_fragment(self) -> None:
        text = "Подпишитесь на наш Telegram. Google Gemini выпустили обновление."
        self.assertEqual(
            clean_tg_post(text),
            "Google Gemini выпустили обновление.",
        )

    def test_clean_tg_post_strips_source_metadata_line(self) -> None:
        text = "Источник: 6\nGoogle Gemini выпустили обновление."
        self.assertEqual(
            clean_tg_post(text),
            "Google Gemini выпустили обновление.",
        )

    def test_clean_tg_post_strips_source_prefix_but_keeps_sentence(self) -> None:
        text = "Источник: 6. Google Gemini выпустили обновление."
        self.assertEqual(
            clean_tg_post(text),
            "Google Gemini выпустили обновление.",
        )

    def test_get_text_preview_uses_first_sentence_after_source_line(self) -> None:
        text = "ТАСС\nGoogle Gemini выпустили обновление. Подробности позже."
        self.assertEqual(
            get_text_preview(text),
            "Google Gemini выпустили обновление.",
        )

    def test_get_text_preview_skips_numeric_sentence_marker(self) -> None:
        text = "1. Google Gemini выпустили обновление. Подробности позже."
        self.assertEqual(
            get_text_preview(text),
            "Google Gemini выпустили обновление.",
        )

    def test_get_text_preview_skips_numeric_marker_line(self) -> None:
        text = "6.\nНовый законопроект внесли в Госдуму. Детали позже."
        self.assertEqual(
            get_text_preview(text),
            "Новый законопроект внесли в Госдуму.",
        )

    def test_edge_random_voices_use_supported_set(self) -> None:
        self.assertEqual(
            EDGE_VOICES,
            ["ru-RU-SvetlanaNeural", "ru-RU-DmitryNeural"],
        )

    def test_legacy_local_model_settings_are_removed(self) -> None:
        settings = {
            "engine": "legacy-local",
            "silero_model_id": "legacy-model",
            "silero_speaker": "legacy-speaker",
            "device": "cpu",
            "edge_voice": "ru-RU-DmitryNeural",
        }

        normalized = normalize_user_settings(settings)

        self.assertIs(normalized, settings)
        self.assertNotIn("engine", normalized)
        self.assertNotIn("silero_model_id", normalized)
        self.assertNotIn("silero_speaker", normalized)
        self.assertNotIn("device", normalized)
        self.assertEqual(normalized["edge_voice"], "ru-RU-DmitryNeural")

    def test_telegram_audio_pipeline_keeps_pause_in_order(self) -> None:
        speech_calls: list[str] = []
        pause_calls: list[int] = []
        routed_chunks: dict[str, tuple[str, str | int]] = {}

        async def fake_speech(
            text: str,
            file_path: Path,
            voice: str,
            rate: str,
            semaphore,
        ) -> None:
            speech_calls.append(text)
            routed_chunks[file_path.name] = ("speech", text)
            file_path.write_bytes(b"ID3" + b"\x00" * 32)

        async def fake_pause(
            duration_ms: int,
            file_path: Path,
            ffmpeg_path: str,
            semaphore,
            loglevel: str = "error",
        ) -> None:
            pause_calls.append(duration_ms)
            routed_chunks[file_path.name] = ("pause", duration_ms)
            file_path.write_bytes(b"ID3" + b"\x00" * 32)

        with (
            tempfile.TemporaryDirectory() as tmp_dir,
            patch(
                "tg_audiobooker.synthesize_chunk_edge",
                new=fake_speech,
            ),
            patch(
                "tg_audiobooker.synthesize_silence_mp3",
                new=fake_pause,
            ),
        ):
            result = asyncio.run(
                generate_audio(
                    "До паузы. [[PAUSE: 5000]] После паузы.",
                    Path(tmp_dir),
                    settings={
                        "chunk_size": 100,
                        "max_concurrent_tasks": 1,
                        "merge_chunks": False,
                        "edge_voice": "ru-RU-SvetlanaNeural",
                        "edge_speed": "+0%",
                        "ffmpeg_path": "ffmpeg",
                        "random": False,
                    },
                )
            )

        self.assertEqual(speech_calls, ["До паузы.", "После паузы."])
        self.assertEqual(pause_calls, [5000])
        self.assertEqual(len(result), 3)
        self.assertEqual(
            [routed_chunks[name] for name in sorted(routed_chunks)],
            [
                ("speech", "До паузы."),
                ("pause", 5000),
                ("speech", "После паузы."),
            ],
        )

    def test_telegram_rejects_malformed_pause_before_synthesis(self) -> None:
        with (
            tempfile.TemporaryDirectory() as tmp_dir,
            patch("tg_audiobooker.synthesize_chunk_edge") as speech_mock,
            patch("tg_audiobooker.synthesize_silence_mp3") as pause_mock,
        ):
            with self.assertRaisesRegex(ValueError, "PAUSE"):
                asyncio.run(
                    generate_audio(
                        "До. [[PAUSE::5000]] После.",
                        Path(tmp_dir),
                        settings={
                            "chunk_size": 100,
                            "max_concurrent_tasks": 1,
                            "merge_chunks": False,
                            "edge_voice": "ru-RU-SvetlanaNeural",
                            "edge_speed": "+0%",
                            "ffmpeg_path": "ffmpeg",
                            "random": False,
                        },
                    )
                )

        speech_mock.assert_not_called()
        pause_mock.assert_not_called()

    def test_is_slash_command_detects_telegram_commands(self) -> None:
        self.assertTrue(is_slash_command("/start"))
        self.assertTrue(is_slash_command("   /help extra"))
        self.assertTrue(is_slash_command("/settings@my_bot"))

    def test_is_slash_command_ignores_regular_text(self) -> None:
        self.assertFalse(is_slash_command("/"))
        self.assertFalse(is_slash_command("text /start"))
        self.assertFalse(is_slash_command("https://example.com/path"))


if __name__ == "__main__":
    unittest.main()
