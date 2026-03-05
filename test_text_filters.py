import sys
import types
import unittest


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

from audiobooker import sanitize_for_tts, transliterate_latin_for_ru_tts
from tg_audiobooker import (
    EDGE_VOICES,
    choose_random_silero_model_id,
    clean_tg_post,
    get_silero_model_major,
    get_text_preview,
)


class TextFilterTests(unittest.TestCase):
    def test_sanitize_for_tts_keeps_english_words(self) -> None:
        text = "Google Gemini 2.5 Pro и OpenAI GPT-4.1"
        self.assertEqual(sanitize_for_tts(text), text)

    def test_clean_tg_post_keeps_regular_google_sentence(self) -> None:
        text = "Сегодня мы в Google обсуждаем Gemini. Подробности позже."
        self.assertEqual(clean_tg_post(text), text)

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

    def test_transliterate_latin_for_ru_tts_converts_latin_tokens(self) -> None:
        text = "Google и OpenAI представили GPT-4.1"
        converted = transliterate_latin_for_ru_tts(text)
        self.assertNotIn("Google", converted)
        self.assertNotIn("OpenAI", converted)
        self.assertRegex(converted, r"[А-Яа-яЁё]")

    def test_random_silero_models_are_v5_or_newer(self) -> None:
        picked_models = {choose_random_silero_model_id() for _ in range(20)}
        self.assertTrue(picked_models)
        self.assertTrue(all(get_silero_model_major(model_id) >= 5 for model_id in picked_models))

    def test_edge_random_voices_use_supported_set(self) -> None:
        self.assertEqual(
            EDGE_VOICES,
            ["ru-RU-SvetlanaNeural", "ru-RU-DmitryNeural"],
        )


if __name__ == "__main__":
    unittest.main()
