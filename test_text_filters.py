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

from audiobooker import (
    apply_audiobook_best_practices,
    build_output_basename,
    sanitize_for_tts,
    transliterate_latin_for_ru_tts,
)
from tg_audiobooker import (
    EDGE_VOICES,
    choose_random_silero_model_id,
    clean_tg_post,
    get_silero_model_major,
    get_text_preview,
    is_slash_command,
)


class TextFilterTests(unittest.TestCase):
    def test_sanitize_for_tts_keeps_english_words(self) -> None:
        text = "Google Gemini 2.5 Pro и OpenAI GPT-4.1"
        self.assertEqual(sanitize_for_tts(text), text)

    def test_clean_tg_post_keeps_regular_google_sentence(self) -> None:
        text = "Сегодня мы в Google обсуждаем Gemini. Подробности позже."
        self.assertEqual(clean_tg_post(text), text)

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
