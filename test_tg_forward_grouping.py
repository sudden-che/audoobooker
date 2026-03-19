import asyncio
import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch


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

from tg_audiobooker import (
    FORWARDED_BATCHES_KEY,
    build_forward_batch_key,
    build_forward_job_name,
    ensure_forwarded_batches,
    get_user_processing_lock,
    handle_document,
)


def test_forward_batch_key_separates_chat_group_and_thread() -> None:
    base_key = build_forward_batch_key(chat_id=100, user_id=7)
    same_key = build_forward_batch_key(chat_id=100, user_id=7)
    other_chat_key = build_forward_batch_key(chat_id=101, user_id=7)
    media_group_key = build_forward_batch_key(
        chat_id=100,
        user_id=7,
        media_group_id="album-1",
    )
    thread_key = build_forward_batch_key(
        chat_id=100,
        user_id=7,
        message_thread_id=55,
    )

    assert base_key == same_key
    assert base_key != other_chat_key
    assert base_key != media_group_key
    assert base_key != thread_key


def test_forward_job_name_is_stable_per_batch() -> None:
    first_key = build_forward_batch_key(chat_id=100, user_id=7)
    second_key = build_forward_batch_key(
        chat_id=100,
        user_id=7,
        media_group_id="album-2",
    )

    assert build_forward_job_name(first_key) == build_forward_job_name(first_key)
    assert build_forward_job_name(first_key) != build_forward_job_name(second_key)


def test_ensure_forwarded_batches_reuses_user_storage() -> None:
    user_data: dict = {}

    batches = ensure_forwarded_batches(user_data)
    batches["batch-a"] = [("hello", 1, "#tag")]

    assert FORWARDED_BATCHES_KEY in user_data
    assert user_data[FORWARDED_BATCHES_KEY]["batch-a"][0][0] == "hello"
    assert ensure_forwarded_batches(user_data) is batches


def test_user_processing_lock_is_stable_per_user() -> None:
    first_lock = get_user_processing_lock(100)
    second_lock = get_user_processing_lock(100)
    third_lock = get_user_processing_lock(101)

    assert first_lock is second_lock
    assert first_lock is not third_lock
    assert get_user_processing_lock(None) is None


class _DummyJobQueue:
    def __init__(self) -> None:
        self.run_once_calls: list[dict] = []

    def get_jobs_by_name(self, _name: str) -> list:
        return []

    def run_once(self, callback, when, data, name, user_id, chat_id) -> None:
        self.run_once_calls.append(
            {
                "callback": callback,
                "when": when,
                "data": data,
                "name": name,
                "user_id": user_id,
                "chat_id": chat_id,
            }
        )


def test_forwarded_document_is_buffered_for_collector() -> None:
    job_queue = _DummyJobQueue()
    reply_text = AsyncMock()

    update = SimpleNamespace(
        message=SimpleNamespace(
            document=SimpleNamespace(file_name="part1.txt", file_id="doc-1"),
            forward_origin=SimpleNamespace(
                sender_user=SimpleNamespace(id=321, username="source_news"),
                sender_chat=None,
                chat=None,
                sender_user_name=None,
            ),
            media_group_id="group-42",
            message_thread_id=None,
            reply_text=reply_text,
        ),
        effective_user=SimpleNamespace(id=7),
        effective_chat=SimpleNamespace(id=100),
    )
    context = SimpleNamespace(
        user_data={},
        job_queue=job_queue,
        bot=SimpleNamespace(),
    )

    with patch(
        "tg_audiobooker._load_document_text",
        new=AsyncMock(return_value="Первая часть книги"),
    ):
        asyncio.run(handle_document(update, context))

    batch_key = build_forward_batch_key(
        chat_id=100,
        user_id=7,
        media_group_id="group-42",
    )
    buffered = context.user_data[FORWARDED_BATCHES_KEY][batch_key]
    assert buffered == [("Первая часть книги", 321, "#source_news")]

    assert len(job_queue.run_once_calls) == 1
    run_once_call = job_queue.run_once_calls[0]
    assert run_once_call["name"] == build_forward_job_name(batch_key)
    assert run_once_call["data"]["batch_key"] == batch_key
    assert run_once_call["user_id"] == 7
    assert run_once_call["chat_id"] == 100
    reply_text.assert_not_called()
