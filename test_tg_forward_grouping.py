from tg_audiobooker import (
    FORWARDED_BATCHES_KEY,
    build_forward_batch_key,
    build_forward_job_name,
    ensure_forwarded_batches,
    get_user_processing_lock,
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
