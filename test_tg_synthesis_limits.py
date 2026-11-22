import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from test_tg_forward_grouping import _install_tg_stubs

_install_tg_stubs()
import tg_audiobooker as bot


class TelegramSynthesisLimitsTests(unittest.TestCase):
    def test_failure_cancels_other_chunk_tasks(self):
        cancelled = []

        async def fake_synthesis(text, file_path, voice, rate, semaphore):
            if text.startswith('fail'):
                await asyncio.sleep(0)
                raise RuntimeError('synthesis failed')
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.append(True)

        async def run():
            with tempfile.TemporaryDirectory() as tmp, patch.object(bot, 'CHUNK_SIZE', 4), patch.object(
                bot, 'synthesize_chunk_edge', fake_synthesis
            ):
                with self.assertRaisesRegex(RuntimeError, 'synthesis failed'):
                    await bot.generate_audio('fail more', Path(tmp), {'chunk_size': 4})
                self.assertEqual(cancelled, [True])

        asyncio.run(run())

    def test_saved_settings_are_capped_for_generation(self):
        observed = []
        progress = []

        async def fake_synthesis(text, file_path, voice, rate, semaphore):
            observed.append((len(text), semaphore._value))
            file_path.write_bytes(b'ID3' + b'\0' * 32)

        async def report(current, total):
            progress.append((current, total))

        settings = {'chunk_size': 10000, 'max_concurrent_tasks': 40, 'merge_chunks': False}
        with tempfile.TemporaryDirectory() as tmp, patch.object(bot, 'CHUNK_SIZE', 2000), patch.object(
            bot, 'MAX_CONCURRENT_TASKS', 2
        ), patch.object(bot, 'synthesize_chunk_edge', fake_synthesis):
            outputs = asyncio.run(bot.generate_audio('word ' * 1000, Path(tmp), settings, on_progress=report))
        self.assertGreater(len(outputs), 1)
        self.assertTrue(all(length <= 2000 and permits <= 2 for length, permits in observed))
        self.assertEqual(progress[-1], (len(outputs), len(outputs)))
        self.assertEqual(settings['chunk_size'], 10000)
        self.assertEqual(settings['max_concurrent_tasks'], 40)


if __name__ == '__main__':
    unittest.main()
