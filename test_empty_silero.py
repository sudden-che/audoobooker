import asyncio
from pathlib import Path
from audiobooker import synthesize_chunk_silero

async def main():
    import tempfile
    semaphore = asyncio.Semaphore(1)
    file_path = Path("test_silero.wav")
    await synthesize_chunk_silero(
        text="random=true",
        file_path=file_path,
        language="ru",
        speaker="aidar",
        sample_rate=48000,
        put_accent=True,
        put_yo=True,
        device="cpu",
        model_id="v5_ru",
        semaphore=semaphore
    )

asyncio.run(main())    
