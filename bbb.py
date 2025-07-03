import asyncio
from edge_tts import Communicate

# Все известные русские женские голоса в Azure TTS (нейросетевые)
RUSSIAN_FEMALE_VOICES = [


    "ru-RU-SvetlanaNeural"
]

TEXT = "Здравствуйте! Это тест русского женского голоса для озвучивания текста."

async def test_voice(voice_name: str):
    print(f"[=] Генерация для голоса: {voice_name}")
    communicator = Communicate(
        text=TEXT,
        voice=voice_name,
        rate="+25%",    # чуть медленнее
        
    )
    output_file = f"{voice_name}.mp3"
    await communicator.save(output_file)
    print(f"[+] Сохранено: {output_file}")

async def main():
    for voice in RUSSIAN_FEMALE_VOICES:
        await test_voice(voice)
    print("✅ Все голоса сохранены.")

try:
    asyncio.run(main())
except Exception as e:
    print(f"[-] Ошибка: {e}")
    