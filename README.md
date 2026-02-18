# Audiobooker

Audiobooker - это скрипт на Python, который берет файл текста в качестве входных данных и генерирует аудиокнигу в формате MP3 с помощью сервиса Azure TTS (Text-to-Speech).

## Использование
0. скачайте ffmpeg для склейки mp3 в 1 файл
1. Установите необходимые пакеты, запустив `pip install -r requirements.txt`.

    Файл конфигурации Azure TTS.

    Файл должен содержать следующие переменные:
    - `VOICE`: голос, который будет использоваться для аудиокниги. Например, "Male: ru-RU-DmitryNeural, Female: ru-RU-SvetlanaNeural".
    - `RATE`: скорость речи. Например, "+0%".
    - `CHUNK_SIZE`: размер каждого фрагмента текста для обработки. Например, 1000.
    - `SKIP_CHUNKS`: флаг, указывающий, нужно ли пропускать фрагменты, которые уже существуют. Например, False.
    - `SKIP_MERGE`: флаг, указывающий, нужно ли пропускать объединение фрагментов. Например, False.
    - `OUTPUT_PATH`: путь к директории вывода. Например, "output".

        #    check voices for your locale
        from edge_tts import Communicate,list_voices

        async def main():
            voices = await list_voices()
            for v in voices:
                if v["Locale"] == "ru-RU":
                    print(v["ShortName"], "-", v["Gender"])
        asyncio.run(main())



    Например:

        VOICE = "en-US-JennyNeural"
        RATE = "+0%"
        CHUNK_SIZE = 1000
        SKIP_CHUNKS = False
        SKIP_MERGE = False
        OUTPUT_PATH = "output"

2. Запустите скрипт с помощью следующей команды:

    `python audiobooker.py <input_file>`

    Например:

        python audiobooker.py book.txt

    Это сгенерирует аудиокнигу в формате MP3 в директории `output`.

## Веб-интерфейс

Добавлен отдельный файл `web_audiobooker.py`, который не меняет поведение исходного CLI-скрипта.

### Установка зависимостей для веб-версии

```bash
pip install -r requirements-web.txt
```

### Запуск веб-приложения

```bash
python web_audiobooker.py
```

или

```bash
uvicorn web_audiobooker:app --host 0.0.0.0 --port 8000
```

После запуска откройте `http://localhost:8000`.

В интерфейсе можно:
- загрузить `.txt` или `.fb2`;
- настроить voice, speed, chunk size, параллелизм, merge/skip флаги и путь к ffmpeg;
- нажать Start и скачать результат: единый `.mp3` (если merge включён) или `.tar` с частями (если merge выключен).

## Другие скрипты

Репозиторий содержит два других скрипта:

- `fb2_to_txt.py`: скрипт, который берет файл FB2 в качестве входных данных и генерирует текстовый файл. Этот скрипт полезен для преобразования файлов FB2 в текстовые файлы, которые можно использовать в качестве входных данных для скрипта `audiobooker.py`.
- `clean.py`: скрипт, который берет текстовый файл в качестве входных данных и генерирует очищенный текстовый файл. Этот скрипт полезен для очистки текстовых файлов перед использованием их в качестве входных данных для скрипта `audiobooker.py`.

Эти скрипты можно использовать независимо от скрипта `audiobooker.py`. Например, вы можете использовать скрипт `fb2_to_txt.py` для преобразования файла FB2 в текстовый файл, а затем использовать скрипт `clean.py` для очистки текстового файла перед использованием его в качестве входных данных для скрипта `audiobooker.py`.

examples of sample.txt in example dir
