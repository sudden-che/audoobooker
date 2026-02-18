import xml.etree.ElementTree as ET
from pathlib import Path
import re
import sys

# === Аргументы ===
if len(sys.argv) > 1:
    INPUT_FILE = sys.argv[1]
else:
    INPUT_FILE = "sample.fb2"

input_path = Path(INPUT_FILE)
OUTPUT_FILE = input_path.with_suffix(".txt")


# === Очистка текста ===
def clean_text(text: str) -> str:
    # Базовая нормализация
    text = text.replace("\xa0", " ")  # NBSP → пробел
    text = text.replace("«", '"').replace("»", '"')  # «» → "
    text = "".join(c for c in text if c.isprintable() or c in "\n\t")

    # Внутри абзацев — схлопываем многократные пробелы
    text = re.sub(r"[ \t]+", " ", text)

    # Стрип по краям каждой строки
    text = "\n".join(line.strip() for line in text.splitlines())

    # ВАЖНО: сохраняем пустую строку между абзацами.
    # Убираем только ТРИ и более пустых строк → оставляем максимум ДВЕ (\n\n)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Финальный трим
    return text.strip()


# === Извлечение текста из FB2 с учётом вложенных тегов ===
def extract_and_clean_text(fb2_path: str) -> str:
    tree = ET.parse(fb2_path)
    root = tree.getroot()
    ns = {"fb": "http://www.gribuser.ru/xml/fictionbook/2.0"}

    # Собираем все абзацы
    paragraphs = root.findall(".//fb:body//fb:p", ns)

    # ВАЖНО: используем itertext(), чтобы не терять текст из вложенных тегов (<emphasis> и др.)
    def p_text(p_el) -> str:
        return "".join(p_el.itertext()).strip()

    lines = [p_text(p) for p in paragraphs if p_text(p)]
    raw_text = "\n\n".join(lines)  # одна пустая строка между абзацами

    return clean_text(raw_text)


# === Основной процесс ===
def main():
    print(f"[=] Конвертация {INPUT_FILE} -> {OUTPUT_FILE} ...")
    final_text = extract_and_clean_text(INPUT_FILE)
    OUTPUT_FILE.write_text(final_text, encoding="utf-8")
    print(f"[!] Готово. Сохранено как: {OUTPUT_FILE.name}")


if __name__ == "__main__":
    main()
