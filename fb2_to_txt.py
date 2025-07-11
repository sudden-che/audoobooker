import xml.etree.ElementTree as ET
from pathlib import Path
import re


# Входной FB2-файл
INPUT_FILE = "Торрес Армандо. Свидетели нагваля - royallib.ru.txt"

# Автоматически получить имя output-файла
input_path = Path(INPUT_FILE)
OUTPUT_FILE = input_path.with_suffix(".txt")

def extract_text_from_fb2(fb2_path):
    tree = ET.parse(fb2_path)
    root = tree.getroot()
    ns = {'fb': 'http://www.gribuser.ru/xml/fictionbook/2.0'}
    paragraphs = root.findall('.//fb:body//fb:p', ns)
    lines = [p.text.strip() for p in paragraphs if p.text]
    return "\n\n".join(lines)




def clean_text(text: str) -> str:
    # NBSP → обычный пробел
    text = text.replace("\xa0", " ")

    # Удаление управляющих символов кроме \n, \t
    text = ''.join(c for c in text if c.isprintable() or c in '\n\t')

    # Замена последовательностей пробелов/табов на один пробел
    text = re.sub(r'[ \t]+', ' ', text)

    # Удаление лишних пробелов в начале/конце строк
    text = '\n'.join(line.strip() for line in text.splitlines())

    # Удаление лишних пустых строк (оставляем максимум 1 подряд)
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()

def main():
    print(f"[=] Конвертация {INPUT_FILE} → {OUTPUT_FILE} ...")
    text = extract_text_from_fb2(INPUT_FILE)
    OUTPUT_FILE.write_text(text, encoding="utf-8")
    print("✅ Готово. Сохранено как:", OUTPUT_FILE.name)


    text = Path(INPUT_FILE).read_text(encoding="utf-8")
    cleaned = clean_text(text)
    Path(OUTPUT_FILE).write_text(cleaned, encoding="utf-8")
    print(f"✅ Файл очищен и сохранён как: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
