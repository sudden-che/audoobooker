import xml.etree.ElementTree as ET
from pathlib import Path
import re
import sys

#sys.stdout.reconfigure(encoding='utf-8')

# === Аргументы ===
if len(sys.argv) > 1:
    INPUT_FILE = sys.argv[1]
else:
    INPUT_FILE = "sample.fb2"

input_path = Path(INPUT_FILE)
OUTPUT_FILE = input_path.with_suffix(".txt")

# === Очистка текста ===
def clean_text(text: str) -> str:
    text = text.replace("\xa0", " ")  # NBSP → обычный пробел
    text = ''.join(c for c in text if c.isprintable() or c in '\n\t')  # Удаление управляющих символов
    text = re.sub(r'[ \t]+', ' ', text)  # Повторяющиеся пробелы и табуляции

    # Удаление пробелов в начале и конце строк
    text = '\n'.join(line.strip() for line in text.splitlines())

    # Объединение строк, если нет двойного \n между ними
    # Это "склеит" разорванные абзацы
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)

    # Удаление более двух \n подряд → оставить максимум два
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Удаление пустых строк в начале и в конце
    text = text.strip()

    return text

# === Извлечение и очистка из FB2 ===
def extract_and_clean_text(fb2_path):
    tree = ET.parse(fb2_path)
    root = tree.getroot()
    ns = {'fb': 'http://www.gribuser.ru/xml/fictionbook/2.0'}
    paragraphs = root.findall('.//fb:body//fb:p', ns)
    lines = [p.text.strip() for p in paragraphs if p.text]
    raw_text = "\n\n".join(lines)
    return clean_text(raw_text)

# === Основной процесс ===
def main():
    
    print(f"[=] Конвертация {INPUT_FILE} - {OUTPUT_FILE} ...")
    final_text = extract_and_clean_text(INPUT_FILE)
    OUTPUT_FILE.write_text(final_text, encoding="utf-8")
    print(f"[!] Готово. Сохранено как: {OUTPUT_FILE.name}")

if __name__ == "__main__":
    main()
