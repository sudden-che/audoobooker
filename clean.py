import re
from pathlib import Path

INPUT_FILE = "Vafin_Govorit-Vafin.IOENYQ.559764.txt"
input_path = Path(INPUT_FILE)
OUTPUT_FILE = input_path.with_suffix(".txt")

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
    text = Path(INPUT_FILE).read_text(encoding="utf-8")
    cleaned = clean_text(text)
    Path(OUTPUT_FILE).write_text(cleaned, encoding="utf-8")
    print(f"✅ Файл очищен и сохранён как: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
