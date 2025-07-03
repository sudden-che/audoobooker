import xml.etree.ElementTree as ET
from pathlib import Path

# Входной FB2-файл
INPUT_FILE = "Vafin_Govorit-Vafin.IOENYQ.559764.fb2"

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

def main():
    print(f"[=] Конвертация {INPUT_FILE} → {OUTPUT_FILE} ...")
    text = extract_text_from_fb2(INPUT_FILE)
    OUTPUT_FILE.write_text(text, encoding="utf-8")
    print("✅ Готово. Сохранено как:", OUTPUT_FILE.name)

if __name__ == "__main__":
    main()
