from striprtf.striprtf import rtf_to_text

with open("book.rtf", "r", encoding="utf-8") as rtf_file:
    rtf_content = rtf_file.read()

plain_text = rtf_to_text(rtf_content)

with open("plaintext.txt", "w", encoding="utf-8") as txt_file:
    txt_file.write(plain_text)

print("✅ Сохранено в plaintext.txt")
