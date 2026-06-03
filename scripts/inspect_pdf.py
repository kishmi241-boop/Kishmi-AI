import pypdf

def extract_pdf_text():
    pdf_path = r"d:\Kishmi\kishmi all products.pdf"
    reader = pypdf.PdfReader(pdf_path)
    print(f"Total pages: {len(reader.pages)}")
    
    with open("pdf_text.txt", "w", encoding="utf-8") as f:
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            f.write(f"--- PAGE {i+1} ---\n")
            f.write(text)
            f.write("\n\n")
    print("Done! Extracted text to pdf_text.txt")

if __name__ == "__main__":
    extract_pdf_text()
