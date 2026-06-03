import os
import fitz  # PyMuPDF

def render_pages():
    pdf_path = r"d:\Kishmi\kishmi all products.pdf"
    output_dir = r"d:\Kishmi\extracted_pdf_pages"
    os.makedirs(output_dir, exist_ok=True)
    
    doc = fitz.open(pdf_path)
    print(f"Total pages: {len(doc)}")
    
    for i, page in enumerate(doc):
        # Extract text using PyMuPDF
        text = page.get_text()
        print(f"\n--- PAGE {i+1} Text ---")
        print(text[:200]) # print first 200 chars
        
        # Render page to PNG
        # Increase zoom factor for high quality (e.g. 2x)
        zoom = 2.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        
        out_path = os.path.join(output_dir, f"page_{i+1}.png")
        pix.save(out_path)
        print(f"Rendered Page {i+1} to {out_path}")

if __name__ == "__main__":
    render_pages()
