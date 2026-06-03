import os
import pypdf

def extract_images():
    pdf_path = r"d:\Kishmi\kishmi all products.pdf"
    output_dir = r"d:\Kishmi\extracted_pdf_images"
    os.makedirs(output_dir, exist_ok=True)
    
    reader = pypdf.PdfReader(pdf_path)
    img_count = 0
    
    for page_idx, page in enumerate(reader.pages):
        print(f"Checking page {page_idx + 1} for images...")
        # Check resources
        if "/Resources" in page and "/XObject" in page["/Resources"]:
            xobjects = page["/Resources"]["/XObject"].get_object()
            for obj_name in xobjects:
                obj = xobjects[obj_name]
                if obj["/Subtype"] == "/Image":
                    img_count += 1
                    try:
                        # Extract image data
                        image_data = obj.get_data()
                        # Get image metadata
                        width = obj["/Width"]
                        height = obj["/Height"]
                        color_space = obj.get("/ColorSpace", "Unknown")
                        
                        # Determine extension from filter
                        filter_type = obj.get("/Filter", "")
                        ext = ".png"
                        if filter_type == "/DCTDecode" or "DCT" in str(filter_type):
                            ext = ".jpg"
                        elif filter_type == "/JPXDecode":
                            ext = ".jp2"
                        
                        out_path = os.path.join(output_dir, f"page_{page_idx + 1}_{obj_name.replace('/', '_')}{ext}")
                        with open(out_path, "wb") as img_file:
                            img_file.write(image_data)
                        
                        print(f"  -> Extracted: {out_path} ({width}x{height}, filter: {filter_type})")
                    except Exception as e:
                        print(f"  -> Failed to extract image {obj_name} on page {page_idx + 1}: {e}")
                        
    print(f"Finished extracting images. Total images saved: {img_count}")

if __name__ == "__main__":
    extract_images()
