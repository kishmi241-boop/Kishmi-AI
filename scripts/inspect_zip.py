import zipfile
from collections import Counter

def inspect_zip_folders():
    zip_path = r"d:\Kishmi\archive.zip"
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            infos = zip_ref.infolist()
            print(f"Total files in archive: {len(infos)}")
            
            # Count files in each directory path (first two components of the path)
            paths = []
            for info in infos:
                if not info.is_dir():
                    parts = info.filename.split('/')
                    if len(parts) >= 2:
                        paths.append("/".join(parts[:2]))
                    else:
                        paths.append("root")
                        
            counts = Counter(paths)
            print("\nFolder structure and file counts:")
            for folder, count in sorted(counts.items()):
                print(f"  {folder}: {count} files")
                
    except Exception as e:
        print(f"Error reading zip: {e}")

if __name__ == "__main__":
    inspect_zip_folders()
