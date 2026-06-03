import os
import json
import re

def verify_system():
    base_dir = r"d:\Kishmi"
    js_path = os.path.join(base_dir, "skin_database.js")
    
    print("--- STARTING SYSTEM INTEGRITY CHECKS ---")
    
    # 1. Check if database file exists
    if not os.path.exists(js_path):
        print(f"[FAIL] Database file not found at: {js_path}")
        return False
    print("[PASS] skin_database.js file exists.")
    
    # 2. Read database content and parse it
    with open(js_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    # Extract the JSON array from the JS file
    # Format: const SKIN_DATABASE = [ ... ];
    match = re.search(r'const SKIN_DATABASE = (\[.*?\]);', content, re.DOTALL)
    if not match:
        print("[FAIL] Could not extract SKIN_DATABASE variable from skin_database.js.")
        return False
        
    json_str = match.group(1)
    try:
        subjects = json.loads(json_str)
        print(f"[PASS] Successfully parsed skin_database JSON containing {len(subjects)} subjects.")
    except Exception as e:
        print(f"[FAIL] JSON parsing failed: {e}")
        return False
        
    # 3. Check subject count
    if len(subjects) != 15:
        print(f"[FAIL] Expected 15 subjects in database, found {len(subjects)}.")
        return False
    print("[PASS] Database contains exactly 15 subjects.")
    
    # 4. Verify that every image file path resolves and exists
    missing_files = 0
    checked_files = 0
    for sub in subjects:
        sub_id = sub['id']
        for key in ['frontImage', 'rightImage', 'leftImage']:
            rel_path = sub[key]
            full_path = os.path.join(base_dir, rel_path.replace('/', os.sep))
            checked_files += 1
            if not os.path.exists(full_path):
                print(f"[FAIL] Subject {sub_id} missing {key}: {rel_path} (resolved: {full_path})")
                missing_files += 1
                
    if missing_files > 0:
        print(f"[FAIL] Missing {missing_files} out of {checked_files} referenced image files!")
        return False
        
    print(f"[PASS] Verified all {checked_files} image file paths on disk successfully.")
    print("--- ALL INTEGRITY TESTS PASSED ---")
    return True

if __name__ == "__main__":
    success = verify_system()
    exit(0 if success else 1)
