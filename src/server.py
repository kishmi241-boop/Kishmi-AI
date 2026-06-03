"""
KISHMI AURA - AI Skin Analysis Backend Server
Flask API that accepts an image, runs YOLOv8 detection, calculates severity scores,
and returns product recommendations mapped to the Kishmi product catalog.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import glob
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
app = Flask(__name__, static_folder=ROOT_DIR, static_url_path='')
CORS(app)  # Allow all cross-origin requests from the web app

@app.route('/')
def serve_index():
    return app.send_static_file('index.html')

# ─── PRODUCT RECOMMENDATION ENGINE ─────────────────────────────────────────────
# Kishmi product catalog mapped to detected skin conditions & severity

PRODUCT_CATALOG = {
    "Acne": {
        "morning": [
            {"product": "Kishmi Clarifying Gel Cleanser",       "benefit": "removes excess oil & kills acne bacteria"},
            {"product": "Kishmi 2% Salicylic Acid BHA Exfoliant","benefit": "unclogs pores & prevents future breakouts"},
            {"product": "Kishmi Matte SPF 50 PA+++",            "benefit": "protects skin without clogging pores"},
        ],
        "night": [
            {"product": "Kishmi Clarifying Gel Cleanser",       "benefit": "deep-cleanses daily pollution & sebum"},
            {"product": "Kishmi Cica & Zinc Spot Gel",          "benefit": "calms active inflammations overnight"},
        ]
    },
    "Acne Marks": {
        "morning": [
            {"product": "Kishmi 10% Niacinamide Serum",         "benefit": "fades dark marks & evens skin tone"},
            {"product": "Kishmi Matte SPF 50 PA+++",            "benefit": "prevents UV from darkening existing marks"},
        ],
        "night": [
            {"product": "Kishmi 10% Niacinamide Serum",         "benefit": "overnight mark fading & brightening"},
        ]
    },
    "Pigmentation": {
        "morning": [
            {"product": "Kishmi 10% Niacinamide Serum",         "benefit": "blocks melanin transfer to skin surface"},
            {"product": "Kishmi Matte SPF 50 PA+++",            "benefit": "essential UV guard for pigmentation"},
        ],
        "night": [
            {"product": "Kishmi Barrier Repair Gel",            "benefit": "repairs skin barrier & reduces discoloration"},
        ]
    },
    "Wrinkles": {
        "morning": [
            {"product": "Kishmi Lightweight Ceramide Gel",      "benefit": "plumps fine lines with deep hydration"},
            {"product": "Kishmi Matte SPF 50 PA+++",            "benefit": "prevents UV-induced collagen breakdown"},
        ],
        "night": [
            {"product": "Kishmi Barrier Repair Gel",            "benefit": "rebuilds collagen & moisture overnight"},
        ]
    },
    "Dark Circles": {
        "morning": [
            {"product": "Kishmi 10% Niacinamide Serum",         "benefit": "brightens periorbital dark patches"},
            {"product": "Kishmi Lightweight Ceramide Gel",      "benefit": "hydrates and firms under-eye area"},
        ],
        "night": [
            {"product": "Kishmi Barrier Repair Gel",            "benefit": "overnight repair for dark circles"},
        ]
    },
    "Oily Skin": {
        "morning": [
            {"product": "Kishmi Clarifying Gel Cleanser",       "benefit": "removes excess oil without stripping"},
            {"product": "Kishmi 2% Salicylic Acid BHA Exfoliant","benefit": "controls sebum production"},
            {"product": "Kishmi Matte SPF 50 PA+++",            "benefit": "matte finish protection for oily skin"},
        ],
        "night": [
            {"product": "Kishmi Clarifying Gel Cleanser",       "benefit": "deep-cleanses accumulated oil"},
            {"product": "Kishmi Lightweight Ceramide Gel",      "benefit": "lightweight hydration without greasiness"},
        ]
    },
    "Dry Skin": {
        "morning": [
            {"product": "Kishmi Hydrating Foam Cleanser",       "benefit": "gentle cleanse that retains moisture"},
            {"product": "Kishmi Lightweight Ceramide Gel",      "benefit": "deep ceramide hydration for dry skin"},
            {"product": "Kishmi Matte SPF 50 PA+++",            "benefit": "protects dry skin from UV damage"},
        ],
        "night": [
            {"product": "Kishmi Barrier Repair Gel",            "benefit": "rebuilds skin moisture barrier overnight"},
            {"product": "Kishmi Lightweight Ceramide Gel",      "benefit": "locks in deep hydration while you sleep"},
        ]
    },
    "Redness": {
        "morning": [
            {"product": "Kishmi Hydrating Foam Cleanser",       "benefit": "gentle non-irritating cleanse"},
            {"product": "Kishmi Barrier Repair Gel",            "benefit": "calms inflammation and soothes redness"},
            {"product": "Kishmi Matte SPF 50 PA+++",            "benefit": "shields sensitive skin from UV flare-ups"},
        ],
        "night": [
            {"product": "Kishmi Barrier Repair Gel",            "benefit": "repairs irritated skin overnight"},
            {"product": "Kishmi Cica & Zinc Spot Gel",          "benefit": "soothes redness and calms inflammation"},
        ]
    },
    "Normal Skin": {
        "morning": [
            {"product": "Kishmi Hydrating Foam Cleanser",       "benefit": "maintains healthy pH balance"},
            {"product": "Kishmi Matte SPF 50 PA+++",            "benefit": "daily sun protection"},
        ],
        "night": [
            {"product": "Kishmi Lightweight Ceramide Gel",      "benefit": "maintains healthy moisture barrier"},
        ]
    },
    "Combination Skin": {
        "morning": [
            {"product": "Kishmi Hydrating Foam Cleanser",       "benefit": "balances oily T-zone and dry cheeks"},
            {"product": "Kishmi 10% Niacinamide Serum",         "benefit": "regulates overall oil production"},
            {"product": "Kishmi Matte SPF 50 PA+++",            "benefit": "protects without adding shine"},
        ],
        "night": [
            {"product": "Kishmi Clarifying Gel Cleanser",       "benefit": "removes impurities effectively"},
            {"product": "Kishmi Lightweight Ceramide Gel",      "benefit": "hydrates dry areas lightly"},
        ]
    },
    "Sensitive Skin": {
        "morning": [
            {"product": "Kishmi Hydrating Foam Cleanser",       "benefit": "ultra-gentle cleansing"},
            {"product": "Kishmi Barrier Repair Gel",            "benefit": "fortifies skin against irritants"},
        ],
        "night": [
            {"product": "Kishmi Barrier Repair Gel",            "benefit": "soothes and repairs daily damage"},
        ]
    },
}

BASE_ROUTINE = {
    "morning": [
        {"product": "Kishmi Hydrating Foam Cleanser",   "benefit": "gentle daily cleanse for all skin types"},
        {"product": "Kishmi Lightweight Ceramide Gel",  "benefit": "locks in moisture throughout the day"},
    ],
    "night": [
        {"product": "Kishmi Hydrating Foam Cleanser",   "benefit": "removes sunscreen and day's buildup"},
    ]
}

# Class names matching the new merged YOLO model's dataset.yaml
# Classes 4-10 are junk labels from Roboflow metadata — they will never fire
YOLO_CLASS_NAMES = {
    0: "melasma",
    1: "Dry Skin",
    2: "pores",
    3: "Oily Skin",
    11: "Dark Circles",
    12: "Acne",
    13: "Normal Skin",
    14: "Combination Skin",
    15: "Sensitive Skin",
}

# All conditions we track (YOLO + OpenCV Redness)
ALL_CONDITIONS = ["Acne", "Dark Circles", "Oily Skin", "Dry Skin", "Normal Skin", "Combination Skin", "Sensitive Skin", "Redness", "melasma", "pores"]

# ─── OPENCV REDNESS DETECTION ──────────────────────────────────────────────────
def detect_redness(img):
    """
    Detects redness in a face image using HSV color space analysis.
    Takes an OpenCV image (numpy array) already loaded in memory.
    Returns a score from 0.0 (no redness) to 1.0 (severe redness).
    """
    if img is None:
        return 0.0

    # Convert BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define red color range in HSV (red wraps around 0/180 in OpenCV)
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 50, 50])
    upper_red2 = np.array([180, 255, 255])

    # Create red masks and combine
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    # Calculate % of red pixels
    red_pixels = cv2.countNonZero(red_mask)
    total_pixels = img.shape[0] * img.shape[1]
    redness_ratio = red_pixels / total_pixels

    # Scale: typical skin redness covers 2-15% of face area
    # Map to 0-5 severity scale
    redness_score = min(redness_ratio * 5, 1.0)
    return round(redness_score, 3)

def redness_to_severity(score):
    """Convert redness score (0.0-1.0) to severity (0-5)."""
    if score < 0.05: return 0
    if score < 0.15: return 1
    if score < 0.30: return 2
    if score < 0.50: return 3
    if score < 0.75: return 4
    return 5

def calculate_severity(count):
    """Convert detection count to a 0-5 severity score."""
    if count == 0:   return 0
    if count <= 2:   return 1
    if count <= 5:   return 2
    if count <= 10:  return 3
    if count <= 20:  return 4
    return 5

def build_recommendations(severity_scores):
    """Build a merged morning/night routine based on detected conditions."""
    morning_products = {}
    night_products   = {}



    # Find the single most prominent condition
    if not severity_scores:
        return {"morning": [], "night": []}
        
    top_condition = max(severity_scores, key=severity_scores.get)
    top_score = severity_scores[top_condition]

    # Only recommend products for the #1 top concern to keep the routine minimal and targeted
    if top_score >= 1 and top_condition in PRODUCT_CATALOG:
        for step in PRODUCT_CATALOG[top_condition]["morning"]:
            morning_products[step["product"]] = step["benefit"]
        for step in PRODUCT_CATALOG[top_condition]["night"]:
            night_products[step["product"]] = step["benefit"]

    return {
        "morning": [{"product": k, "benefit": v} for k, v in morning_products.items()],
        "night":   [{"product": k, "benefit": v} for k, v in night_products.items()],
    }

def build_skin_story(severity_scores, detected_conditions):
    """Generate a natural-language skin story based on detections."""
    if not detected_conditions:
        return "Your skin looks impressively healthy! Our AI found no visible concerns. Keep up your current routine and stay consistent with SPF."

    top = max(severity_scores, key=severity_scores.get)
    severity_level = severity_scores[top]
    level_text = ["clear", "very mild", "mild", "moderate", "severe", "very severe"][severity_level]

    conditions_text = ", ".join(detected_conditions)
    return (
        f"Our AURA AI detected {conditions_text} on your facial scan. "
        f"The most prominent concern is {top} at a {level_text} severity level. "
        f"We have curated a targeted Kishmi ritual to address each concern with clinically-backed active ingredients, "
        f"tailored for diverse Indian skin tones."
    )

# ─── MODEL LOADER ──────────────────────────────────────────────────────────────
model = None

def load_model():
    global model
    # Search for best.pt in common locations
    search_paths = [
        "models/best.pt",
        "models/skin_analysis_v1/weights/best.pt",
        "runs/detect/train/weights/best.pt",
    ]
    for path in search_paths:
        if os.path.exists(path):
            model = YOLO(path)
            print(f"[INFO] Model loaded from: {path}")
            return
    print("[WARNING] best.pt not found! Using base YOLOv8n as fallback for testing.")
    model = YOLO("yolov8n.pt")

# ─── API ENDPOINT ───────────────────────────────────────────────────────────────
@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Accepts a base64-encoded image (JSON body: { "image": "data:image/jpeg;base64,..." })
    Returns: detected markers, severity scores, skin story, product recommendations.
    """
    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "No image provided"}), 400

    try:
        # Decode base64 image
        img_data = data["image"].split(",")[1]  # strip "data:image/jpeg;base64,"
        img_bytes = base64.b64decode(img_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        h, w = img.shape[:2]

        # Run YOLO inference
        results = model.predict(img, imgsz=640, conf=0.25, verbose=False)
        boxes = results[0].boxes

        # Parse detections
        condition_counts  = {c: 0 for c in ALL_CONDITIONS}
        markers = []

        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                cls_idx = int(box.cls[0])
                if cls_idx in YOLO_CLASS_NAMES:
                    condition = YOLO_CLASS_NAMES[cls_idx]
                    condition_counts[condition] += 1
                    conf = float(box.conf[0])

                    # Convert bbox to percentage coordinates for frontend
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cx = ((x1 + x2) / 2) / w * 100
                    cy = ((y1 + y2) / 2) / h * 100

                    markers.append({
                        "concern": condition,
                        "desc": f"Confidence: {conf:.0%}",
                        "x": round(cx, 1),
                        "y": round(cy, 1),
                    })

        # OpenCV Redness detection
        redness_score = detect_redness(img)
        redness_severity = redness_to_severity(redness_score)
        condition_counts["Redness"] = redness_severity  # Use severity directly as count
        if redness_severity > 0:
            markers.append({
                "concern": "Redness",
                "desc": f"Score: {redness_score:.1%}",
                "x": 50.0,  # Center of face
                "y": 50.0,
            })

        # Calculate severity scores
        severity_scores = {c: calculate_severity(condition_counts[c]) for c in ALL_CONDITIONS}
        # Override Redness severity with the direct calculation
        severity_scores["Redness"] = redness_severity
        detected = [c for c, s in severity_scores.items() if s > 0]

        # Build response
        response = {
            "markers": markers,
            "severity_scores": severity_scores,
            "detected_conditions": detected,
            "story": build_skin_story(severity_scores, detected),
            "noticed": [
                f"**{c}:** Severity {s}/5 — {condition_counts[c]} instance(s) detected."
                for c, s in severity_scores.items() if s > 0
            ] or ["No visible skin concerns detected. Great skin health!"],
            "ritual": build_recommendations(severity_scores),
            "tips": [
                "Apply SPF 50 every single morning — even on cloudy days.",
                "Use a clean microfibre towel to gently pat your face dry.",
                "Drink at least 2.5L of water daily to support skin hydration.",
                "Never skip your night routine — skin repairs itself while you sleep.",
            ],
            "dermWarning": (
                "If you experience persistent acne lasting more than 3 months, severe pigmentation, or painful cysts, "
                "please consult a certified dermatologist. This tool is for cosmetic guidance only."
            )
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "AURA AI Server is running", "model_loaded": model is not None})

if __name__ == "__main__":
    load_model()
    app.run(host="0.0.0.0", port=5000, debug=False)
