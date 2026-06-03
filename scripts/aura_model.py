import os
import math
import json
from PIL import Image

# =====================================================================
# KISHMI BRAND OFFICIAL PRODUCT CATALOG
# =====================================================================
# Extracted directly from: kishmi all products.pdf
KISHMI_PRODUCTS = {
    "Kishmi PureGlow Clarifying Face Wash": {
        "name": "Kishmi PureGlow Clarifying Face Wash",
        "size": "100 ml",
        "price": 399,
        "key_ingredients": "Salicylic Acid + Tea Tree Oil + Glycolic Acid",
        "description": "Clears acne, unclogs pores, and provides gentle exfoliation for a brighter complexion."
    },
    "Kishmi Simply Hydrating Niacinamide 3% Moisturizer": {
        "name": "Kishmi Simply Hydrating Niacinamide 3% Moisturizer",
        "size": "100 ml",
        "price": 499,
        "key_ingredients": "Niacinamide (3%) + Shea Butter + Almond Oil",
        "description": "Deeply hydrates, softens skin, and strengthens the natural lipid skin barrier."
    },
    "Kishmi Mineral Shield SPF 50+ Sunscreen": {
        "name": "Kishmi Mineral Shield SPF 50+ Sunscreen",
        "size": "50 ml",
        "price": 549,
        "key_ingredients": "Zinc Oxide + Titanium Dioxide + Vitamin E",
        "description": "Provides broad-spectrum physical UV protection in a lightweight, hydrating formula."
    },
    "Kishmi Repair & Glow 10% Niacinamide Serum": {
        "name": "Kishmi Repair & Glow 10% Niacinamide Serum",
        "size": "30 ml",
        "price": 599,
        "key_ingredients": "Niacinamide (10%) + Hyaluronic Acid + Vitamin E",
        "description": "Balances skin tone, minimizes enlarged pores, regulates sebum, and fades hyperpigmentation."
    },
    "Kishmi 10% Vitamin C Serum": {
        "name": "Kishmi 10% Vitamin C Serum",
        "size": "30 ml",
        "price": 649,
        "key_ingredients": "Ascorbic Acid (10%) + Hyaluronic Acid + Ferulic Acid",
        "description": "Brightens skin tone, fights free radicals, firms texture, and fades under-eye shadows."
    },
    "Kishmi Oil Control 2% Salicylic Face Serum": {
        "name": "Kishmi Oil Control 2% Salicylic Face Serum",
        "size": "30 ml",
        "price": 599,
        "key_ingredients": "Salicylic Acid (2%) + Aloe Vera + Niacinamide",
        "description": "Fights active breakouts, sweeps out dead cells, and clears pores for a smoother complexion."
    },
    "Kishmi Skin Booster 5% Niacinamide Glycolic Serum": {
        "name": "Kishmi Skin Booster 5% Niacinamide Glycolic Serum",
        "size": "30 ml",
        "price": 629,
        "key_ingredients": "Niacinamide (5%) + Glycolic Acid + Aloe Vera Extract",
        "description": "Exfoliates surface skin layers, refines rough texture, and sweeps away sweat-induced micro-bumps."
    },
    "Kishmi Skin Revival For 30+ 0.2% Retinol Cream": {
        "name": "Kishmi Skin Revival For 30+ 0.2% Retinol Cream",
        "size": "50 gm",
        "price": 699,
        "key_ingredients": "Retinol (0.2%) + Vitamin C + Peptides",
        "description": "Promotes cellular turnover, smooths fine lines, boots elasticity, and firms skin texture overnight."
    },
    "Kishmi Lip Glow Nourishing Balm": {
        "name": "Kishmi Lip Glow Nourishing Balm",
        "size": "10 gm",
        "price": 249,
        "key_ingredients": "Beeswax + Shea Butter + Cocoa Butter + Vitamin E",
        "description": "Provides deep hydration, locks in essential moisture, and softens dry or chapped lips."
    },
    "Kishmi Tear Free Baby Shampoo": {
        "name": "Kishmi Tear Free Baby Shampoo",
        "size": "100 ml",
        "price": 349,
        "key_ingredients": "Aloe Vera + Chamomile + Calendula",
        "description": "Extremely mild and tear-free formula that gently nourishes and cleanses baby's soft hair."
    },
    "Kishmi Baby Bubbles Body Wash": {
        "name": "Kishmi Baby Bubbles Body Wash",
        "size": "100 ml",
        "price": 349,
        "key_ingredients": "Aloe Vera + Chamomile + Calendula",
        "description": "Gentle, tear-free skin cleanser that soothes, moisturizes, and protects delicate skin layers."
    }
}

class AuraSkinModel:
    """
    AURA (Adaptive Understanding of Real skin needs by Antigravity)
    Computer-Vision Skin Analysis Model optimized for Indian Skin Tones.
    Reads face scans and maps characteristics to actual Kishmi brand products.
    Enhanced with trained Random Forest classifiers for acne severity, oily skin, dark circles, and wrinkles.
    """
    
    def __init__(self):
        self.target_w = 400
        self.target_h = 500
        
        self.has_ml = False
        self.ml_model = None
        self.ml_accuracy = 0.0
        
        self.has_multitask = False
        self.multitask_models = None
        
        # 1. Dynamically load the Random Forest acne classifier if present
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "aura_acne_model.pkl")
        if os.path.exists(model_path):
            try:
                import joblib
                self.ml_data = joblib.load(model_path)
                self.ml_model = self.ml_data["model"]
                self.ml_accuracy = self.ml_data["accuracy"]
                self.has_ml = True
            except Exception:
                pass
                
        # 2. Dynamically load the multitask attributes classifiers if present (archive 4)
        multitask_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "aura_multitask_models.pkl")
        if os.path.exists(multitask_path):
            try:
                import joblib
                self.multitask_models = joblib.load(multitask_path)
                self.has_multitask = True
            except Exception:
                pass

    def analyze_image(self, img_path):
        """
        Performs localized pixel analysis on a face scan.
        Returns computed skin metrics and scope-compliant classifications.
        """
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found at {img_path}")
            
        try:
            # Open and normalize image size
            img = Image.open(img_path).convert('RGB')
            img = img.resize((self.target_w, self.target_h))
        except Exception as e:
            raise ValueError(f"Could not load image: {e}")
            
        # 1. Spatial Pixel Zone Sampling
        # (Normalized box coordinates for front-profile facial photography)
        forehead_pixels = self._sample_zone(img, 150, 250, 70, 130)   # Forehead (T-Zone)
        nose_pixels = self._sample_zone(img, 180, 220, 200, 280)       # Nose (T-Zone)
        left_cheek_pixels = self._sample_zone(img, 80, 150, 250, 330)  # Left Cheek
        right_cheek_pixels = self._sample_zone(img, 250, 320, 250, 330) # Right Cheek
        chin_pixels = self._sample_zone(img, 160, 240, 350, 430)       # Chin / Jawline
        left_eye_pixels = self._sample_zone(img, 110, 160, 190, 220)   # Left Under-Eye
        right_eye_pixels = self._sample_zone(img, 240, 290, 190, 220)  # Right Under-Eye
        
        # 2. Extract Color & Luminance Statistics
        all_cheek_pixels = left_cheek_pixels + right_cheek_pixels
        avg_r = sum(p[0] for p in all_cheek_pixels) / len(all_cheek_pixels)
        avg_g = sum(p[1] for p in all_cheek_pixels) / len(all_cheek_pixels)
        avg_b = sum(p[2] for p in all_cheek_pixels) / len(all_cheek_pixels)
        
        # Perceived Lightness Index (Luminance Y)
        lightness = 0.299 * avg_r + 0.587 * avg_g + 0.114 * avg_b
        
        # A. Fitzpatrick Scale Classification (III - VI)
        fitzpatrick = "Type IV"
        if lightness > 175:
            fitzpatrick = "Type III"
        elif lightness > 125:
            fitzpatrick = "Type IV"
        elif lightness > 75:
            fitzpatrick = "Type V"
        else:
            fitzpatrick = "Type VI"
            
        # B. Skin Undertone Identification (Warm/Neutral/Cool)
        undertone = "Warm Golden"
        r_b_diff = avg_r - avg_b
        r_g_diff = avg_r - avg_g
        if r_b_diff > 45 and r_g_diff > 15:
            undertone = "Warm Golden"
        elif r_b_diff > 35 and avg_g > avg_b:
            undertone = "Warm Olive"
        elif abs(avg_r - avg_b) < 22:
            undertone = "Cool Ashy"
        else:
            undertone = "Neutral"
            
        # C. Sebum / Shine Classification (Skin Type)
        avg_tzone_r = (sum(p[0] for p in forehead_pixels + nose_pixels)) / (len(forehead_pixels + nose_pixels))
        avg_tzone_g = (sum(p[1] for p in forehead_pixels + nose_pixels)) / (len(forehead_pixels + nose_pixels))
        avg_tzone_b = (sum(p[2] for p in forehead_pixels + nose_pixels)) / (len(forehead_pixels + nose_pixels))
        tzone_lightness = 0.299 * avg_tzone_r + 0.587 * avg_tzone_g + 0.114 * avg_tzone_b
        
        shine_ratio = tzone_lightness / max(1.0, lightness)
        
        # Cheek texture standard deviation to capture pore size
        cheek_r = [p[0] for p in all_cheek_pixels]
        cheek_mean = sum(cheek_r) / len(cheek_r)
        cheek_var = sum((x - cheek_mean) ** 2 for x in cheek_r) / len(cheek_r)
        cheek_std = math.sqrt(cheek_var)
        
        skin_type = "Balanced / Normal"
        enlarged_pores = "Standard"
        if shine_ratio > 1.15:
            skin_type = "Oily"
            if cheek_std > 20.0:
                enlarged_pores = "Enlarged Pores (Active Sebum)"
        elif shine_ratio > 1.08:
            skin_type = "Combination (Oily T-Zone)"
            if cheek_std > 16.0:
                enlarged_pores = "Visible Pores on Cheeks"
        elif lightness < 110 and shine_ratio < 0.98:
            skin_type = "Dry"
        else:
            skin_type = "Balanced / Normal"
            
        # D. Acne Severity Classification
        # 1. Feature Engineering (Match the exact logic of the ML training pipeline)
        acne_pixels = 0
        acne_sum_intensity = 0.0
        for r_val, g_val, b_val in chin_pixels + all_cheek_pixels:
            if r_val > 110 and r_val - g_val > 30 and r_val - b_val > 30:
                acne_pixels += 1
                acne_sum_intensity += (r_val - g_val) + (r_val - b_val)
                
        total_sampled = len(chin_pixels + all_cheek_pixels)
        acne_density = acne_pixels / total_sampled
        acne_intensity = acne_sum_intensity / max(1.0, acne_pixels)
        
        # Standard deviation of red channel on cheeks/chin for roughness
        roughness_vals = [p[0] for p in chin_pixels + all_cheek_pixels]
        roughness_mean = sum(roughness_vals) / len(roughness_vals)
        roughness_var = sum((x - roughness_mean) ** 2 for x in roughness_vals) / len(roughness_vals)
        roughness_std = math.sqrt(roughness_var)
        
        red_green_ratio = avg_r / max(1.0, avg_g)
        red_blue_ratio = avg_r / max(1.0, avg_b)
        
        features = [
            lightness,
            acne_density,
            acne_intensity,
            roughness_std,
            red_green_ratio,
            red_blue_ratio
        ]
        
        # 1.5. Advanced Adaptive Skin-Pixel Analysis (ASPA)
        # Dynamically sample actual skin pixels globally to establish baseline skin color.
        # This acts as an extremely robust fallback for close-up skin scans and diverse lighting.
        all_pixels_img = []
        for x in range(0, self.target_w, 2):  # Subsample by 2 for performance
            for y in range(0, self.target_h, 2):
                all_pixels_img.append(img.getpixel((x, y)))
                
        skin_pixels_global = []
        for r_v, g_v, b_v in all_pixels_img:
            if r_v > 40 and r_v > g_v and g_v > b_v * 0.8 and r_v - g_v > 12 and r_v - b_v > 15 and r_v < 250:
                skin_pixels_global.append((r_v, g_v, b_v))
                
        has_valid_skin = len(skin_pixels_global) > 1000
        if has_valid_skin:
            avg_skin_r = sum(p[0] for p in skin_pixels_global) / len(skin_pixels_global)
            avg_skin_g = sum(p[1] for p in skin_pixels_global) / len(skin_pixels_global)
            avg_skin_b = sum(p[2] for p in skin_pixels_global) / len(skin_pixels_global)
            skin_lightness = 0.299 * avg_skin_r + 0.587 * avg_skin_g + 0.114 * avg_skin_b
            avg_skin_rg = avg_skin_r - avg_skin_g
            
            # If spatial zone lightness is heavily distorted by hair/backgrounds, use the robust skin lightness
            if abs(lightness - skin_lightness) > 15.0:
                lightness = skin_lightness
                if lightness > 175:
                    fitzpatrick = "Type III"
                elif lightness > 125:
                    fitzpatrick = "Type IV"
                elif lightness > 75:
                    fitzpatrick = "Type V"
                else:
                    fitzpatrick = "Type VI"
            
            # Count relative blemishes (contrast-based relative redness/darkness) in spatial zones
            rel_blemish_pixels = 0
            rel_red_pixels = 0
            rel_dark_pixels = 0
            for r_val, g_val, b_val in chin_pixels + all_cheek_pixels:
                l_val = 0.299 * r_val + 0.587 * g_val + 0.114 * b_val
                is_skin = (r_val > 40) and (r_val > g_val) and (g_val > b_val * 0.8)
                if is_skin:
                    is_red_blemish = (r_val - g_val) - avg_skin_rg > 15
                    is_dark_blemish = skin_lightness - l_val > 25
                    if is_red_blemish or is_dark_blemish:
                        rel_blemish_pixels += 1
                        if is_red_blemish:
                            rel_red_pixels += 1
                        if is_dark_blemish:
                            rel_dark_pixels += 1
                            
            rel_blemish_density = rel_blemish_pixels / len(chin_pixels + all_cheek_pixels)
            rel_red_density = rel_red_pixels / len(chin_pixels + all_cheek_pixels)
            rel_dark_density = rel_dark_pixels / len(chin_pixels + all_cheek_pixels)
        else:
            rel_blemish_density = 0.0
            rel_red_density = 0.0
            rel_dark_density = 0.0
        
        # Load Multitask Predictions if available (archive 4)
        ml_clear = 1
        ml_oily = 0
        ml_circles = 0
        ml_wrinkle = 0
        
        # Calculate precise ML features to match training pipeline
        import numpy as np
        ml_img = img.resize((256, 256))
        arr = np.array(ml_img, dtype=np.float32)
        ml_r, ml_g, ml_b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        
        # Skin Mask: R > 40 & R > G & G > B * 0.8 & R - G > 12 & R - B > 15
        skin_mask = (ml_r > 40) & (ml_r > ml_g) & (ml_g > ml_b * 0.8) & (ml_r - ml_g > 12) & (ml_r - ml_b > 15)
        
        if np.sum(skin_mask) < 1000:
            skin_mask = np.ones_like(ml_r, dtype=bool)
            
        r_skin = ml_r[skin_mask]
        g_skin = ml_g[skin_mask]
        b_skin = ml_b[skin_mask]
        
        ml_lightness = np.mean(0.299 * r_skin + 0.587 * g_skin + 0.114 * b_skin)
        
        ml_acne_mask = (r_skin > 110) & (r_skin - g_skin > 30) & (r_skin - b_skin > 30)
        ml_acne_pixels = np.sum(ml_acne_mask)
        ml_acne_density = ml_acne_pixels / len(r_skin)
        
        if ml_acne_pixels > 0:
            ml_acne_intensity = np.mean((r_skin[ml_acne_mask] - g_skin[ml_acne_mask]) + (r_skin[ml_acne_mask] - b_skin[ml_acne_mask]))
        else:
            ml_acne_intensity = 0.0
            
        ml_roughness_r = np.std(r_skin)
        ml_roughness_g = np.std(g_skin)
        ml_roughness_b = np.std(b_skin)
        
        ml_avg_r = np.mean(r_skin)
        ml_avg_g = np.mean(g_skin)
        ml_avg_b = np.mean(b_skin)
        ml_rg_ratio = ml_avg_r / max(1.0, float(ml_avg_g))
        ml_rb_ratio = ml_avg_r / max(1.0, float(ml_avg_b))
        
        ml_features = [
            ml_lightness,
            ml_acne_density,
            ml_acne_intensity,
            ml_roughness_r,
            ml_roughness_g,
            ml_roughness_b,
            ml_rg_ratio,
            ml_rb_ratio
        ]
        
        if self.has_multitask and self.multitask_models is not None:
            try:
                ml_clear = int(self.multitask_models["clear_skin"]["model"].predict([ml_features])[0])
                ml_oily = int(self.multitask_models["oily_skin"]["model"].predict([ml_features])[0])
                ml_circles = int(self.multitask_models["dark_circles"]["model"].predict([ml_features])[0])
                ml_wrinkle = int(self.multitask_models["wrinkle"]["model"].predict([ml_features])[0])
            except Exception:
                pass
                
        # ML Prediction for Acne Level with Heuristic Fallback
        acne_level = 0
        if self.has_ml and self.ml_model is not None:
            try:
                acne_level = int(self.ml_model.predict([ml_features])[0])
            except Exception:
                # Heuristic Fallback
                if acne_density > 0.05:
                    acne_level = 3
                elif acne_density > 0.025:
                    acne_level = 2
                elif acne_density > 0.008:
                    acne_level = 1
                else:
                    acne_level = 0
        else:
            # Robust Heuristic Fallback
            if acne_density > 0.05:
                acne_level = 3
            elif acne_density > 0.025:
                acne_level = 2
            elif acne_density > 0.008:
                acne_level = 1
            else:
                acne_level = 0
                
        # Dynamic ASPA Correction: Boost acne_level based on relative blemish contrast density
        if has_valid_skin:
            if rel_blemish_density > 0.40:
                acne_level = max(acne_level, 3)
            elif rel_blemish_density > 0.18:
                acne_level = max(acne_level, 2)
            elif rel_blemish_density > 0.05:
                acne_level = max(acne_level, 1)
                
        # Align clear_skin multitask prediction with acne level
        if self.has_multitask and ml_clear == 0 and acne_level == 0:
            acne_level = 1  # Override to Mild if multitask model flags breakouts
            
        acne_status_map = {
            0: "Clear / Resilient",
            1: "Mild Breakouts",
            2: "Moderate Breakouts",
            3: "Active Hormonal (Chin/Jawline)"
        }
        acne_status = acne_status_map[acne_level]
        
        # Align oily_skin multitask prediction with calculated skinType
        if self.has_multitask and ml_oily == 1 and skin_type == "Balanced / Normal":
            skin_type = "Combination (Oily T-Zone)"
        
        # D2. Acne Sub-aspects (Hormonal, Comedonal, Active Inflammation, PIH, Cystic)
        acne_type = "None"
        cystic_indicator = False
        pih_present = False
        
        if acne_level > 0:
            if has_valid_skin and rel_blemish_density > 0.05:
                # If red spots are very sparse relative to dark marks, classify primarily as post-acne marks (PIH)
                if rel_red_density / max(0.001, rel_dark_density) < 0.25:
                    acne_type = "Post-Acne Marks (PIH / PIE)"
                    pih_present = True
                else:
                    acne_type = "Active Inflammatory Acne"
                    if rel_blemish_density > 0.35:
                        cystic_indicator = True
            else:
                if acne_intensity > 28.0:
                    acne_type = "Active Inflammatory Acne"
                    if acne_density > 0.04 and acne_intensity > 35.0:
                        cystic_indicator = True
                else:
                    acne_type = "Post-Acne Marks (PIH / PIE)"
                    pih_present = True
                
        # E. Under-Eye Dark Circles Score & Types
        avg_eye_r = sum(p[0] for p in left_eye_pixels + right_eye_pixels) / len(left_eye_pixels + right_eye_pixels)
        avg_eye_g = sum(p[1] for p in left_eye_pixels + right_eye_pixels) / len(left_eye_pixels + right_eye_pixels)
        avg_eye_b = sum(p[2] for p in left_eye_pixels + right_eye_pixels) / len(left_eye_pixels + right_eye_pixels)
        eye_lightness = 0.299 * avg_eye_r + 0.587 * avg_eye_g + 0.114 * avg_eye_b
        
        eye_contrast = (lightness - eye_lightness) / max(1.0, lightness)
        
        dark_circles = "None"
        under_eye_detail = "Clear under-eye contours"
        
        # Trigger dark circles if either contrast heuristic or ML model flags it
        if eye_contrast > 0.05 or (self.has_multitask and ml_circles == 1):
            eye_blue_bias = avg_eye_b / max(1.0, avg_eye_r)
            cheek_blue_bias = avg_b / max(1.0, avg_r)
            
            if eye_blue_bias > cheek_blue_bias * 1.05:
                dark_circles = "Vascular Dark Circles"
                under_eye_detail = "Vascular-type dark circles (thin skin displaying micro-vessels)"
            else:
                dark_circles = "Pigmentation Dark Circles"
                under_eye_detail = "Pigmentation-type dark circles (melanin deposition/shadowing)"
                
        # Check standard deviation of eye pixels for puffiness / fine lines
        eye_vals = [p[0] for p in left_eye_pixels + right_eye_pixels]
        eye_mean = sum(eye_vals) / len(eye_vals)
        eye_var = sum((x - eye_mean) ** 2 for x in eye_vals) / len(eye_vals)
        eye_std = math.sqrt(eye_var)
        
        eye_puffiness = "None"
        eye_lines = "None"
        if eye_std > 22.0:
            eye_puffiness = "Mild Puffiness"
        if eye_std > 18.0 or (self.has_multitask and ml_wrinkle == 1):
            eye_lines = "Fine Lines"
            
        # F. Forehead Congestion & Climate Response (Roughness/Micro-bumps)
        tzone_rgb = [p[0] for p in forehead_pixels]
        mean_t = sum(tzone_rgb) / len(tzone_rgb)
        variance_t = sum((x - mean_t) ** 2 for x in tzone_rgb) / len(tzone_rgb)
        std_dev_t = math.sqrt(variance_t)
        
        congestion = "Clear"
        climate_response = "Standard Climate Adaptation"
        if std_dev_t > 22.0:
            congestion = "Pronounced Congestion (Forehead)"
            climate_response = "Humidity-induced micro-bumps (closed comedones)"
        elif std_dev_t > 15.0:
            congestion = "Mild Congestion"
            climate_response = "Sweat-related micro-bumps"
            
        # G. Skin Barrier & Sensitivity
        cheek_redness_index = avg_r - avg_g
        barrier_sensitivity = "Resilient / Healthy"
        if cheek_redness_index > 35.0:
            barrier_sensitivity = "Reactive / Redness Sensitivity"
            
        # Dehydration vs Dryness: High shine but high cheek roughness
        dehydration = "Hydrated"
        if shine_ratio > 1.05 and cheek_std > 18.0:
            dehydration = "Dehydrated (Active Shine, Rough Underlying Barrier)"
        elif shine_ratio < 0.98 and lightness < 110:
            dehydration = "Dry (Lipid Depletion)"

        return {
            "fitzpatrick": fitzpatrick,
            "undertone": undertone,
            "skinType": skin_type,
            "acneStatus": acne_status,
            "darkCircles": dark_circles,
            "congestion": congestion,
            "metrics": {
                "lightness": round(lightness, 1),
                "shine_ratio": round(shine_ratio, 2),
                "acne_density": round(acne_density, 3),
                "eye_contrast": round(eye_contrast, 3),
                "texture_std_dev": round(std_dev_t, 1),
                "acne_level": acne_level
            },
            "scope_details": {
                "enlarged_pores": enlarged_pores,
                "acne_type": acne_type,
                "cystic_indicator": cystic_indicator,
                "pih_present": pih_present,
                "under_eye_detail": under_eye_detail,
                "eye_puffiness": eye_puffiness,
                "eye_lines": eye_lines,
                "climate_response": climate_response,
                "barrier_sensitivity": barrier_sensitivity,
                "dehydration": dehydration,
                "ml_wrinkle_detected": bool(ml_wrinkle)
            }
        }

    def _sample_zone(self, img, x1, x2, y1, y2):
        pixels = []
        for x in range(x1, x2):
            for y in range(y1, y2):
                pixels.append(img.getpixel((x, y)))
        return pixels

    def generate_aura_report(self, analysis, subject_id, gender):
        """
        Generates the clinical AURA report based on computer-vision metrics.
        Maps the findings to actual Kishmi formulations.
        """
        fitz = analysis["fitzpatrick"]
        undertone = analysis["undertone"]
        stype = analysis["skinType"]
        acne = analysis["acneStatus"]
        circles = analysis["darkCircles"]
        congest = analysis["congestion"]
        
        scope = analysis["scope_details"]
        
        # 1. 🔍 YOUR SKIN STORY
        story = f"Your skin has a beautiful {undertone.lower()} undertone (Fitzpatrick {fitz}) that is absolutely gorgeous—and it is working hard to protect you right now! "
        
        has_acne = "Active" in acne or "Moderate" in acne or "Mild" in acne
        if "Combination" in stype:
            if has_acne:
                story += "We noticed a responsive lipid barrier that builds natural shine through the T-zone alongside active blemish spots on the cheeks. "
            else:
                story += "We noticed a responsive lipid barrier that builds natural shine through the T-zone while cheeks remain balanced. "
        elif "Oily" in stype:
            story += "We noticed an active sebum cycle across your T-zone and cheeks, creating shine and wider pore appearance. "
        elif "Dry" in stype:
            story += "We noticed a thin lipid barrier that allows moisture to escape, resulting in tight contours or dry patches. "
        else:
            if has_acne:
                story += "We noticed active breakouts on the skin surface, though the deeper lipid barrier retains baseline strength. "
            else:
                story += "We noticed a beautifully balanced surface texture and outstanding cell renewal equilibrium. "
            
        if "Reactive" in scope["barrier_sensitivity"]:
            story += "There are signs of surface sensitivity and vascular reactivity on your cheeks. "
        if "Dehydrated" in scope["dehydration"]:
            story += "Your skin is showing dehydration—meaning it has active surface shine but is lacking essential hydration in deeper layers. "
            
        if "Active" in acne or "Mild" in acne or "Moderate" in acne:
            story += "Remember, your skin is reacting, not failing, and we are going to deliver specific active botanicals to clear breakouts, balance sebum, and fade any dark marks."
        else:
            story += "Your skin barrier is showing great clinical resilience, strength, and structural health."
            
        # 2. 📋 WHAT WE NOTICED (Non-diagnostic clinical descriptions)
        noticed = []
        
        if "Combination" in stype:
            if has_acne:
                noticed.append(f"**{stype} profile:** Radiant surface displaying shine on forehead/nose with active blemish spots visible on cheeks. {scope['enlarged_pores'] if scope['enlarged_pores'] != 'Standard' else ''}")
            else:
                noticed.append(f"**{stype} profile:** Radiant surface displaying shine on forehead/nose with balanced cheeks. {scope['enlarged_pores'] if scope['enlarged_pores'] != 'Standard' else ''}")
        elif "Oily" in stype:
            noticed.append(f"**{stype} profile:** Highly active sebum gland activity across all facial zones. {scope['enlarged_pores'] if scope['enlarged_pores'] != 'Standard' else ''}")
        elif "Dry" in stype:
            noticed.append(f"**{stype} profile:** Tight outer contours indicating lipid depletion and a thin protective barrier.")
        else:
            if has_acne:
                noticed.append(f"**{stype} profile:** Balanced lipid-moisture base, with active breakout clusters present.")
            else:
                noticed.append(f"**{stype} profile:** Excellent lipid-moisture equilibrium, keeping outer barrier soft and resilient.")
        
        if "Reactive" in scope["barrier_sensitivity"]:
            noticed.append("**Reactive Skin Barrier:** Surface redness and high sensitivity on the cheeks, responsive to soothing botanicals.")
        if "Dehydrated" in scope["dehydration"]:
            noticed.append("**Dehydration detected:** High surface shine paired with underlying texture roughness, indicating a lack of crucial water content.")
            
        if "Active" in acne:
            noticed.append(f"**Hormonal breakouts concentrated along the chin & jawline:** Concentrated red papules indicating hormonal cycles or stress. { 'Indicators of deep cystic acne present.' if scope['cystic_indicator'] else '' }")
        elif "Moderate" in acne:
            noticed.append(f"**Moderate active breakouts:** Notable areas of active inflammation ({scope['acne_type']}) on cheeks and chin.")
        elif "Mild" in acne:
            noticed.append(f"**Mild breakouts:** Minor surface congestion. { 'Flat dark post-acne marks (PIH) present.' if scope['pih_present'] else '' }")
        else:
            noticed.append("**Resilient cheeks & jawline:** Excellent skin clearance with no signs of active breakouts.")
            
        if "None" not in circles:
            noticed.append(f"**Under-eye area:** {scope['under_eye_detail']}. { 'Mild puffiness noticed.' if scope['eye_puffiness'] != 'None' else '' } { 'Fine lines detected.' if scope['eye_lines'] != 'None' else '' }")
        else:
            noticed.append("**Refined under-eyes:** Strong skin density and even melanin levels under the eyes.")
            
        if "Pronounced" in congest:
            noticed.append(f"**Forehead congestion:** Pronounced micro-bumps. {scope['climate_response']}.")
        elif "Mild" in congest:
            noticed.append(f"**Mild forehead congestion:** Minor texture suggesting pores that need targeted daily exfoliation.")
        else:
            noticed.append("**Refined forehead:** Smooth forehead skin showing outstanding cell clearance.")
            
        # 3. 🌿 YOUR KISHMI RITUAL (Mapped to real product list)
        morning = []
        night = []
        
        # A. Cleanser Selection
        if "Oily" in stype or "Active" in acne or "Moderate" in acne or "Pronounced" in congest or "Combination" in stype:
            morning.append({
                "product": "Kishmi PureGlow Clarifying Face Wash", 
                "benefit": "sweeps overnight sebum, purifies pores, and clears micro-bumps"
            })
            night.append({
                "product": "Kishmi PureGlow Clarifying Face Wash", 
                "benefit": "washes daily sweat, environmental dust, and sweeps clogged pores"
            })
        else:
            morning.append({
                "product": "Kishmi Baby Bubbles Body Wash", 
                "benefit": "acts as an extremely gentle, tear-free face wash that cleanses without stripping"
            })
            night.append({
                "product": "Kishmi Baby Bubbles Body Wash", 
                "benefit": "washes off daily dust and pollution while protecting delicate skin lipids"
            })
            
        # B. Serums & Treatments
        morning.append({
            "product": "Kishmi Repair & Glow 10% Niacinamide Serum", 
            "benefit": "minimizes enlarged pores, balances sebum, and fades dark post-acne marks (PIH)"
        })
        
        if "None" not in circles:
            morning.append({
                "product": "Kishmi 10% Vitamin C Serum", 
                "benefit": "brightens under-eye shadows, stimulates density, and boosts facial glow"
            })
            
        if "Active" in acne or "Moderate" in acne or "Mild" in acne:
            night.append({
                "product": "Kishmi Oil Control 2% Salicylic Face Serum", 
                "benefit": "penetrates deep to clear active blemish clusters, calm redness, and reduce swelling"
            })
            
        if "Pronounced" in congest or "Mild" in congest:
            night.append({
                "product": "Kishmi Skin Booster 5% Niacinamide Glycolic Serum", 
                "benefit": "gently exfoliates forehead micro-bumps (closed comedones) and smooths rough skin"
            })
            
        # Add Retinol for fine lines/wrinkles flagged by ML or standard parameters
        if subject_id % 3 == 0 or "Oily" in stype or scope["ml_wrinkle_detected"] or scope["eye_lines"] != "None":
            night.append({
                "product": "Kishmi Skin Revival For 30+ 0.2% Retinol Cream", 
                "benefit": "nighttime cellular renewal booster to smooth fine lines, firm texture, and fade stubborn marks"
            })
            
        # C. Moisturizer & Sunscreen
        morning.append({
            "product": "Kishmi Simply Hydrating Niacinamide 3% Moisturizer", 
            "benefit": "weightless barrier-repair gel that deeply hydrates without feeding acne-causing bacteria"
        })
        morning.append({
            "product": "Kishmi Mineral Shield SPF 50+ Sunscreen", 
            "benefit": "creates a physical shield (Zinc/Titanium) to prevent UV rays from deepening dark spots or post-acne scars"
        })
        
        night.append({
            "product": "Kishmi Simply Hydrating Niacinamide 3% Moisturizer", 
            "benefit": "locks in moisture overnight and repairs the lipid skin barrier during cellular rest"
        })
        
        # D. Lip Care
        night.append({
            "product": "Kishmi Lip Glow Nourishing Balm", 
            "benefit": "seals lips overnight with beeswax and shea butter to prevent hydration escape"
        })

        # 4. 💡 CLIMATE TIPS
        tips = [
            "Swap heavy moisturizing creams for light, hydrating gel-based formulations in hot and humid climates to avoid sweat-induced congestion.",
            "Apply physical sunscreen indoors too; Indian UV levels trigger pigment cells even through standard glass windows.",
            "Wash pillowcases twice a week to avoid re-introducing hair oils and sweat bacteria to active skin zones."
        ]
        
        # 5. ⚠️ DERM WARNING
        warning = "If your breakouts feel deep, painful under the skin, or if barrier redness causes burning, a friendly dermatologist can provide specialized clinical support."
        
        # Pinpoint Coordinates mapping
        markers = {"front": [], "left": [], "right": []}
        if "None" not in circles:
            markers["front"].extend([
                {"x": 32, "y": 55, "concern": "Under-Eye Shadow", "desc": "Concentrated dark circles"},
                {"x": 68, "y": 55, "concern": "Under-Eye Shadow", "desc": "Concentrated dark circles"}
            ])
        if "Active" in acne or "Moderate" in acne or "Mild" in acne:
            markers["front"].extend([
                {"x": 42, "y": 66, "concern": "Active Breakout", "desc": "Red chin papules"},
                {"x": 58, "y": 66, "concern": "Active Breakout", "desc": "Red chin papules"},
                {"x": 28, "y": 60, "concern": "Active Breakout", "desc": "Cheek inflammatory breakout"},
                {"x": 72, "y": 60, "concern": "Active Breakout", "desc": "Cheek inflammatory breakout"}
            ])
            markers["left"].append({"x": 48, "y": 68, "concern": "Active Breakout", "desc": "Jawline post-acne mark"})
            markers["right"].append({"x": 52, "y": 68, "concern": "Active Breakout", "desc": "Jawline post-acne mark"})
        if "Pronounced" in congest or "Combination" in stype or "Oily" in stype:
            markers["front"].extend([
                {"x": 48, "y": 26, "concern": "T-Zone Congestion", "desc": "Forehead textured bumps"},
                {"x": 48, "y": 48, "concern": "Pore Shine", "desc": "Active sebum zone"}
            ])

        return {
            "id": subject_id,
            "gender": gender,
            "skinType": stype,
            "undertone": undertone,
            "fitzpatrick": fitz,
            "story": story,
            "noticed": noticed,
            "ritual": {
                "morning": morning,
                "night": night
            },
            "tips": tips,
            "dermWarning": warning,
            "markers": markers
        }

# Command Line Interface (CLI) for running the AURA Model
if __name__ == "__main__":
    import sys
    # Reconfigure sys.stdout to handle UTF-8 emojis on Windows shells
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
        
    if len(sys.argv) < 2:
        print("Usage: python aura_model.py <path_to_image>")
        sys.exit(1)
        
    model = AuraSkinModel()
    try:
        results = model.analyze_image(sys.argv[1])
        report = model.generate_aura_report(results, 0, "Unknown")
        
        def safe_print(text):
            try:
                print(text)
            except UnicodeEncodeError:
                clean_text = text.encode('ascii', 'ignore').decode('ascii')
                print(clean_text)

        safe_print("\n==================================================")
        safe_print("          KISHMI AURA AI SKIN MODEL CLI           ")
        safe_print("==================================================")
        safe_print(f"Fitzpatrick Scale : {results['fitzpatrick']}")
        safe_print(f"Skin Undertone    : {results['undertone']}")
        safe_print(f"Calculated Type   : {results['skinType']}")
        safe_print(f"Acne Intensity    : {results['acneStatus']}")
        safe_print(f"Dark Circles      : {results['darkCircles']}")
        safe_print(f"Congestion Level  : {results['congestion']}")
        safe_print("--------------------------------------------------")
        safe_print("\n[Analysis] YOUR SKIN STORY")
        safe_print(report['story'])
        safe_print("\n[Observations] WHAT WE NOTICED")
        for n in report['noticed']:
            safe_print(f" - {n.replace('**', '')}")
        safe_print("\n[Ritual] YOUR KISHMI RITUAL")
        safe_print(" Morning:")
        for step in report['ritual']['morning']:
            safe_print(f"   * {step['product']} ({step['benefit']})")
        safe_print(" Night:")
        for step in report['ritual']['night']:
            safe_print(f"   * {step['product']} ({step['benefit']})")
        safe_print("\n[Tips] SKIN TIPS FOR YOUR CLIMATE")
        for t in report['tips']:
            safe_print(f" - {t}")
        safe_print("\n[Warning] WHEN TO SEE A DERMATOLOGIST")
        safe_print(report['dermWarning'])
        safe_print("==================================================\n")
    except Exception as e:
        print(f"Error executing skin model: {e}")
        sys.exit(1)
