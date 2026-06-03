def get_recommendations(condition_scores):
    """
    Takes a dictionary of skin conditions and their severity scores (0-5).
    Returns a list of recommended products based on the custom product list.
    """
    
    # Base routine products
    recommendations = {
        "FACEWASH 100 ML": "Step 1: Cleanse",
        "MOISTURISER 100 ML": "Step 3: Moisturize",
        "SPF 50 ML": "Step 4: Sun Protection"
    }
    
    # Targeted treatments (Step 2) based on highest severity
    highest_severity = max(condition_scores.values()) if condition_scores else 0
    
    if highest_severity >= 2:
        # Prioritize conditions if multiple have high severity
        # You could refine this logic based on dermatologist advice
        
        if condition_scores.get('Acne', 0) >= 2:
            recommendations["2% SALICYLIC ACID SERUM"] = "Step 2: Targeted Acne Treatment"
            
        if condition_scores.get('Acne Marks', 0) >= 2:
            if "10% NIACINAMIDE FACE SERUM 30 ML" not in recommendations:
                recommendations["10% NIACINAMIDE FACE SERUM 30 ML"] = "Step 2: Dark Spot / Mark Fading"
                
        if condition_scores.get('Pigmentation spots', 0) >= 2:
            recommendations["5% GLYCOLIC ACID + 5% NIACINAMIDE FACE SERUM"] = "Step 2: Exfoliation and Brightening"
            recommendations["SKIN WHITENING CREAM 50 GM"] = "Step 3: Pigmentation Repair Cream"
            
        if condition_scores.get('Wrinkles', 0) >= 2:
            recommendations["ANTI AGING CREAM 50 ML"] = "Step 3: Anti-Aging Treatment"
            
        if condition_scores.get('Dark circles', 0) >= 2:
            recommendations["10% VIT C + COMBINATION FACE SERUM 30 ML"] = "Step 2: Under Eye Brightening"

    return recommendations

if __name__ == "__main__":
    # Test example
    sample_patient = {
        "Acne": 3,
        "Acne Marks": 1,
        "Pigmentation spots": 0,
        "Wrinkles": 2,
        "Dark circles": 0
    }
    
    print("Detected Condition Scores:", sample_patient)
    print("\nRecommended Routine:")
    routine = get_recommendations(sample_patient)
    for product, step in routine.items():
        print(f"- {step}: {product}")
