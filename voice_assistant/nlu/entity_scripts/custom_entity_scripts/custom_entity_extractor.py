import re

# --- Rule-based lookup tables ---
WORKOUT_TYPES = [
    "yoga", "meditation", "cycling", "bike", "ride", "run", "treadmill", "strength",
    "cardio", "HIIT", "stretching", "pilates", "walking", "bootcamp",
]
GOALS = [
    "build muscle", "lose weight", "fat burn", "mobility", "flexibility",
    "get stronger", "tone body", "calm mind", "energy boost", "stress relief"
]
INTENSITIES = ["low impact", "high intensity", "beginner", "intermediate", "advanced"]

def keyword_matcher(text: str) -> dict:
    """Extracts custom entities by scanning for keywords."""
    text_lower = text.lower()
    entities = {}

    for word in WORKOUT_TYPES:
        if word in text_lower:
            entities["workout_type"] = word
            break

    for word in GOALS:
        if word in text_lower:
            entities["goal"] = word
            break

    for word in INTENSITIES:
        if word in text_lower:
            entities["intensity"] = word
            break

    return entities

