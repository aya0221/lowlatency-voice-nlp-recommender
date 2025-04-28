import re
from opensearchpy import OpenSearch
from voice_assistant.utils.config import OPENSEARCH_HOST
from word2number import w2n

INDEX_NAME = "workouts"
client = OpenSearch(hosts=[OPENSEARCH_HOST])

# === Synonym Normalization Map ===
WORKOUT_TYPE_SYNONYMS = {
    "bike": "cycling", "ride": "cycling", "spin": "cycling", "cycling": "cycling",
    "run": "running", "jog": "running", "walk": "walking",
    "stretch": "stretching", "strength training": "strength",
    "hiit": "hiit", "yoga": "yoga", "meditation": "meditation", "pilates": "pilates"
}
INTENSITY_SYNONYMS = {
    "easy": "low impact",
    "light": "low impact",
    "beginner": "low impact",
    "gentle": "low impact",
    "moderate": "moderate",
    "intermediate": "moderate",
    "medium intensity": "moderate",
    "high": "high intensity",
    "challenging": "high intensity",
    "advanced": "high intensity",
    "tough": "high intensity",
    "super intense": "high intensity",
    "killer": "high intensity",
    "hardcore": "high intensity",
}
# === Goal → Tag expansion ===
GOAL_TO_TAGS = {
    "weight loss": ["weight loss", "cardio"],
    "relax": ["relaxing", "mood"],
    "flexibility": ["flexibility", "yoga"],
    "strength": ["strength", "core"],
    "endurance": ["endurance", "running", "cycling"],
    "mood": ["mood", "relaxing"]
}


# === Utilities ===
def extract_minutes(time_str):
    """Convert textual or numeric time expression to integer minutes."""
    if not time_str:
        return None
    match = re.search(r"\d+", time_str)
    if match:
        return int(match.group())
    try:
        return w2n.word_to_num(time_str)
    except Exception:
        return None

def normalize_entities(entities):
    """Normalize synonyms for consistent filtering."""
    if "workout_type" in entities:
        raw = entities["workout_type"].lower()
        entities["workout_type"] = WORKOUT_TYPE_SYNONYMS.get(raw, raw)

    if "intensity" in entities:
        raw = entities["intensity"].lower()
        entities["intensity"] = INTENSITY_SYNONYMS.get(raw, raw)

    return entities
    
# === Main Search Logic ===
def search_workouts(intent: str, entities: dict, top_k: int = 10):
    entities = normalize_entities(entities)
    must_clauses = []
    should_clauses = []

    # Duration filtering with tolerance range ±5 min
    minutes = extract_minutes(entities.get("duration", ""))
    if minutes:
        # Range match to allow a tolerance window
        must_clauses.append({
            "range": {
                "duration": {
                    "gte": max(minutes - 5, 5),
                    "lte": minutes + 5
                }
            }
        })

        # Exact match as additional scoring boost
        should_clauses.append({
            "match": {
                "duration": {
                    "query": minutes,
                    "boost": 10  # Boost exact time match to top
                }
            }
        })

    if "instructor" in entities:
        must_clauses.append({
            "match": {
                "instructor": {
                    "query": entities["instructor"],
                    "boost": 7
                }
            }
        })

    if "intensity" in entities:
        must_clauses.append({
            "match": {
                "intensity": {
                    "query": entities["intensity"],
                    "boost": 2.5
                }
            }
        })

    if "workout_type" in entities:
        must_clauses.append({
            "match": {
                "type": {
                    "query": entities["workout_type"],
                    "boost": 6
                }
            }
        })

    # Expand goal into relevant tags
    if "goal" in entities:
        tag_list = GOAL_TO_TAGS.get(entities["goal"].lower(), [entities["goal"]])
        for tag in tag_list:
            must_clauses.append({
                "match": {
                    "tags": {
                        "query": tag,
                        "boost": 2
                    }
                }
            })
    else:
        # Fallback: widen tag relevance if no goal is provided
        for tag_list in GOAL_TO_TAGS.values():
            for tag in tag_list:
                should_clauses.append({
                    "match": {
                        "tags": {
                            "query": tag,
                            "boost": 1
                        }
                    }
                })

    query = {
        "size": top_k,
        "query": {
            "bool": {
                "must": must_clauses,
                "should": should_clauses,
                "minimum_should_match": 0
            }
        }
    }

    response = client.search(index=INDEX_NAME, body=query)

    # Return full metadata + score for visibility
    return [
        {
            **hit["_source"],
            "score": round(hit["_score"], 2)
        }
        for hit in response["hits"]["hits"]
    ]
