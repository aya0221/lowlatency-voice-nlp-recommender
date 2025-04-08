import re
from opensearchpy import OpenSearch
from voice_assistant.utils.config import OPENSEARCH_HOST
from word2number import w2n

INDEX_NAME = "workouts"
client = OpenSearch(hosts=[OPENSEARCH_HOST])

# Normalize synonyms to consistent internal representation
WORKOUT_TYPE_SYNONYMS = {
    "bike": "cycling",
    "ride": "cycling",
    "spin": "cycling",
    "cycling": "cycling",
    "run": "running",
    "jog": "running",
    "walk": "walking",
    "stretching": "stretch",
    "strength training": "strength",
    "hiit": "hiit",
    "yoga": "yoga",
    "meditation": "meditation",
    "pilates": "pilates"
}

# Map goals to multiple relevant tags
GOAL_TO_TAGS = {
    "weight loss": ["weight loss", "cardio"],
    "relax": ["relaxing", "mood"],
    "flexibility": ["flexibility", "yoga"],
    "strength": ["strength", "core"],
    "endurance": ["endurance", "running", "cycling"],
    "mood": ["mood", "relaxing"]
}

def extract_minutes(time_str):
    """Handles numeric or text-based durations."""
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
    """Normalizes workout type using synonyms."""
    if "workout_type" in entities:
        raw = entities["workout_type"].lower()
        entities["workout_type"] = WORKOUT_TYPE_SYNONYMS.get(raw, raw)
    return entities

def search_workouts(intent: str, entities: dict, top_k: int = 5):
    entities = normalize_entities(entities)
    must_clauses = []
    should_clauses = []

    # Time
    minutes = extract_minutes(entities.get("time", ""))
    if minutes:
        must_clauses.append({
            "range": {
                "duration": {
                    "gte": max(minutes - 5, 5),
                    "lte": minutes + 5
                }
            }
        })

    if "person" in entities:
        must_clauses.append({"match": {"instructor": entities["person"]}})
    if "intensity" in entities:
        must_clauses.append({"match": {"intensity": entities["intensity"]}})
    if "workout_type" in entities:
        must_clauses.append({"match": {"type": entities["workout_type"]}})

    # Goal → tags
    if "goal" in entities:
        goal = entities["goal"].lower()
        tag_list = GOAL_TO_TAGS.get(goal, [goal])
        for tag in tag_list:
            must_clauses.append({"match": {"tags": tag}})
    else:
        # If no goal, expand with general relevance tags
        for tag_list in GOAL_TO_TAGS.values():
            for tag in tag_list:
                should_clauses.append({"match": {"tags": tag}})

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
    return [hit["_source"] for hit in response["hits"]["hits"]]
