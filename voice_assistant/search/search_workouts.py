from opensearchpy import OpenSearch
from voice_assistant.utils.config import OPENSEARCH_HOST

INDEX_NAME = "workouts"
client = OpenSearch(hosts=[OPENSEARCH_HOST])

def search_workouts(intent: str, entities: dict, top_k: int = 5):
    must_clauses = []

    if "person" in entities:
        must_clauses.append({"match": {"instructor": entities["person"]}})
    if "time" in entities:
        # Parse minutes
        minutes = int(entities["time"].split()[0])
        must_clauses.append({"match": {"duration": minutes}})
    if "intensity" in entities:
        must_clauses.append({"match": {"intensity": entities["intensity"]}})
    if "workout_type" in entities:
        must_clauses.append({"match": {"type": entities["workout_type"]}})
    if "goal" in entities:
        must_clauses.append({"match": {"tags": entities["goal"]}})

    query = {
        "size": top_k,
        "query": {
            "bool": {
                "must": must_clauses
            }
        }
    }

    results = client.search(index=INDEX_NAME, body=query)
    return [hit["_source"] for hit in results["hits"]["hits"]]
