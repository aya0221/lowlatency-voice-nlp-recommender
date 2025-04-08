import json
from opensearchpy import OpenSearch

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))
from voice_assistant.utils.config import OPENSEARCH_HOST

INDEX_NAME = "workouts"

# Connect to OpenSearch
client = OpenSearch(hosts=[OPENSEARCH_HOST])

# Load data
with open("voice_assistant/data/workouts.json") as f:
    workouts = json.load(f)

# Create index (if doesn't exist)
if not client.indices.exists(INDEX_NAME):
    client.indices.create(index=INDEX_NAME)

# Index data
for i, doc in enumerate(workouts):
    client.index(index=INDEX_NAME, id=i, body=doc)

print(f"Indexed {len(workouts)} workouts.")
