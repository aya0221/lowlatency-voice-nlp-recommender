import json
from pathlib import Path
import sys

# Dynamically determine the project root and add it to sys.path
project_root = Path(__file__).resolve().parent.parent
print('project_root:', project_root)
sys.path.append(str(project_root))

from voice_assistant.utils import config

with open("voice_assistant/data/workouts.json", "r") as f:
    data = json.load(f)

print(f"Loaded {len(data)} workouts.")
print("OpenSearch host:", config.OPENSEARCH_HOST)

