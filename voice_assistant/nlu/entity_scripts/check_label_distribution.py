import sys
from collections import Counter
import json
import os
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(project_root))

DATA_DIR = os.path.join(project_root, "voice_assistant/data/entity_data")
with open(os.path.join(DATA_DIR, "train.json")) as f:
    data = json.load(f)

counts = Counter()
for example in data:
    for _, _, label in example["entities"]:
        counts[label] += 1

print("Label counts:", counts)
