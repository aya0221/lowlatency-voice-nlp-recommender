import sys
from pathlib import Path
import os
import json
import spacy
from transformers import pipeline, DistilBertForSequenceClassification, DistilBertTokenizerFast

# === Setup path ===
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))
print(f"[INFO] Project root: {project_root}")

from voice_assistant.nlu.custom_entity_extractor import keyword_matcher
from voice_assistant.utils import config
from voice_assistant.search.search_workouts import search_workouts
from voice_assistant.asr.record_and_transcribe import record_and_transcribe
# === Load label map ===
MODEL_DIR = os.path.join(project_root, "voice_assistant/models/intent_classifier")
with open(os.path.join(MODEL_DIR, "label_map.json")) as f:
    label_to_id = json.load(f)
id_to_label = {v: k for k, v in label_to_id.items()}

# === Load intent classifier ===
print("[INFO] Loading intent classifier...")
model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
intent_classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, top_k=1)

# === Load NER model ===
print("[INFO] Loading spaCy NER pipeline...")
nlp = spacy.load(config.SPACY_MODEL)

# === Entity Extraction ===
def extract_entities(text):
    print(f"[INFO] Extracting entities from: {text}")
    doc = nlp(text)
    entities = {}

    for ent in doc.ents:
        if ent.label_ == "TIME":
            entities["time"] = ent.text
        elif ent.label_ == "PERSON":
            entities["person"] = ent.text

    custom_entities = keyword_matcher(text)
    entities.update(custom_entities)

    print("[INFO] Extracted entities:")
    for key, value in entities.items():
        print(f"    - {key}: {value}")
    return entities

# === Intent Detection ===
def detect_intent(text: str) -> str:
    result = intent_classifier(text)

    if isinstance(result, list) and len(result) > 0:
        top = result[0][0] if isinstance(result[0], list) else result[0]
        raw_label = top["label"]
        score = top["score"]
        label_id = int(raw_label.split("_")[-1])
        intent = id_to_label.get(label_id, "unknown")
        print(f"[INFO] Detected intent: {intent} (confidence={score:.2f})")
        return intent

    raise ValueError("Unexpected output from intent classifier")

# === Full Pipeline ===
def parse_text(text: str) -> dict:
    print("\n================ NLP PIPELINE START ================\n")
    print(f"[INFO] User said: {text}")
    intent = detect_intent(text)
    entities = extract_entities(text)
    result = {"intent": intent, "entities": entities}
    print("\n[RESULT] Parsed result:")
    print(result)
    print("\n================ NLP PIPELINE END ==================\n")
    return result

# === Run ===
if __name__ == "__main__":
    from voice_assistant.asr.transcribe import transcribe_audio
    sample = record_and_transcribe()
    parsed = parse_text(sample)

    if parsed["intent"] == "search_class":
        results = search_workouts(parsed["intent"], parsed["entities"])
        print("[INFO] Top matches:")
        for res in results:
            print(res)
