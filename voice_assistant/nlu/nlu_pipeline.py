import sys
from pathlib import Path
import os
import json
import spacy
from transformers import pipeline, DistilBertForSequenceClassification, DistilBertTokenizerFast
import argparse
# === Setup project root ===
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from voice_assistant.nlu.custom_entity_scripts.custom_entity_extractor import keyword_matcher
from voice_assistant.utils import config
from voice_assistant.search.search_workouts import search_workouts
from voice_assistant.asr.record_and_transcribe import record_and_transcribe
from voice_assistant.asr.transcribe import transcribe_audio

# === Record user input ===
# speech_input = record_and_transcribe()

# === Load models ===
print("[INFO] Loading intent classifier (fine-tuned DistilBERT model)...")
MODEL_DIR = os.path.join(project_root, "voice_assistant/models/intent_model")
with open(os.path.join(MODEL_DIR, "label_map.json")) as f:
    label_to_id = json.load(f)
id_to_label = {v: k for k, v in label_to_id.items()}

model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
intent_classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, top_k=1)

print(f"[INFO] Loading spaCy NER pipeline ({config.SPACY_MODEL})...")
nlp = spacy.load(config.SPACY_MODEL)

# === Entity Extraction ===
def extract_entities(text):
    print(f"[INFO] Extracting entities from: {text}")
    doc = nlp(text)
    entities = {}

    for ent in doc.ents:
        label = ent.label_.lower()
        entities[label] = ent.text

        # comment below when my entity model is fine-tuned.
        # if ent.label_ == "TIME":
        #     entities["time"] = ent.text
        # elif ent.label_ == "PERSON":
        #     entities["person"] = ent.text

    # custom_entities = keyword_matcher(text)
    # entities.update(custom_entities)

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
        print(f"[INFO] Detected intent: {intent}")
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
    return result

def run_pipeline(transcript: str):
    intent = detect_intent(transcript)
    entities = extract_entities(transcript)
    return {"intent": intent, "entities": entities}

def parse_text(text: str):
    print(f"\n[INFO] User said: {text}")
    parsed = run_pipeline(text)
    print(f"[RESULT] Parsed: {parsed}")
    return parsed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", action="store_true", help="Run end-to-end pipeline on CLI")
    args = parser.parse_args()

    if args.cli:
        speech_input = record_and_transcribe()
        parsed = parse_text(speech_input)

        print("\n================ SEARCH & RECOMMENDATION START ==================\n")
        if parsed["intent"] == "search_class":
            top_k = 10
            results = search_workouts(parsed["intent"], parsed["entities"], top_k=top_k)
            print("[INFO] Top {} recommended classes (ranked by relevance score):".format(top_k))
            for res in results:
                print(
                    f"- [{res['score']}] {res['title']} | {res['duration']} min | {res['instructor']} | {res['intensity']} | {res['type']}"
                )


# === Run Pipeline ===
# if __name__ == "__main__":
#     parsed = parse_text(speech_input)

#     print("\n================ SEARCH & RECOMMENDATION START ==================\n")
#     if parsed["intent"] == "search_class":
#         top_k = 10
#         results = search_workouts(parsed["intent"], parsed["entities"], top_k=top_k)
#         print("[INFO] Top {} recommended classes (ranked by relevance score):".format(top_k))
#         for res in results:
#             print(
#                 f"- [{res['score']}] {res['title']} | {res['duration']} min | {res['instructor']} | {res['intensity']} | {res['type']}"
#             )
