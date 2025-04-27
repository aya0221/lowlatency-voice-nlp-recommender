import spacy
from spacy.tokens import DocBin
from spacy.training.example import Example
from spacy.util import minibatch
import random
import json
from pathlib import Path

# === Paths ===
DATA_PATH = Path("voice_assistant/data/ner_training_data.json")
DOCBIN_PATH = Path("voice_assistant/data/ner_data.spacy")
MODEL_OUTPUT_DIR = Path("voice_assistant/models/custom_ner")

# === Load raw JSON training data ===
with open(DATA_PATH, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# === Build DocBin (for future reuse) ===
nlp_blank = spacy.blank("en")
doc_bin = DocBin()

for record in raw_data:
    doc = nlp_blank.make_doc(record["text"])
    ents = []
    for start, end, label in record["entities"]:
        span = doc.char_span(start, end, label=label)
        if span is None:
            print(f"[WARN] Skipped misaligned span: {record['text'][start:end]} ({label})")
            continue
        ents.append(span)
    doc.ents = ents
    doc_bin.add(doc)

doc_bin.to_disk(DOCBIN_PATH)
print(f"[INFO] Saved DocBin to: {DOCBIN_PATH}")

# === Training the model ===
nlp = spacy.blank("en")
ner = nlp.add_pipe("ner")

# Add entity labels
for entry in raw_data:
    for _, _, label in entry["entities"]:
        ner.add_label(label)

# Initialize optimizer
optimizer = nlp.begin_training()

# Prepare training examples
examples = []
for record in raw_data:
    doc = nlp.make_doc(record["text"])
    spans = [
        doc.char_span(start, end, label=label)
        for start, end, label in record["entities"]
        if doc.char_span(start, end, label=label)
    ]
    example = Example.from_dict(doc, {"entities": [(s.start_char, s.end_char, s.label_) for s in spans]})
    examples.append(example)

# === Training loop ===
NUM_EPOCHS = 20
BATCH_SIZE = 32

for epoch in range(NUM_EPOCHS):
    random.shuffle(examples)
    losses = {}
    batches = minibatch(examples, size=BATCH_SIZE)
    for batch in batches:
        nlp.update(batch, sgd=optimizer, losses=losses)
    print(f"[INFO] Epoch {epoch+1} Loss: {losses.get('ner', 0.0)}")

# === Save model ===
nlp.to_disk(MODEL_OUTPUT_DIR)
print(f"[INFO] Model saved to {MODEL_OUTPUT_DIR}")
