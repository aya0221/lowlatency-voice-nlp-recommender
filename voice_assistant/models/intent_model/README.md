# Intent Classifier — DistilBERT Fine-Tuned

Fine-tuned `distilbert-base-uncased` model for classifying voice-based workout queries into high-level intents.

Trained on a labeled dataset of voice assistant utterances, this model enables fast and accurate routing of user input in a low-latency recommendation system.

---

## Model Overview

- **Model**: DistilBERT (`distilbert-base-uncased`)
- **Architecture**: `DistilBertForSequenceClassification`
- **Task**: Single-label intent classification
- **Labels**:
  - `track_metric`
  - `greeting`
  - `search_class`
- **Base Config**: 6 layers, 12 heads, 768 hidden dim
- **Training Framework**: Hugging Face Transformers

---

## Training Details

- **Epochs**: 20
- **Batch Size**: 16
- **Best Checkpoint**: `checkpoint-47`
  - Selected based on highest evaluation accuracy (`eval_accuracy = 1.0`)
  - Logged at step 47 / epoch 1.0
- **Loss Function**: Cross-Entropy
- **Optimizer**: AdamW
- **Learning Rate**: 1e-5
- **Dropout**: 0.1 (attention), 0.2 (classification head)

---

## File Structure

```

intent\_model/
├── checkpoint-47/              # Best model weights
│   ├── config.json             # Model architecture config
│   ├── model.safetensors       # Weights
│   ├── tokenizer.json          # Tokenizer config
│   └── trainer\_state.json      # Training logs and best checkpoint info
├── config.json                 # HF Transformers config
├── tokenizer\_config.json
├── tokenizer.json
├── special\_tokens\_map.json
├── vocab.txt
├── label\_map.json              # Maps class labels ↔ integers
└── README.md

````

---

## Example Usage

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load fine-tuned model
model_dir = "voice_assistant/models/intent_model/checkpoint-47"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)

# Run inference
text = "Find me a 30-minute yoga session"
inputs = tokenizer(text, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits

# Get predicted intent
predicted_class = logits.argmax().item()

# Label mapping (from label_map.json)
label_map = {0: "track_metric", 1: "greeting", 2: "search_class"}
print("Predicted intent:", label_map[predicted_class])
````

**Expected Output:**

```
Predicted intent: search_class
```

---

## Notes

* This model was fine-tuned specifically for voice-controlled workout assistant commands.
* All training and evaluation steps were tracked for reproducibility.
* Only `checkpoint-47` is used for inference; later checkpoints did not surpass its evaluation accuracy.

---