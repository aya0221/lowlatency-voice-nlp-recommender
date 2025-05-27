# Entity Extraction Model — RoBERTa + spaCy Transformer (Production-Grade)

This folder contains the fine-tuned spaCy NER pipeline for structured entity extraction from voice-transcribed fitness queries. The model uses a RoBERTa backbone and is trained on over 30,000 ASR-style user queries for real-world robustness.

---

## Extracted Entities

- `WORKOUT_TYPE` — e.g., yoga, spin, running  
- `DURATION` — e.g., 20 minutes, 45 min  
- `INSTRUCTOR` — e.g., Alex, Cody  
- `INTENSITY` — e.g., high intensity, beginner  
- `GOAL` — e.g., weight loss, core strength  
- `MOOD` — e.g., relaxing, energizing  

---

## Architecture Summary

- **Backbone**: `roberta-base` via `spacy-transformers`
- **Embedding Layer**: Transformer subwords pooled to spaCy tokens using `reduce_mean.v1`
- **NER Decoder**: `TransitionBasedParser.v2`
- **Pipeline**: `["transformer", "ner"]`
- **Mixed Precision**: Enabled
- **Trained with**: MLflow + GPU (CUDA)

---

## Performance

| Metric           | Value     |
|------------------|-----------|
| Overall F1       | **99.97%** |
| Token Accuracy   | 100%      |
| GPU Used         | 6GB CUDA  |
| Inference Speed  | ~5950 tokens/sec |

Entity-wise F1 (dev set):

- `WORKOUT_TYPE`: 100.00%  
- `INSTRUCTOR`: 99.97%  
- `INTENSITY`: 99.86%  
- `GOAL`: 100.00%  
- `DURATION`: 99.97%  
- `MOOD`: 99.90%

---

## Training Config (config.cfg)

- **Optimizer**: `Adam` with `L2=0.01`, `grad_clip=1.0`
- **Learning Rate**: `1e-4`
- **Dropout**: `0.15`
- **Max Epochs**: `600` with early stopping
- **Batcher**: `batch_by_words.v1` with size=2048, tolerance=0.2
- **Evaluation Frequency**: every 200 steps
- **Logger**: `MLflowLogger.v1`

---

## Directory Structure

```

entity\_model/
└── model-best/
├── config.cfg            # Full training configuration
├── meta.json             # Metadata and final eval scores
├── ner/
│   ├── cfg               # NER-specific transition config
│   └── moves             # Transition graph for entities
├── transformer/          # RoBERTa transformer weights
├── tokenizer/            # Tokenizer configs
└── vocab/                # Vocab + lookup tables

````

---
## Example Usage

### Recommended: Load Directly from Hugging Face

No setup needed — just install dependencies and load the model directly from Hugging Face:

```bash
pip install spacy spacy-transformers
````

```python
import spacy
nlp = spacy.load("Aya-In-Brooklyn/spaCy-roberta-workout-entity-model")

doc = nlp("Give me a relaxing 30 minute yoga with Cody.")
for ent in doc.ents:
    print(ent.text, ent.label_)
```

Expected Output:

```
relaxing      -> MOOD  
30 minute     -> DURATION  
yoga          -> WORKOUT_TYPE  
Cody          -> INSTRUCTOR  
```

---

### Local Load (time-consuming)

If you have the full model weights downloaded locally:

```python
nlp = spacy.load("voice_assistant/models/entity_model/model-best")
```
> This will only work if you have the full `model/`, `transformer/`, and `tokenizer/` subfolders locally.
> These files are `.gitignored` and not included in this github repository.

---

## Hugging Face Hub

Hosted model with full weights + documentation:
 [Aya-In-Brooklyn/spaCy-roberta-workout-entity-model](https://huggingface.co/Aya-In-Brooklyn/spaCy-roberta-workout-entity-model)

---

## Need Local Weights?

If you need the full model files for offline use, either:

* Download from Hugging Face:

  ```bash
  git lfs install
  git clone https://huggingface.co/Aya-In-Brooklyn/spaCy-roberta-workout-entity-model
  ```

* Or contact: [ayaoshima.us@gmail.com](mailto:ayaoshima.us@gmail.com)
---

## Notes

* This model is trained purely on labeled span data, no rule-based logic is hardcoded.
* Optimized for voice-assistant scenarios with noisy ASR transcriptions.
* Reproducible via MLflow experiment logs (run tracked locally).

---

## Files Under Git

This folder only tracks metadata and configs — model weights are `.gitignored`.
Trackable files include:

* `config.cfg`, `meta.json`, `ner/cfg`, `ner/moves`, `README.md`, `.gitignore`