# 📦 Custom NER Model (Predecessor to spaCy-RoBERTa)

This model is an **older, lightweight spaCy NER pipeline** trained before the Roberta+spaCy Transformer upgrade  [`entity_model`](../entity_model).  
It uses `HashEmbedCNN` embeddings and a `TransitionBasedParser`, with **no transformer backbone**, no pretraining, and no MLflow tracking.


---

## 🧪 Example Usage

```python
import spacy
nlp = spacy.load("voice_assistant/models/custom_ner")

doc = nlp("Give me a high intensity strength workout with Cody at 7am for weight loss.")
for ent in doc.ents:
    print(ent.text, ent.label_)
````

Expected Output:

```
high intensity     -> INTENSITY  
strength workout   -> WORKOUT_TYPE  
Cody               -> PERSON  
7am                -> TIME  
weight loss        -> GOAL  
```

