# Voice-Driven AI Workout Assistant

A production-grade, voice-enabled NLP assistant that transcribes real-time audio, extracts structured workout intents and entities, and returns personalized recommendations via OpenSearch.

---

## Overview

This project demonstrates an end-to-end machine learning pipeline that takes spoken user input and delivers personalized workout recommendations. The architecture is modular, production-ready, and optimized for accuracy, latency, and future deployment.

### Pipeline

```
Voice Input
→ Whisper (ASR)
→ Intent Classification (Fine-Tuned DistilBERT)
→ Named Entity Recognition (spaCy + rule-based matcher)
→ OpenSearch Query (boosted relevance)
→ Top-K Workout Recommendations
```

---

## Key Features

- **Speech-to-Text (Whisper ASR)**  
  Real-time voice transcription using OpenAI’s Whisper, optimized for natural commands and multi-accent robustness.

- **Intent Classification (Fine-Tuned DistilBERT)**  
  Classifies user intent (e.g., `search_class`, `track_metric`) using a transformer model fine-tuned on 1000+ examples. Training used GPU acceleration, hyperparameter tuning, and early stopping to maximize confidence.

- **Entity Extraction (spaCy + Rule-based)**  
  Combines spaCy’s statistical NER with a custom rule-based keyword matcher to extract fields like workout type, duration, goal, instructor, and intensity.

- **Search & Recommendation (OpenSearch)**  
  Indexed workout metadata is queried using relevance boosting (type, time, instructor, tags). Returns top-K personalized matches with similarity scoring.

- **Modular Architecture**  
  ASR, NLU, and Search are decoupled into clean, testable modules ready for production or further research iterations.

---

## Technologies Used

- Python 3.11
- HuggingFace Transformers (`distilbert-base-uncased`)
- PyTorch (fine-tuning with GPU)
- spaCy (NER)
- OpenAI Whisper (ASR)
- OpenSearch (search engine)
- Docker (OpenSearch cluster)
- GPU for training (NVIDIA RTX 4050)

---

## Pipeline Demo

Here’s the assistant in action, showing voice input → NLP extraction → real-time recommendations.

> Example voice input:  
> _“Find me a 20-minute yoga with Alex”_

### 🎥 Demo Video

[![Watch the demo](assets/demo_thumbnail.png)](assets/demo.mp4)

![Demo Output](./assets/demo_pipeline_output.png)

---

## Project Structure

```bash
voice_assistant/
├── asr/                    # Whisper ASR pipeline
│   ├── transcribe.py
│   └── record_and_transcribe.py
├── data/                   # Audio inputs + training CSVs
├── models/                 # Fine-tuned model weights
├── nlu/                    # Intent & Entity logic
│   ├── train_intent_classifier.py
│   ├── nlu_pipeline.py
│   └── custom_entity_extractor.py
├── search/                 # Indexing + Query with OpenSearch
│   ├── index_workouts.py
│   └── search_workouts.py
└── utils/                  # Configs
```

---

## Installation and Usage

### Environment Setup

```bash
git clone https://github.com/aya0221/voice-ai-workout-assistant.git
cd voice-ai-workout-assistant
python -m venv venv-voice-ai-coach
source venv-voice-ai-coach/bin/activate
pip install -r requirements.txt
```

### Run End-to-End Assistant

```bash
python voice_assistant/nlu/nlu_pipeline.py
```

This runs: Voice input → Transcription → Intent + Entity Extraction → OpenSearch Query → Workout Recommendation

---

## Intent Classification (Fine-Tuned DistilBERT)

The system uses a fine-tuned transformer to classify user intent with high precision.

| Parameter         | Value                           |
|------------------|----------------------------------|
| Model             | DistilBERT (fine-tuned)         |
| Training Set      | 1000+ labeled utterances        |
| Epochs            | 20                              |
| Batch Size        | 16 (train) / 8 (eval)           |
| Learning Rate     | 2e-5 with warmup                |
| Hardware          | RTX 4050                        |
| Best Checkpoint   | Step 47 (early convergence)     |
| Final Accuracy    | 100% (held-out set)             |
| Final Loss        | 1.60 → 0.0005                    |

The best model was automatically selected using `load_best_model_at_end=True`, restoring checkpoint-47 based on highest validation accuracy.

---

## Entity Recognition (spaCy + Rules)

A hybrid NER pipeline extracts structured metadata from natural language:

| Entity        | Example           |
|---------------|-------------------|
| `time`        | “30 minutes”       |
| `intensity`   | “low impact”       |
| `person`      | “Robin”            |
| `workout_type`| “yoga”, “ride”     |
| `goal`        | “lose weight”      |
| `tags`        | “cardio”, “relax”  |

This enables coverage of both traditional ML-extracted entities and custom business domain keywords.

---

## Search & Recommendation (OpenSearch)

### Index Workouts

```bash
docker-compose up -d          # Start OpenSearch
python voice_assistant/search/index_workouts.py
```

### Retrieval Logic

- Relevance boosting on instructor, duration, intensity, type
- Synonym normalization (e.g., “bike”, “ride” → “cycling”)
- Duration range matching (±5 mins)
- Goal-to-tag mapping (e.g., “lose weight” → “cardio”, “fat burn”)

Results are ranked by OpenSearch `_score` and returned in descending order.

---

## Model Training

To fine-tune the intent classifier on new data:

```bash
python voice_assistant/nlu/train_intent_classifier.py
```

Outputs:

- `model.safetensors` – final model
- `tokenizer/`, `label_map.json`
- Trainer logs printed to console or MLflow (if enabled)

---

## Future Enhancements

> Prioritized for production-readiness and ML engineering interview relevance:

- Web frontend (Streamlit or React)
- Follow-up questions / conversational flow
- HuggingFace Spaces hosting
- MLflow logging + visualization
- Airflow DAG for automated indexing and retraining

---

## ASR Test (Transcription Only)

```bash
ffmpeg -f alsa -i default -t 5 voice_assistant/data/input.wav
python voice_assistant/asr/transcribe.py --file voice_assistant/data/input.wav
```

---

## Contact

For questions, opportunities, or collaborations, contact:  
**ayaoshima.us@gmail.com**
