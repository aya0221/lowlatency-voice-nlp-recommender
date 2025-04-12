# Voice-Driven AI Workout Assistant

A modular, production-grade voice assistant that transcribes real-time audio, extracts structured workout queries using NLP, and delivers ranked recommendations via OpenSearch. The system showcases scalable architecture, fine-tuned transformer models, and relevance-boosted semantic search.

---

## System Architecture

```
Voice Input (Mic / UI)
        │
        ▼
  Whisper ASR (Transcription)
        │
        ▼
  Intent Classifier (Fine-tuned DistilBERT)
        │
        ▼
  Entity Extractor (spaCy + Rules)
        │
        ▼
  Query Constructor → OpenSearch
        │
        ▼
 Ranked Workout Recommendations
        │
        ▼
     Web UI (Text / Voice)
```

---

## Engineering Highlights

- **Modular Design**: Clean separation of ASR, NLU, search, and UI for testability and future scaling.
- **Transformer-Based NLP**: Intent classification powered by fine-tuned DistilBERT for task-specific voice command understanding.
- **Hybrid Entity Recognition**: Combines spaCy with domain-specific rule matchers to maximize coverage and accuracy.
- **Boosted Ranking Engine**: OpenSearch used with weighted field relevance (e.g., instructor, duration, type) and synonym/goal normalization.
- **Web UI Integration**: React frontend with full voice input + FastAPI backend for end-to-end interaction.

---

## Technologies

- Python 3.11 / JavaScript (React)
- OpenAI Whisper (ASR)
- HuggingFace Transformers (DistilBERT)
- PyTorch (GPU training)
- spaCy (NER)
- OpenSearch (ranking engine)
- FastAPI (API server)
- Vite + Tailwind CSS (UI)
- Docker (OpenSearch)
- NVIDIA RTX 4050 (local training)

---

## Demo

> Query example:  
> _“Find me a 20-minute yoga with Alex”_

[![Watch Demo on YouTube](https://img.youtube.com/vi/GDH2nT_EzUI/hqdefault.jpg)](https://youtu.be/GDH2nT_EzUI)

---

## Project Structure

```bash
voice_assistant/
├── asr/                        # Whisper ASR
├── nlu/                        # Intent + entity logic
├── search/                     # OpenSearch indexing & query
├── api/                        # FastAPI server
├── models/                     # Fine-tuned model weights
├── utils/                      # Configs
voice_ui/                       # React web interface
```

---

## Setup & Run

### Option 1: Web UI + API

```bash
# Clone & setup backend
git clone https://github.com/aya0221/Voice-AI-Workout-Assistant.git
cd Voice-AI-Workout-Assistant
python -m venv venv-voice-ai-coach
source venv-voice-ai-coach/bin/activate
pip install -r requirements.txt

# Start FastAPI backend
uvicorn voice_assistant.api.main:app --reload
```

```bash
# Launch frontend
cd voice_ui
npm install
npm run dev
```

Open `http://localhost:5173`  
Supports voice and text input with real-time NLP + recommendations.

---

### Option 2: CLI Mode (Voice → Result)

Run everything in one shot from terminal:

```bash
python voice_assistant/nlu/nlu_pipeline.py --cli
```

---

## Intent Classification

Maps transcribed text to high-level user goals.

| Intent         | Sample Input                                  |
|----------------|------------------------------------------------|
| `search_class` | "Find me a 30-minute HIIT ride with Alex"     |
| `track_metric` | "How many workouts did I do last week?"       |
| `greeting`     | "Hey there!"                                   |

> Powered by a fine-tuned transformer. See [NLP Model Fine-Tuned for Intent](#nlp-model-fine-tuned-for-intent) for training details.

---

## Entity Recognition

Extracts key workout filters:

| Entity        | Example              |
|---------------|----------------------|
| `time`        | “20 minutes”          |
| `intensity`   | “low impact”          |
| `person`      | “Robin”               |
| `workout_type`| “yoga”, “ride”        |
| `goal`        | “lose weight”         |
| `tags`        | “cardio”, “relax”     |

---

## Ranking Engine (OpenSearch Boosting)

Performs semantic search over an indexed JSON workout dataset with real-world ranking logic.

### Boosting Strategy

- Weighted fields: instructor, type, tags, intensity, duration
- Goal-to-tag mapping (“burn fat” → “cardio”)
- Synonym normalization (“bike”, “ride” → “cycling”)
- ±5 min duration tolerance

### Add / Update Workout Data

```bash
# Start OpenSearch server
docker-compose up -d

# Re-index workouts
python voice_assistant/search/index_workouts.py
```

---

## NLP Model Fine-Tuned for Intent

Intent classification is powered by a fine-tuned `distilbert-base-uncased`, optimized for this specific voice task.

| Parameter         | Value                           |
|-------------------|---------------------------------|
| Model             | `distilbert-base-uncased`       |
| Training Samples  | 1000+ task-specific utterances  |
| Epochs            | 20                              |
| Batch Size        | 16 (train) / 8 (eval)           |
| Learning Rate     | 2e-5 with warmup + decay        |
| Device            | NVIDIA RTX 4050                 |
| Best Checkpoint   | Step 47 (early convergence)     |
| Final Accuracy    | 100% (held-out set)             |
| Final Loss        | ↓ from 1.60 → 0.0005             |

> `load_best_model_at_end=True` ensured best checkpoint was restored for deployment.

---

## Model Training

```bash
python voice_assistant/nlu/train_intent_classifier.py
```

Produces:
- `model.safetensors`, `tokenizer`, `label_map.json`
- Logs to stdout (MLflow support available)

---

## ASR Test Only

```bash
ffmpeg -f alsa -i default -t 5 voice_assistant/data/input.wav
python voice_assistant/asr/transcribe.py --file voice_assistant/data/input.wav
```
---

## Contact

**ayaoshima.us@gmail.com**
