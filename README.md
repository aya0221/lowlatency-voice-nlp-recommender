# Voice-Driven AI Workout Assistant

A production-grade voice-driven NLP system that transcribes user speech, extracts intent and entities, and delivers highly relevant workout recommendations using OpenSearch with a boosting-based ranking algorithm.

---

## Pipeline Overview

```
🎙️ Voice Input
→ Whisper ASR (transcription)
→ Intent Classification (DistilBERT, fine-tuned)
→ Entity Recognition (spaCy + rule-based matcher)
→ OpenSearch Query (boosted scoring engine)
→ Personalized Workout Recommendations
```

---

## Key Features

- **Real-Time ASR (Whisper)**  
  Converts spoken input into accurate transcriptions, supporting natural, multi-accent voice commands.

- **Intent Classification (Fine-Tuned DistilBERT)**  
  Classifies user goals (e.g., `search_class`, `track_metric`) using a GPU-fine-tuned transformer model trained on over 1000 curated examples.

- **Hybrid Entity Recognition (spaCy + Rules)**  
  Combines ML-based and domain-specific rule-based entity extraction for robust handling of workout-specific terms.

- **Semantic Search & Ranking (OpenSearch)**  
  Fast, top-K recommendation retrieval from an indexed dataset with weighted field-based boosting:
  - Instructor, intensity, type, time, and tags scored by relevance
  - Synonym + goal-to-tag normalization
  - Duration flexibility (±5 min window)

- **Web UI Interface (React + Tailwind CSS)**  
  Fully integrated assistant front-end that supports voice input and text search with real-time NLP results displayed in an elegant, modern interface.

- **Clean, Modular Architecture**  
  All components (ASR, NLP, search, UI, API) are decoupled and production-ready — allowing flexible experimentation, scaling, and deployment.

---

## Demo

**Example Query:** _"Find me a 20-minute yoga with Alex"_

[![Watch Demo on YouTube](https://img.youtube.com/vi/GDH2nT_EzUI/hqdefault.jpg)](https://youtu.be/GDH2nT_EzUI)

---

## Technologies

- Python 3.11 / JavaScript (React)
- OpenAI Whisper (ASR)
- HuggingFace Transformers (DistilBERT)
- PyTorch (GPU fine-tuning)
- spaCy (NER + custom matcher)
- OpenSearch (search engine)
- FastAPI (backend API)
- Vite + React + Tailwind CSS (UI)
- Docker (OpenSearch setup)
- NVIDIA RTX 4050 (local training)

---

## Project Structure

```bash
voice_assistant/
├── asr/                        # Audio recording + Whisper transcription
├── data/                       # Training CSVs, audio input
├── models/                     # Fine-tuned model & tokenizer
├── nlu/                        # Intent classifier + entity extractor
├── search/                     # Indexing + OpenSearch query logic
├── api/                        # FastAPI server (connects UI ↔ NLP pipeline)
├── utils/                      # Shared configuration
voice_ui/                       # React + Tailwind frontend (calls /api)
```

---

## Setup & Run

### Option 1: Run Full Stack (Web UI + API)

This launches the assistant with a FastAPI backend and a React web interface for voice/text interaction.

#### 1. Clone & Set Up Python Backend

```bash
git clone https://github.com/aya0221/Voice-AI-Workout-Assistant.git
cd Voice-AI-Workout-Assistant
python -m venv venv-voice-ai-coach
source venv-voice-ai-coach/bin/activate
pip install -r requirements.txt
```

#### 2. Launch FastAPI Server (NLP API)

```bash
uvicorn voice_assistant.api.main:app --reload
```

#### 3. Launch React Frontend

```bash
cd voice_ui
npm install
npm run dev
```

Open `http://localhost:5173` to interact with the assistant.  
Supports both voice and text input.

---

### Option 2: CLI Mode (Record + Recommend via Terminal)

This mode runs the full pipeline directly from the terminal:  
Record voice → Transcribe → NLP Parsing → Search → Recommend.

```bash
# Record + Run Full Pipeline from audio
python voice_assistant/nlu/nlu_pipeline.py --cli
```

Returns parsed intent, extracted entities, and ranked workout recommendations in the console.

---

## Intent Classification

Classifies user intent from free-form voice input, enabling the assistant to understand the user’s high-level goal.

### Supported Intents

| Intent         | Example Input                                    |
|----------------|--------------------------------------------------|
| `search_class` | "Find me a 20-minute yoga with Alex"            |
| `track_metric` | "How many calories did I burn yesterday?"       |
| `greeting`     | "Hey there!"                                     |

> The intent classifier is a fine-tuned `distilbert-base-uncased` transformer, trained specifically for this task. Its weights are used in the full production pipeline. See [NLP Model Fine-Tuned for Intent](#nlp-model-fine-tuned-for-intent) for full training details.

---

## Entity Recognition

Extracts structured workout-related fields from voice input using hybrid Named Entity Recognition.

| Entity        | Example           |
|---------------|-------------------|
| `time`        | “30 minutes”       |
| `intensity`   | “low impact”       |
| `person`      | “Robin”            |
| `workout_type`| “yoga”, “ride”     |
| `goal`        | “lose weight”      |
| `tags`        | “cardio”, “relax”  |

A combined ML (spaCy) + rule-based system ensures robustness across flexible phrasing and domain-specific terminology.

---

## Search and Recommendation

Performs **semantic matching and ranking** of workouts based on extracted user intent and entities. Search is performed over a structured JSON-based workout dataset, indexed with OpenSearch.

### Ranking Strategy (Boosted)
- Instructor, workout type, duration, intensity, tags
- Synonym normalization ("ride" → "cycling")
- Goal-to-tag mapping ("lose weight" → "cardio")
- Duration range flexibility (±5 min)

### To Add / Replace Workouts:
The workout metadata is stored in a structured JSON format. Each entry includes title, duration, instructor, tags, intensity, and type.

1. Edit the source JSON dataset with your new entries
2. Re-index workouts:

```bash
# Start OpenSearch (via Docker)
docker-compose up -d

# Re-index workouts from JSON\python voice_assistant/search/index_workouts.py
```

---

## NLP Model Fine-Tuned for Intent

A domain-specific intent classifier was built by fine-tuning the `distilbert-base-uncased` transformer to recognize high-level task directives from natural speech (e.g., `search_class`, `track_metric`, `greeting`).  
Training was conducted on GPU with evaluation-based early stopping and automatic best checkpoint restoration to ensure maximum generalization performance.

| Parameter         | Value                             |
|-------------------|-----------------------------------|
| Model             | `distilbert-base-uncased`         |
| Training Samples  | 1000+ curated, balanced utterances |
| Epochs            | 20                                |
| Batch Size        | 16 (train) / 8 (eval)             |
| Learning Rate     | 2e-5 with warmup and decay        |
| Device            | NVIDIA RTX 4050                   |
| Best Checkpoint   | Step 47 (early convergence)       |
| Final Accuracy    | 100% on held-out eval set         |
| Final Loss        | ↓ from 1.60 → 0.0005              |

> `load_best_model_at_end=True` ensured automatic selection of the best-performing checkpoint (`checkpoint-47`), which was deployed in the production inference pipeline.

---

### Model Training

To reproduce or retrain the intent classifier:

```bash
python voice_assistant/nlu/train_intent_classifier.py
```

Outputs:
- Fine-tuned model: `model.safetensors`
- Tokenizer and label mapping: `tokenizer/`, `label_map.json`
- Training metrics: Console logs (MLflow integration optional)

---

## ASR Only

```bash
ffmpeg -f alsa -i default -t 5 voice_assistant/data/input.wav
python voice_assistant/asr/transcribe.py --file voice_assistant/data/input.wav
```

---

## Contact

For technical inquiries or collaboration:  
**ayaoshima.us@gmail.com**