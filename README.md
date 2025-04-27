# Personalized Fitness Voice Recommender | Whisper ASR, Transformers, Semantic Search

Production-grade modular system that transcribes real-time voice, extracts structured workout requests, and delivers ranked fitness class recommendations.  
Built for scalability, robustness, and showcasing fine-tuned transformer models and custom semantic ranking.

---

## System Architecture

```
Voice Input (Mic / UI)
        ↓
Whisper ASR (Audio Transcription)
        ↓
Intent Classification (Fine-tuned DistilBERT)
        ↓
Entity Extraction (Fine-tuned RoBERTa + spaCy Transformer NER)
        ↓
Query Construction → OpenSearch + Custom Relevance Boosting
        ↓
Ranked Workout Recommendations
        ↓
Web UI Interaction (React) / LLM Feedback (planned)
```

---

## Engineering Highlights

- **Strict Modularization**: Isolated components for ASR, intent, entity, search, UI to ensure scalability and testability.
- **Fine-Tuned Intent Detection**: DistilBERT-based classifier achieving **100% validation accuracy** for voice command understanding.
- **Transformer-Based Entity Extraction**: RoBERTa fine-tuned with spaCy’s TransformerListener and Transition-Based Parser achieving **99.97% F1**.
- **Custom Boosted Ranking Algorithm**: OpenSearch combined with hand-crafted field boosting, synonym normalization, and goal mapping for real-world ranking optimization.
- **Voice-to-Search Full Pipeline**: Whisper → Intent → Entity → Boosted Search → Real-time Recommendation → Web UI.
- **GPU-Optimized Training**: Entire model training executed on local Linux GPU environment.

---

## Tech Stack

- **Languages**: Python, JavaScript (React, Vite, Tailwind)
- **ASR**: Whisper (openai/whisper-small.en)
- **NLP Models**:
  - **Intent**: DistilBERT (fine-tuned)
  - **Entity**: RoBERTa-base + spaCy Transformer Pipeline
- **Ranking**: OpenSearch + Custom Field Boosting
- **Infra**: Docker, FastAPI (Backend), HuggingFace Hub (Model Hosting)
- **Local Training**: Ubuntu Linux, CUDA 12.x, PyTorch GPU

---

## Project Structure

```bash
voice_assistant/
├── asr/                          # Whisper transcription
├── nlu/
│   ├── intent_scripts/            # Intent model fine-tuning
│   ├── entity_scripts/            # Entity model fine-tuning, configs
│   ├── nlu_pipeline.py            # Full voice → intent/entity → search pipeline
├── models/
│   ├── intent_model/              # Fine-tuned DistilBERT model
│   ├── entity_model/              # Fine-tuned RoBERTa NER model
├── search/                        # OpenSearch indexing and search query construction
├── api/                           # FastAPI web server
├── data/                         # Input data (ASR audio, NLU training datasets)
├── utils/                         # .env, configuration loading
voice_ui/                          # React web frontend
```

---

## Setup & Run

### Option 1: Web UI + API

```bash
# Install Backend
git clone https://github.com/aya0221/Voice-AI-Workout-Assistant.git
cd Voice-AI-Workout-Assistant
python -m venv venv-voice-ai-coach
source venv-voice-ai-coach/bin/activate
pip install -r requirements.txt

# Start FastAPI Backend Server
uvicorn voice_assistant.api.main:app --reload
```

```bash
# Launch Frontend
cd voice_ui
npm install
npm run dev
```

Open your browser and access the URL printed after npm run dev (typically http://localhost:5173 unless otherwise specified in the terminal output).
Supports voice and text input with real-time NLP + recommendations.

---

### Option 2: CLI Mode (Voice → Result)

Run everything in one shot from terminal:

```bash
python voice_assistant/nlu/nlu_pipeline.py --cli
```

---

## Fine-Tuned Models Overview

### Intent Classification (DistilBERT)

| **Parameter**                | **Value**                               |
|------------------------------|-----------------------------------------|
| Base Model                   | `distilbert-base-uncased`               |
| Training Samples             | 1000 task-specific utterances           |
| Batch Size                   | 16 (train) / 8 (eval)                   |
| Training Epochs              | 20                                      |
| Learning Rate                | 2e-5 (with 0.1 warmup ratio + linear decay) |
| Dropout                      | 0.1                                     |
| Optimizer                    | AdamW                                   |
| Scheduler                    | Linear                                  |
| Device                       | Local NVIDIA GPU (6GB)                  |
| Final Validation Accuracy    | **100%** (held-out set)                 |
| Loss (start → end)           | 1.60 → 0.0005                           |
| Best Model Selection         | Enabled (`load_best_model_at_end=True`, metric: accuracy) |

**Training Command:**

```bash
python voice_assistant/nlu/intent_scripts/train_intent_classifier.py
```

---

### Entity Extraction (RoBERTa + spaCy Transformer NER)

| **Parameter**                | **Value**                               |
|------------------------------|-----------------------------------------|
| Base Model                   | `roberta-base`                          |
| spaCy Architecture           | TransformerListener + Transition-Based Parser |
| Training Samples             | 30,000+ synthetic ASR-style workout queries |
| Batch Size                   | 16 (doc batch)                          |
| Training Epochs (Max)        | 600 (early stopping applied)            |
| Max Steps                    | 6000                                    |
| Learning Rate                | 0.0001                                  |
| Dropout                      | 0.15                                    |
| Optimizer                    | Adam (with L2 regularization)           |
| Gradient Clipping            | 1.0                                     |
| Evaluation Metric            | Entities F1                             |
| Final Evaluation F1          | **99.97%** (dev set)                    |
| Device                       | Local NVIDIA GPU (6GB)                  |
| MLflow Tracking              | Enabled (via `source mlflow_env.sh`)    |

**Training Commands:**

```bash
# Fill missing hyperparameters
python -m spacy init fill-config voice_assistant/nlu/entity_scripts/configs/config_manual.cfg voice_assistant/nlu/entity_scripts/configs/config_filled.cfg

# Validate training data
python -m spacy debug data voice_assistant/nlu/entity_scripts/configs/config_filled.cfg

# Train on GPU
source voice_assistant/nlu/entity_scripts/mlflow_env.sh
python -m spacy train voice_assistant/nlu/entity_scripts/configs/config_filled.cfg --output voice_assistant/models/entity_model --gpu-id 0
```

**Evaluation Command:**

```bash
python -m spacy evaluate voice_assistant/models/entity_model/model-best voice_assistant/data/entity_data/dev.spacy --gpu-id 0 --output ./evaluation_result.json
```

Both the intent classifier and entity extractor models achieved >99% accuracy/F1 on realistic, ASR-style voice commands.
**Generated Model Directory:**

```
voice_assistant/models/entity_model/model-best/
```

**Hugging Face Model Deployment:**

- Model Name: `workout-entity-extractor-roberta`
- Hosted at: [https://huggingface.co/Aya-In-Brooklyn/workout-entity-extractor-roberta](https://huggingface.co/Aya-In-Brooklyn/workout-entity-extractor-roberta)
- Format: Full `spaCy` pipeline (`TransformerListener + Transition-Based Parser`) trained on ASR-style synthetic workout queries.


---

## OpenSearch Ranking Engine

- Index: Fitness classes dataset
- Boosted fields:
  - Instructor Name
  - Workout Type
  - Tags
  - Intensity Level
  - Duration (±5 min tolerance)
- Synonym normalization (e.g., "bike" → "cycling")
- Goal-to-tag mapping (e.g., "burn fat" → "cardio")
- Manual field boosting rules on top of OpenSearch scoring

Indexing:

```bash
docker-compose up -d  # Start OpenSearch
python voice_assistant/search/index_workouts.py
```

---

## MLflow Tracking (for NER Model)

- MLflow experiment configured via `mlflow_env.sh`.
- Captures losses, learning curves, evaluation metrics automatically during training.

Activate environment:

```bash
source voice_assistant/nlu/entity_scripts/mlflow_env.sh
```

---

## ASR: Whisper Transcription

- Using HuggingFace pre-trained openai/whisper-small.en model
- Real-time mic input supported

Test recording:

```bash
ffmpeg -f alsa -i default -t 5 voice_assistant/data/input.wav
python voice_assistant/asr/transcribe.py --file voice_assistant/data/input.wav
```

---

## Future Enhancements

- **Chat Feedback with LLMs**: Open a conversational loop after search to refine recommendations dynamically. Combine retrieval + context-aware generation (RAG flow) for class specific information.
- **Multilingual Support**: ASR transcription in multiple languages → auto-translate → English pipeline processing.
- **End-to-End Transformer NER**: Compare Transformer-based span classification vs current transition-based parser.

---

## Contact

📧 ayaoshima.us@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/ayaoshima/)