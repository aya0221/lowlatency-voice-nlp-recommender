# Low-Latency Voice-to-NLP Recommender | Fine-Tuned for Fast, High-Accuracy Personalized Search

Production-grade modular system that transcribes real-time voice, extracts structured search intents, and delivers ranked personalized recommendations.  
Built for scalability, robustness, and showcasing fine-tuned transformer models with custom semantic retrieval.

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
- **Voice-to-Search Full Pipeline**: Whisper → Intent → Entity → Boosted Search → Real-time Recommendation → Web UI (**<0.2s end-to-end latency**).
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

## Demo

> Query example:  
> _“Find me a 20-minute yoga with Alex”_

<a href="https://youtu.be/EcXSCofSh-E" target="_blank">
  <img src="https://img.youtube.com/vi/EcXSCofSh-E/hqdefault.jpg" alt="Watch Demo on YouTube">
</a>
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
| Learning Rate                | 1e-4                               |
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

Insert the new section **right after** `## OpenSearch Ranking Engine` and **before** `## MLflow Tracking (for NER Model)`. This places the cold-start logic directly after retrieval and ranking, in context with how recommendations are served when user history is unavailable.

Here is the complete section to insert:

---

## Cold-Start Recommendation Pipeline (Segment-Based Personalization)

This module addresses the cold-start problem by **precomputing personalized recommendations** per user segment (based on profile only) using statistical and algorithmic techniques — eliminating the need for runtime models.

### 1. Segment Formation

Each user is bucketed into a unique segment via categorical cross keying:

```math
\text{Segment Key} = \text{Age Group} \times \text{Fitness Level} \times \text{Workout Type}
```

Example: `26-35|Intermediate|Cycling`

This forms **fine-grained personalization segments** (∼200–1000+ unique keys), supporting granular cold-start logic.

---

### 2. Engagement Scoring (Offline)

Each workout within a segment is assigned a score using a weighted heuristic:

```math
\text{Score} = \alpha \cdot \text{completion\_rate} + \beta \cdot \text{like\_rate} + \gamma \cdot \text{views}_{norm}
```

Where:

* `completion_rate = completed / started`
* `like_rate = likes / views`
* `views_norm` = z-score normalized view count (per segment)
* `α, β, γ` are tunable weights (e.g. 0.5, 0.3, 0.2)

---

### 3. Bayesian Smoothing

Used when engagement data is sparse (few sessions per workout):

```math
\hat{r} = \frac{s + \alpha_0}{n + \alpha_0 + \beta_0}
```

Where:

* `s` = observed success count (e.g. likes)
* `n` = total trials (e.g. views)
* `α₀, β₀` = prior (Beta distribution); avoids overfitting

---

### 4. Freshness Weighting

To prioritize new workouts and avoid stale content, we apply exponential decay:

```math
\text{Score}_{fresh} = \text{Score} \cdot e^{- \lambda \cdot \text{days\_old}}
```

Typically, `λ` = 0.01–0.05 based on desired decay horizon.

---

### 5. Diversity Reranking (Optional)

We rerank top-k per segment via **MMR (Maximal Marginal Relevance)** to ensure tag diversity:

```math
\text{MMR} = \arg\max_{d \in R \setminus S} \left[ \lambda \cdot \text{Rel}(d) - (1 - \lambda) \cdot \max_{s \in S} \text{Sim}(d, s) \right]
```

Where:

* `Rel(d)` = base score
* `Sim(d, s)` = tag vector cosine similarity

---

### Runtime Flow: CLI Cold-Start Demo

Running `onboarding_coldstart/onboarding_cli.py` executes:

1. Collects user profile (age → age group, level, types)
2. Forms keys like `26-35|Intermediate|Cycling`
3. Reads from `segment_recommendations.csv` for top precomputed workout IDs
4. Joins with `augmented_workouts.json` to show:

   * `title`, `instructor`, `tags`, and `score`
5. Renders a top-10 ranked list — no DB or ASR required

![CLI Onboarding Demo](./assets/onbording_demo_cli.png)

---

## MLflow Tracking (for NER Model)

- MLflow experiment configured via `mlflow_env.sh`.
- Captures losses, learning curves, evaluation metrics automatically during training.

Activate environment:

```bash
source voice_assistant/nlu/entity_scripts/mlflow_env.sh
```
---

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