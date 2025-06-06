# Low-Latency Voice-to-NLP Recommender | Fine-Tuned for Fast, High-Accuracy Personalized Search

Production-grade modular system that transcribes real-time voice, extracts structured search intents, and delivers ranked personalized recommendations.  
Built for scalability, robustness, and showcasing fine-tuned transformer models with custom semantic retrieval.

---

## System Architecture

```
Cold-start-recommendation(Segment,Bayesian,MMR)
        ↓
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

## Cold-Start Recommendation Pipeline (Segment-Based Personalization)

This implementation generates instant personalized workout recommendations for new users by precomputing top-k workouts per profile segment — using offline scoring, smoothing, freshness decay, and diversity reranking with no runtime model or backend required.

---

### 1. Segment Formation

Each user is mapped to one or more segments using cross-keying:

```math
\text{SegmentKey} = \text{AgeGroup} \times \text{FitnessLevel} \times \text{PreferredWorkoutType}
```

Example:
`26-35|Intermediate|Cycling`

This produces **144 fine-grained segments** from combinations of 2000 users and their declared preferences.

---

### 2. Engagement Scoring (Offline)

For each segment–workout pair, an engagement score is computed based on:

```math
\text{RawScore} = \alpha \cdot \text{CompletionRate} + \beta \cdot \text{LikeRate} + \gamma \cdot \text{Views}_{norm}
```

Where:

* $`CompletionRate = |Completed| / |Viewed|`$
* $`LikeRate = |Liked| / |Feedbacks|`$
* $`Views_norm = \text{Views} / \max(\text{Views})`$
* Tunable weights: `α = 0.5`, `β = 0.4`, `γ = 0.1`

---

### 3. Bayesian Smoothing

Used to reduce variance under sparse data:

```math
\widehat{p}_{\text{like}}(w) = \frac{s_w + \alpha_0}{n_w + \alpha_0 + \beta_0}, \quad
\widehat{p}_{\text{completion}}(w) = \frac{c_w + \alpha_0}{v_w + \alpha_0 + \beta_0}
```


Where:

* $\alpha_0 = 2, \beta_0 = 2$ (prior strength)
* $\widehat{p}_{\text{like}}(w)$: smoothed like rate
* $\widehat{p}_{\text{completion}}(w)$: smoothed completion rate

This applies separately to both completion and like rates before computing the final score.

---

### 4. Freshness Weighting

To promote recently engaged workouts, we apply time-based exponential decay:

```math
\text{Score}_{\text{fresh}} = \text{Score} \cdot e^{- \lambda \cdot \text{daysOld}}
```

With decay factor `λ = 0.01`.
This boosts newer content in each segment.

---

### 5. Diversity Reranking (MMR)

From the top-20 scored workouts in each segment, **Maximal Marginal Relevance** reranks the final top-5 based on tag diversity:

```math
\text{MMR}(d) = \lambda \cdot \text{Rel}(d) - (1 - \lambda) \cdot \max_{s \in S} \text{Sim}(d, s)
```

* $\text{Rel}(d)$: self-similarity (TF-IDF tag vector)
* $\text{Sim}(d, s)$: cosine similarity of tags
* $\lambda = 0.5$: relevance vs. diversity balance
---

### Runtime Flow: CLI Cold-Start Demo

Running `onboarding_coldstart/onboarding_cli.py` performs:

1. Accepts user input: age, fitness level, workout preferences
2. Forms one or more segment keys, e.g., `26-35|Advanced|Cycling`
3. Looks up segment recommendations in `segment_recommendations.csv`
4. Joins with metadata from `augmented_workouts.json`
5. Displays top-10 workouts with:

   * `title`, `instructor`, `tags`, and `score`

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