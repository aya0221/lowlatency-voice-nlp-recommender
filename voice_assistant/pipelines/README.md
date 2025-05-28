# Personalized Cold-Start Recommendation System

This repository implements a **production-grade cold-start recommendation pipeline** for first-time users of a workout recommendation platform. It uses segment-level logic, engagement scoring with Bayesian smoothing, freshness-aware ranking, and MMR-based diversity reranking — all performed in a fully offline, pandas-native batch pipeline.

---

## System Architecture Overview

At onboarding, users provide demographic and preference inputs. This pipeline:

* Segments users into 144 fine-grained groups
* Scores 600 workouts using over 34,000 sessions and 14,000 feedback entries
* Applies ranking heuristics and diversity constraints
* Precomputes and exports **721 recommendations** across segments to a CSV file

No model inference or real-time query infrastructure is needed.

---

## Pipeline: `voice_assistnat_pipelines_rec_engine_pipeline.py`

**Language:** Python 3.11
**Libraries:** `pandas`, `numpy`, `scikit-learn`, `TfidfVectorizer`, `cosine_similarity`

### Input Files:

| File                      | Description                                     |
| ------------------------- | ----------------------------------------------- |
| `users.csv`               | 2,000 users with age group, level, preferences  |
| `sessions.csv`            | 34,287 sessions (including completion tracking) |
| `feedback.csv`            | 14,368 binary feedback entries (likes/dislikes) |
| `augmented_workouts.json` | 600 total workouts with full metadata and tags  |

---

## Algorithmic Steps

### 1. Segment Formation

Users are segmented based on:

$$
\text{SegmentKey} = \text{AgeGroup} \times \text{FitnessLevel} \times \text{PreferredWorkoutType}
$$

This yields 144 unique segments, e.g., `"26-35|Advanced|Yoga"`.

---

### 2. Engagement Scoring

Engagement scores are computed from sessions + feedback data:

$$
\text{RawScore} = \alpha \cdot \text{CompletionRate} + \beta \cdot \text{LikeRate} + \gamma \cdot \text{NormalizedViews}
$$

* `CompletionRate = #Completed / #Viewed`
* `LikeRate = #Liked / #Feedbacks`
* `ViewsNorm = \text{Views} / \max(\text{Views})`
* Tunable weights: `α = 0.5`, `β = 0.4`, `γ = 0.1`

---

### 3. Bayesian Smoothing

To stabilize sparse segment scores:

$$
\hat{p}_{\text{like}} = \frac{s + \alpha_0}{n + \alpha_0 + \beta_0}
\quad,\quad
\hat{p}_{\text{completion}} = \frac{c + \alpha_0}{v + \alpha_0 + \beta_0}
$$

* Prior: `Beta(2, 2)`
* `s`: likes, `n`: feedbacks, `c`: completions, `v`: views

These smoothed rates replace raw ratios in the scoring formula.

---

### 4. Freshness Decay

To prefer newer workouts:

$$
\text{FreshScore} = \text{Score} \cdot e^{- \lambda \cdot \text{daysOld}}
\quad,\quad \lambda = 0.01
$$

Workouts played more recently get boosted in ranking.

---

### 5. Diversity Reranking (MMR)

For each segment, top-20 scored workouts are reranked via **Maximal Marginal Relevance** using TF-IDF on workout `tags`.

$$
\text{MMR}(d) = \lambda \cdot \text{Rel}(d) - (1 - \lambda) \cdot \max_{s \in S} \text{Sim}(d, s)
$$

* `Rel(d)`: self-similarity in tag vector space
* `Sim(d, s)`: cosine similarity (via `sklearn.metrics.pairwise`)
* `λ = 0.5`: balance relevance/diversity
* `k = 5`: final top-5 selected from top-20 per segment

---

## Output

* Final output:
  `data/user_datanase/segment_recommendations.csv`
  Contains **721 top-5 recommendations** across all segments.

```csv
segment_key,workout_id,score
18-25|Beginner|Yoga,123,0.92
26-35|Advanced|Strength,456,0.87
...
```

This file can be directly used in production onboarding or batch delivery with zero latency.

---

## Key Highlights

* **True offline pipeline** with tunable heuristics and interpretable scoring
* **Bayesian smoothing + freshness decay** ensure robust rankings
* **MMR diversity** reduces content repetition
* **Scalable** to millions of users with offline batch mode
