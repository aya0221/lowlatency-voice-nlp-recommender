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

## Pipeline: `voice_assistnat/pipelines/rec_engine_pipeline.py`

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


---
## Each Dataset's contents:

| File                          | Description                                      |
| ----------------------------- | ------------------------------------------------ |
| `segment_recommendations.csv` | 721 entries across 144 unique segment keys       |
| `augmented_workouts.json`     | 600 total unique workouts with metadata          |
| `users.csv`                   | 2,000 user profiles with age, level, preferences |
| `sessions.csv`                | 34,287 recorded sessions for engagement modeling |
| `feedback.csv`                | 14,368 binary feedback entries (likes/dislikes)  |


### `segment_recommendations.csv`

```csv
segment_key,workout_id,score
18-25|Advanced|boxing,357,0.64
18-25|Advanced|boxing,540,0.5959157094566421
```

### `sessions.csv`

```csv
session_id,user_id,workout_id,completed,timestamp
1001,1,220,1,2025-03-05T06:32:41.016619
1002,1,357,1,2025-01-20T13:38:41.016647
```
* `completed_at` is `NULL` if user abandoned (used to compute instructor flagging rate)


### `feedback.csv`

```csv
feedback_id,session_id,user_id,workout_id,liked,feedback_time
1,14551,788,103,1,2025-04-06T15:24:41Z
```
* `liked` is `1` if user liked the workout


### `users.csv`

```csv
user_id,age_group,fitness_level,preferred_types
1,18-25,Beginner,"walking,cycling,cardio"
2,26-35,Advanced,cardio
```

### `augmented_workouts.json`
```json
[
  {
    "workout_id": "w123",
    "title": "30-Minute Gentle Yoga Flow",
    "instructor": "Alex Morgan",
    "instructor_id": "i10",
    "workout_type": "Yoga",
    "fitness_level": "Beginner",
    "duration": 30,
    "tags": ["calm", "stretch", "low-impact"]
  },
]
```

### MySQL

| Table                     | Purpose                               |
| ------------------------- | ------------------------------------- |
| `users`                   | Stores user profiles after onboarding |
| `sessions`                | Logs workout session start/completion |
| `feedback`                | Stores thumbs up/down feedback        |
| `cold_start_top_workouts` | Precomputed segment rankings          |
| `flags`                   | Stores instructor-level quality flags |
