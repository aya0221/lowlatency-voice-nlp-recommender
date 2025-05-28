# Personalized Cold-Start Recommendation System

This module delivers **instant, personalized workout recommendations for first-time users** by precomputing segment-specific top-k workouts. It uses a fully offline ML pipeline — with no model inference, real-time database, or API dependencies — combining statistical engagement scoring, Bayesian smoothing, freshness-aware ranking, and MMR-based diversity reranking.

---

## System Architecture

At onboarding, users select their **age group**, **fitness level**, and **preferred workout types**. This pipeline:

* Segments users into **144** fine-grained demographic + preference groups
* Scores **600** workouts based on **34,287** historical sessions and **14,368** feedback entries
* Applies **Bayesian-adjusted scoring**, **freshness decay**, and **MMR reranking**
* Outputs **721 top-5 recommendations** (∼5 per segment) to a single CSV for immediate delivery

---

## Pipeline Location

`voice_assistnat/pipelines/rec_engine_pipeline.py`
Python 3.11, `pandas`, `numpy`, `scikit-learn`, `TfidfVectorizer`, `cosine_similarity`

---

## Core Algorithmic Steps

### 1. Segment Formation

Users are mapped to segments defined as:

$$
\text{SegmentKey} = \text{AgeGroup} \times \text{FitnessLevel} \times \text{PreferredWorkoutType}
$$

Example: `"26-35|Intermediate|Cycling"`

→ **144 unique segments** capturing behavioral priors without any history.

---

### 2. Engagement Scoring

Each workout receives a composite score based on observed user engagement:

$$
\text{Score}_{\text{raw}} = \alpha \cdot \text{CompletionRate} + \beta \cdot \text{LikeRate} + \gamma \cdot \frac{\text{Views}}{\max(\text{Views})}
$$

* $\text{CompletionRate} = \frac{\text{\#Completions}}{\text{\#Views}}$
* $\text{LikeRate} = \frac{\text{\#Likes}}{\text{\#Feedbacks}}$
* Tunable weights: $\alpha = 0.5, \beta = 0.4, \gamma = 0.1$

---

### 3. Bayesian Smoothing

To stabilize scoring under sparse feedback, we apply Beta priors:

$$
\hat{p}_{\text{like}} = \frac{s + \alpha_0}{n + \alpha_0 + \beta_0}, \quad
\hat{p}_{\text{completion}} = \frac{c + \alpha_0}{v + \alpha_0 + \beta_0}
$$

Where:

* $s$: number of likes, $n$: feedback count
* $c$: number of completions, $v$: view count
* Prior: $\text{Beta}(2, 2)$

→ These smoothed rates replace the raw metrics in the scoring equation.

---

### 4. Freshness Decay

Older workouts are downweighted via exponential time decay:

$$
\text{Score}_{\text{fresh}} = \text{Score} \cdot e^{-\lambda \cdot \text{AgeInDays}}, \quad \lambda = 0.01
$$

→ More recent workouts are promoted, reducing content staleness.

---

### 5. Diversity Reranking (MMR)

To prevent similar-tagged workouts from dominating recommendations, top-20 candidates per segment are reranked with Maximal Marginal Relevance:

$$
\text{MMR}(d) = \lambda \cdot \text{Rel}(d) - (1 - \lambda) \cdot \max_{s \in S} \text{Sim}(d, s)
$$

* $\text{Rel}(d)$: TF-IDF self-relevance (diagonal of similarity matrix)
* $\text{Sim}(d, s)$: cosine similarity between workout tag vectors
* $\lambda = 0.5$: relevance-diversity balance
* $k = 5$: top-5 final recommendations per segment

---

## Output

`voice_assistant/data/user_datanase/segment_recommendations.csv`

Precomputed top-5 recommendations per segment:

```csv
segment_key,workout_id,score
18-25|Beginner|Yoga,w123,0.914
26-35|Advanced|Strength,w456,0.872
...
```

→ Immediately retrievable at onboarding with **zero latency**.

---

## CLI Onboarding Runtime Flow

The CLI demo script `onboarding_cli.py` simulates cold-start onboarding:

1. User provides age, fitness level, workout preferences
2. Segment key is constructed (e.g., `"18-25|Beginner|Yoga"`)
3. Top recommendations are loaded from `segment_recommendations.csv`
4. Metadata is joined from `augmented_workouts.json`
5. User receives 5–10 personalized workout cards

![CLI Onboarding Demo](../assets/onbording_demo_cli.png)

---

## Input Datasets

| File                          | Records | Description                                       |
| ----------------------------- | ------- | ------------------------------------------------- |
| `users.csv`                   | 2,000   | Age group, fitness level, preferred workout types |
| `sessions.csv`                | 34,287  | Workout starts, completions, timestamps           |
| `feedback.csv`                | 14,368  | Binary feedback: thumbs-up/down                   |
| `augmented_workouts.json`     | 600     | Metadata per workout: title, tags, instructor     |
| `segment_recommendations.csv` | 721     | Final top-5 workout recs per segment              |

---

### Example: `sessions.csv`

```csv
session_id,user_id,workout_id,completed,timestamp
1001,1,w220,1,2025-03-05T06:32:41.016619
1002,1,w357,1,2025-01-20T13:38:41.016647
```

---

### Example: `feedback.csv`

```csv
feedback_id,session_id,user_id,workout_id,liked,feedback_time
1,14551,788,103,1,2025-04-06T15:24:41Z
```

---

### Example: `users.csv`

```csv
user_id,age_group,fitness_level,preferred_types
1,18-25,Beginner,"walking,cycling"
2,26-35,Advanced,"cardio,boxing"
```

---

### Example: `augmented_workouts.json`

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
  }
]
```

---

## Final Notes

* This is a **fully ML-informed recommendation system**, not rule-based
* No runtime compute is needed — outputs can be cached or preloaded
* Algorithms are mathematically grounded with rigorously tuned parameters
* Easily extensible to A/B test priors, decay rates, or diversity criteria
