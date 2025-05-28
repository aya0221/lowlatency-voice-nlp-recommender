# Personalized Cold-Start Recommendation System

This repository implements a **production-grade cold-start recommendation pipeline** for first-time users of a workout recommendation platform. It integrates segment-based logic, engagement-based scoring, Bayesian smoothing, diversity-aware ranking, and real-time retrieval strategies. It is built to simulate modern industry practices for scalable, latency-sensitive personalization.

---

## System Architecture Overview

**Cold-start** recommendation is triggered **at user onboarding**. Since no historical interaction data exists, the system uses demographic + preference-based segmentation and offline engagement data to precompute and retrieve personalized workouts in constant time.

* **Offline logic**: Computes top workouts per user segment (721 segment-workout entries across 144 unique segments)
* **Online logic**: Lookup and rerank based on registration profile

---

## Modules

### `rec_engine.py` (Production Recommendation Engine)

**Backend Dependencies**: MySQL, OpenSearch

This engine powers both cold-start and returning-user recommendations. It dynamically routes:

* **Cold-start logic** (no user history):

  * Extracts user segment key: `age_group × fitness_level × workout_type`
  * Looks up top-ranked workouts from precomputed SQL table `cold_start_top_workouts`
  * Applies optional filters: flagged instructors, fallback preferences, de-duplication

* **Returning-user logic**:

  * Incorporates user feedback (likes/dislikes), flagged instructors
  * Builds an OpenSearch query with filters:

    * `must`: keyword match (title, tags, description)
    * `filter`: fitness level, duration, workout type
    * `should`: preference boosting (liked types, instructors)
    * `must_not`: flagged/disliked instructors, previously seen workouts
  * Executes query with OpenSearch and returns personalized list

---

### `onboarding_cli.py` (CLI Demo Script)

**Purpose**: Simulates onboarding flow for first-time users and displays top recommended workouts instantly.

**Runtime Steps**:

1. Accepts inputs:

   * Age → grouped into buckets
   * Fitness level (Beginner, Intermediate, Advanced)
   * Preferred workout types (Yoga, Cycling, etc.)
   * Region (country/state)
2. Constructs segment keys like:

   ```
   segment_key = "26-35|Advanced|Yoga"
   ```
3. Loads cold-start recommendations from:

   * `segment_recommendations.csv`: 721 ranked entries for 144 unique segments
   * `augmented_workouts.json`: metadata from 600 unique workout definitions
4. Displays Top-10 workouts sorted by score:

   * Title, instructor, tags, Bayesian engagement score

This lookup is **instantaneous** after registration, showcasing the ability to generate personalized results with **zero latency and no model inference.**

---

## Algorithmic Computations (Offline)

### 1. Segment Formation

Each user segment is defined as:

$\text{SegmentKey} = \text{AgeGroup} \times \text{FitnessLevel} \times \text{WorkoutType}$

Example:

```
segment_key = "18-25|Beginner|Yoga"
```

From the data, 144 unique segments are constructed to cover key demographic and workout type intersections.

---

### 2. Engagement-Based Scoring

Offline engagement metrics are computed for each workout per segment using 34,287 session logs:

$\text{Score} = \alpha \cdot \text{CompletionRate} + \beta \cdot \text{LikeRate} + \gamma \cdot \text{ViewsNorm}$

* `CompletionRate = #Completed / #Started`
* `LikeRate = #Liked / #Viewed` (from 14,368 feedback entries)
* `ViewsNorm`: Z-score normalized views
* Tunable weights `\alpha, \beta, \gamma` allow ranking calibration

---

### 3. Bayesian Smoothing

To mitigate noise in low-traffic segments, we apply **Beta-Binomial smoothing**:

$\hat{r} = \frac{s + \alpha_0}{n + \alpha_0 + \beta_0}$

* `s`: observed likes (successes)
* `n`: total views (trials)
* `\alpha_0, \beta_0`: prior hyperparameters (e.g., Beta(2, 5))

This stabilizes recommendation scores under data sparsity.

---

### 4. Freshness Decay

Favor newer workouts using exponential decay:

$\text{FreshScore} = \text{Score} \cdot e^{-\lambda \cdot \text{daysOld}}$

This reduces rank of stale content while preserving score integrity.

---

### 5. Diversity-Aware Reranking (MMR)

To ensure variety in top-K results, we apply Maximal Marginal Relevance:

$\text{MMR}(d) = \lambda \cdot \text{Rel}(d) - (1 - \lambda) \cdot \max_{s \in S} \text{Sim}(d, s)$

* `Rel(d)`: base relevance score
* `Sim(d, s)`: cosine similarity of tag vectors
* `S`: selected workouts
* `\lambda`: tunable trade-off parameter

---

## Data Files

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


---
### Runtime Flow Demo of CLI Cold-Start

Running `onboarding_coldstart/onboarding_cli.py` executes:

- Collects user profile (age → age group, level, types)
- Forms keys like `26-35|Intermediate|Cycling`
- Reads from `segment_recommendations.csv` for top precomputed workout IDs
- Joins with `augmented_workouts.json` to show:
   * `title`, `instructor`, `tags`, and `score`
5. Renders a top-10 ranked list
![CLI Onboarding Demo](./assets/onbording_demo_cli.png)
---

## Key Highlights

* **Cold-start solved** via **offline precomputation** and **online segment-matching**
* **No training needed** at runtime
* **Scalable** to millions of users
* **Modular** architecture ready for hybrid or neural extensions
* **Industry-ready**: combines ranking theory, information retrieval, and recommender system best practices
