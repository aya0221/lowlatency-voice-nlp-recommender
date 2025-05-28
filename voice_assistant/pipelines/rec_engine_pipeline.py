import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from pathlib import Path
import json
# import ast

BASE_DIR = Path(__file__).resolve().parent.parent  # voice_assistant/

# Tunable Weights
ALPHA = 0.5  # completion rate
BETA = 0.4   # like rate
GAMMA = 0.1  # total views

# Smoothing priors
BETA_PRIOR_ALPHA = 2
BETA_PRIOR_BETA = 2

# Decay constant for freshness
LAMBDA_DECAY = 0.01

# ------------------ Load Data ------------------
# load CSVs
users = pd.read_csv(BASE_DIR / "data/user_datanase/users.csv")
sessions = pd.read_csv(BASE_DIR / "data/user_datanase/sessions.csv")
feedback = pd.read_csv(BASE_DIR / "data/user_datanase/feedback.csv")

# load workouts.json
with open(BASE_DIR / "data/database_workouts/augmented_workouts.json", "r") as f:
    workouts = pd.json_normalize(json.load(f))
    workouts.rename(columns={"id": "workout_id"}, inplace=True)

# Merge workouts and users
sessions = sessions.merge(workouts, on="workout_id", how="left")
sessions = sessions.merge(users, on="user_id", how="left")

# ------------- Step 1: Segment Formation -------------
segment_rows = []
for _, row in users.iterrows():
    for wtype in row["preferred_types"].split(","):
        segment_key = f"{row['age_group']}|{row['fitness_level']}|{wtype.strip()}"
        segment_rows.append({"user_id": row["user_id"], "segment_key": segment_key})
segment_df = pd.DataFrame(segment_rows)
sessions = sessions.merge(segment_df, on="user_id")

# Now merge feedback
feedback = feedback.drop(columns=["workout_id"], errors="ignore")  # remove existing if present

feedback = feedback.merge(
    sessions[["session_id", "workout_id", "segment_key"]].drop_duplicates("session_id"),
    on="session_id", how="left"
)

assert "workout_id" in feedback.columns, "workout_id missing after merge"
assert "segment_key" in feedback.columns, "segment_key missing after merge"
print("Feedback columns:", feedback.columns)
print(feedback.head())

# ------------- Step 2: Engagement Scoring -------------
engagement = sessions.groupby(["segment_key", "workout_id"]).agg(
    views=("session_id", "count"),
    completions=("completed", "sum")
).reset_index()

likes = feedback.groupby(["segment_key", "workout_id"]).agg(
    likes=("liked", "sum"),
    feedbacks=("liked", "count")
).reset_index()

engagement = engagement.merge(likes, on=["segment_key", "workout_id"], how="left")
engagement = engagement.fillna({"likes": 0, "feedbacks": 0})

# Compute rates
engagement["completion_rate"] = engagement["completions"] / engagement["views"]
engagement["like_rate"] = engagement["likes"] / engagement["feedbacks"].replace(0, np.nan)
engagement["like_rate"] = engagement["like_rate"].fillna(0)

# ------------- Step 3: Bayesian Smoothing -------------
engagement["completion_smoothed"] = (
    engagement["completions"] + BETA_PRIOR_ALPHA
) / (engagement["views"] + BETA_PRIOR_ALPHA + BETA_PRIOR_BETA)

engagement["like_smoothed"] = (
    engagement["likes"] + BETA_PRIOR_ALPHA
) / (engagement["feedbacks"] + BETA_PRIOR_ALPHA + BETA_PRIOR_BETA)

# Final score
engagement["score"] = (
    ALPHA * engagement["completion_smoothed"] +
    BETA * engagement["like_smoothed"] +
    GAMMA * (engagement["views"] / engagement["views"].max())
)

# ------------- Step 4: Freshness Boost -------------
sessions["timestamp"] = pd.to_datetime(sessions["timestamp"])
latest_time = sessions["timestamp"].max()
last_play = sessions.groupby("workout_id")["timestamp"].max().reset_index()
last_play["age_days"] = (latest_time - last_play["timestamp"]).dt.days
last_play["freshness"] = np.exp(-LAMBDA_DECAY * last_play["age_days"])

engagement = engagement.merge(last_play[["workout_id", "freshness"]], on="workout_id", how="left")
engagement["score"] *= engagement["freshness"].fillna(1.0)

# ------------- Step 5: Diversity Enforcement (MMR) -------------
def mmr_rerank(tag_list, k=5, lambda_param=0.5):
    # tag_list is expected to be a list of lists already (not strings)
    docs = [" ".join(tags) if isinstance(tags, list) else str(tags) for tags in tag_list]
    tfidf = TfidfVectorizer().fit_transform(docs)
    sim = cosine_similarity(tfidf)
    selected = []
    remaining = list(range(len(tag_list)))
    selected.append(remaining.pop(0))

    while len(selected) < k and remaining:
        scores = []
        for r in remaining:
            rel = sim[r][r]
            div = max([sim[r][s] for s in selected])
            mmr_score = lambda_param * rel - (1 - lambda_param) * div
            scores.append((r, mmr_score))
        scores.sort(key=lambda x: x[1], reverse=True)
        selected.append(scores[0][0])
        remaining.remove(scores[0][0])
    return selected

print("Sample workout tags:", workouts["tags"].dropna().head())
print("Missing tags count:", workouts["tags"].isna().sum())
print("Empty list tags count:", workouts["tags"].apply(lambda x: isinstance(x, list) and len(x) == 0).sum())
print("Engagement workout_id sample:", engagement["workout_id"].dropna().unique()[:5])
print("Workouts workout_id sample:", workouts["workout_id"].dropna().unique()[:5])
print("Engagement ∩ Workouts:", len(set(engagement["workout_id"]) & set(workouts["workout_id"])))

# ------------- Step 6: Output Top-K Per Segment -------------
topk = []
for seg, group in engagement.groupby("segment_key"):
    ranked = group.sort_values("score", ascending=False).head(20)
    merged = ranked.merge(workouts, on="workout_id", how="left")

    if "tags" not in merged.columns:
        print(f"Warning: 'tags' column missing for segment {seg}")
        continue

    tagged = merged.dropna(subset=["tags", "score"])
    if tagged.empty:
        print(f"Warning: No valid tagged workouts for segment {seg}")
        continue

    try:
        selected_indices = mmr_rerank(tagged["tags"].tolist(), k=min(5, len(tagged)))
        selected_rows = tagged.iloc[selected_indices]
        selected_rows["segment_key"] = seg  # re-assign in case grouping dropped it
        topk += selected_rows[["segment_key", "workout_id", "score"]].to_dict("records")
    except Exception as e:
        print(f"MMR rerank failed for segment {seg}: {e}")
        continue

final_df = pd.DataFrame(topk)

if final_df.empty:
    print("No recommendations were generated. Check data inputs and scoring.")
else:
    output_path = BASE_DIR / "data/user_datanase/segment_recommendations.csv"
    final_df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")
