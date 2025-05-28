'''
let first timer register her/his profile.
used the inf to pull up the precomputed cold-start recommendations.
'''
import pandas as pd

def age_to_group(age):
    try:
        age = int(age)
        if age <= 25:
            return "18-25"
        elif age <= 35:
            return "26-35"
        elif age <= 50:
            return "36-50"
        else:
            return "50+"
    except:
        return None

def normalize_country(country):
    c = country.strip().lower()
    if c in {"us", "u.s.", "america", "usa", "united states", "united states of america"}:
        return "United States"
    return country.title()

def main():
    print("=== User Onboarding ===")

    # Step 1: Age input → auto-categorized
    while True:
        age_raw = input("Enter your age: ").strip()
        age_group = age_to_group(age_raw)
        if age_group:
            break
        print("Invalid age. Please enter a number.")

    # Step 2: Fitness level
    fitness_level = input("Enter fitness level [Beginner/Intermediate/Advanced]: ").strip().capitalize()

    # Step 3: Workout types with choices
    available_types = [
        "Cycling", "Running", "Walking", "Stretching", "Meditation", "Strength", "HIIT", "Yoga"
    ]
    print("Available workout types:", ", ".join(available_types))
    types_input = input("Enter preferred workout types (comma-separated): ").strip()

    # Step 4: Region
    country = input("Enter your country (or leave blank): ").strip()
    state = input("Enter your state/province (or leave blank): ").strip()
    region = None
    if country:
        normalized_country = normalize_country(country)
        region = f"{normalized_country}/{state}" if state else normalized_country
    elif state:
        region = state  # edge case: state only

    # Normalize workout types input
    types = [t.strip() for t in types_input.split(",") if t.strip()]
    normalized_types = []
    for t in types:
        tlower = t.lower()
        if tlower == "hiit":
            normalized_types.append("HIIT")
        else:
            normalized_types.append(t.capitalize())
    types_str = ", ".join(normalized_types)

    print(f"\nUser profile created with age_group={age_group}, fitness_level={fitness_level}, workout_types={types_str}, region={region if region else 'N/A'}")

    print("\nTop recommended workouts for your profile:")
    # Load segment-based cold-start recs from CSV
    recs = pd.read_csv("voice_assistant/data/user_database/segment_recommendations.csv")
    segments = [f"{age_group}|{fitness_level}|{t}" for t in normalized_types]
    filtered = recs[recs["segment_key"].isin(segments)]
    if filtered.empty:
        print("No direct match found. Showing fallback recommendations.")
        filtered = recs[(recs["segment_key"].str.startswith(f"{age_group}|{fitness_level}"))]

    if filtered.empty:
        print("⚠️ No cold-start recommendations available.")
        return

    # Load metadata to display titles
    workouts = pd.read_json("voice_assistant/data/database_workouts/augmented_workouts.json")
    workouts.rename(columns={"id": "workout_id"}, inplace=True)
    merged = filtered.merge(workouts, on="workout_id", how="left")
    top = merged.sort_values("score", ascending=False).drop_duplicates("workout_id").head(10)

    for i, r in top.iterrows():
        print(f"- {r['title']} | Instructor: {r['instructor']} | Tags: {r['tags']} | Score: {r['score']:.2f}")

    print("\n🎉 Onboarding complete.")

if __name__ == "__main__":
    main()
