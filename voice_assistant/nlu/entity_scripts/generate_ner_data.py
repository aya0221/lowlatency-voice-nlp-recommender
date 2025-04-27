import random
import json
from pathlib import Path
import spacy
from spacy.tokens import DocBin

nlp = spacy.blank("en")

# Define possible values for each entity type
WORKOUT_SYNONYMS = [
    ("ride", ["bike ride", "bike", "ride", "riding", "bicycle", "spinning"]),
    ("run", ["run", "running", "jog", "jogging", "go for a run"]),
    ("walk", ["walk", "walking", "take a walk", "go for a walk"]),
    ("yoga", ["yoga", "vinyasa", "restorative yoga"]),
    ("strength training", ["strength", "lift", "lifting", "weightlifting", "resistance training"]),
    ("HIIT", ["HIIT", "high intensity interval training", "bursts"]),
]
DURATIONS = [
    "10 min", "ten minutes", "10 minutes",
    "15 min", "fifteen minutes",
    "20 min", "twenty minutes",
    "30 min", "thirty minutes", "half an hour",
    "45 min", "forty five minutes",
    "60 min", "sixty minutes", "an hour", "one hour",
    "a quick one", "short workout", "long session", "about half an hour", "hour long"
]

INTENSITIES = [
    "easy", "light", "beginner", "low effort", "gentle",
    "moderate", "intermediate", "medium intensity",
    "high", "challenging", "advanced", "tough",
    "super intense", "killer", "intensity level 5", "hardcore"
]

GOALS = [
    "lose weight", "drop fat", "burn calories", "slim down",
    "get fit", "improve fitness", "stay in shape", "tone up",
    "build muscle", "get stronger", "gain muscle",
    "get toned", "look defined",
    "de-stress", "reduce anxiety", "relax",
    "gain endurance", "build stamina",
    "boost energy", "wake up",
    "clear my mind", "mental clarity", "feel better"
]

MOODS = [
    "tired", "low energy", "worn out", "sleepy",
    "energized", "excited", "ready to go",
    "unmotivated", "blah", "not feeling it",
    "pumped", "hyped",
    "lazy", "sluggish",
    "anxious", "stressed", "on edge"
]

INSTRUCTORS = [
    "Alex", "Robin", "Kendall", "Tunde", "Matt", "Jess", "Emma", "Cody", "Tatiana", "Aya",
    "Ben", "Ally", "Adrian", "Denis", "Chelsea", "Olivia", "Chris", "Rebecca"
]

FINDERS = [
    "find", "recommend", "give me", "show me", "search", "get me", "pull up",
    "what’s good", "help me find", "anything for", "can you find"
]

OPTIONALS = [
    "please", "now", "today", "asap", "right away",
    "right now", "if possible", "quickly", "soon"
]

TIME_OF_DAY = [
    "morning", "early morning", "late morning",
    "afternoon", "midday",
    "evening", "early evening", "late evening",
    "night", "late night", "before bed",
    "before work", "after work"
]

TEMPLATES = [
    # Minimal or implicit commands
    "{workout}",
    "{instructor}",
    "{time_of_day}",
    "{intensity}",
    "{mood}",

    # Basic phrases and ASR-transcribed short queries
    "{workout} with {instructor}",
    "{workout} by {instructor}",
    "{duration} {workout}",
    "{duration} {intensity} {workout}",
    "{duration} {workout} with {instructor}",
    "{duration} {workout} by {instructor}",
    "{intensity} {workout}",
    "{intensity} {workout} with {instructor}",
    "{intensity} {workout} by {instructor}",
    "{time_of_day} {workout}",
    "{time_of_day} {workout} with {instructor}",
    "{time_of_day} {workout} by {instructor}",
    "Just {duration} of {workout} would be great.",
    "Can I {workout} with {instructor} {time_of_day}?",

    # Finder-driven queries
    "{finder} {workout}",
    "{finder} a {workout}",
    "{finder} me {workout}",
    "{finder} me a {workout}",
    "{finder} {duration} {workout}",
    "{finder} me a {duration} {workout}",
    "{finder} {time_of_day} {workout}",
    "{finder} me a {time_of_day} {workout}",
    "{finder} a {duration} {workout} this {time_of_day}",
    "{finder} a {intensity} {workout}",
    "{finder} a {workout} for {goal}",
    "{finder} me a {workout} for {goal}",
    "{finder} a {workout} with {instructor}",
    "{finder} me a {workout} with {instructor}",
    "{finder} a {workout} by {instructor}",
    "{finder} me a {workout} by {instructor}",
    "{finder} a {duration} {workout} with {instructor}",
    "{finder} a {intensity} {workout} with {instructor}",
    "{finder} a {workout} by {instructor} to {goal}",
    "{finder} a {duration} {intensity} {workout}",
    "{finder} something {intensity} like {workout} to {goal}",
    "{finder} a {duration} {workout} session. I'm {mood}.",
    "{finder} me {duration} {workout} I can squeeze in this {time_of_day}.",
    "{finder} something {intensity} to boost my energy.",
    "{finder} something good for {mood} this {time_of_day}.",

    # Goal-driven phrases
    "I want to {goal}",
    "I wanna {goal}",
    "I gotta {goal}",
    "I need to {goal}",
    "I'd like to {goal}, so {finder} a {duration} {intensity} {workout}.",
    "Anything to help me {goal} this {time_of_day}?",
    "Find a {workout} for {goal} when I feel {mood}.",
    "I need a {workout} to {goal} with {instructor}.",
    "What’s good for {goal} if I have {duration}?",

    # Emotion/context driven
    "I'm {mood}.",
    "I'm {mood}. {finder} a {workout}.",
    "I feel {mood}. Can you {finder} a {intensity} {workout}?",
    "{finder} a {workout} for when I'm {mood}.",
    "Something relaxing like {workout} for this {time_of_day}, please.",

    # Specific use-case patterns
    "Can you {finder} me {duration} of {workout} with {instructor}?",
    "Schedule a {duration} {workout} with {instructor}.",
    "Wanna do {duration} of {workout} — {intensity}, with {instructor}.",
    "{finder} me a {duration} {workout} with {instructor} to {goal}, {optional}.",
    "Looking for a {workout} class by {instructor}.",
    "Any {intensity} {workout} suggestions?",
    "Give me a {duration} {workout} for {goal}.",
    "I want {workout} with {instructor}.",
    "30 min {workout}, please.",
    "Find me something {intensity}.",
    "Need something for {goal}, maybe {duration} of {workout}.",
    "What do you recommend if I'm feeling {mood}?",
    "Pick a class from {instructor} for {goal}.",
    "Is there a {duration} {workout} I can do?",
    "Show me a quick {intensity} {workout} this {time_of_day}.",
    "{finder} {duration} {workout} I can do before work.",
    "Help me {goal} with a {duration} {intensity} session."
]



def generate_example():
    # Randomly pick values for each placeholder
    workout_label, workout_variants = random.choice(WORKOUT_SYNONYMS)
    workout    = random.choice(workout_variants)
    duration   = random.choice(DURATIONS)
    intensity  = random.choice(INTENSITIES)
    goal       = random.choice(GOALS)
    mood       = random.choice(MOODS)
    instructor = random.choice(INSTRUCTORS)
    finder     = random.choice(FINDERS)
    optional   = random.choice(OPTIONALS)
    time_of_day   = random.choice(TIME_OF_DAY)

    template = random.choice(TEMPLATES)
    sentence = template.format(
        workout=workout,
        duration=duration,
        intensity=intensity,
        goal=goal,
        mood=mood,
        instructor=instructor,
        finder=finder,
        optional=optional,
        time_of_day=time_of_day
    )
    doc = nlp.make_doc(sentence)

    ents = []
    used = set()  # track character indices already used by an entity span

    def try_add(label_text, label_name):
        start = sentence.find(label_text)
        if start == -1:
            return  # text not present in this sentence (skip)
        end = start + len(label_text)
        # Skip if any character in this span is already part of another entity
        if any(i in used for i in range(start, end)):
            return
        # Create the span, aligning to token boundaries
        span = doc.char_span(start, end, label=label_name, alignment_mode="contract")
        if span is not None and span.start != span.end:
            ents.append(span)
            # Mark these character positions as used
            used.update(range(start, end))

    # Attempt to add each entity (if present in the sentence)
    try_add(workout, "WORKOUT_TYPE")
    try_add(duration, "DURATION")
    try_add(intensity, "INTENSITY")
    try_add(goal, "GOAL")
    try_add(mood, "MOOD")
    try_add(instructor, "INSTRUCTOR")

    doc.ents = ents
    # Return both the Doc (for .spacy output) and a JSON record for reference
    return doc, {"text": sentence, "entities": [[ent.start_char, ent.end_char, ent.label_] for ent in ents]}

def save_dataset(spacy_path: Path, json_path: Path, n_examples: int):
    db = DocBin(store_user_data=True)
    json_data = []
    for _ in range(n_examples):
        doc, json_rec = generate_example()
        db.add(doc)
        json_data.append(json_rec)
    db.to_disk(spacy_path)
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)

from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    out_dir = Path("voice_assistant/data/entity_data")
    out_dir.mkdir(parents=True, exist_ok=True)

    N_TOTAL = 30000
    all_data = [generate_example() for _ in range(N_TOTAL)]

    seen = {}
    seen_count = 0
    for doc, record in all_data:
        if record["text"] not in seen:
            seen[record["text"]] = (doc, record)
        else:
            seen_count += 1

    print(f"seen count: {seen_count}")
    unique_data = list(seen.values())
    print(f"Unique examples: {len(unique_data)}")

    train_data, dev_data = train_test_split(unique_data, test_size=0.2, random_state=42)

    def save(data, spacy_path, json_path):
        db = DocBin(store_user_data=True)
        json_out = []
        for doc, rec in data:
            db.add(doc)
            json_out.append(rec)
        db.to_disk(spacy_path)
        with open(json_path, "w") as f:
            json.dump(json_out, f, indent=2)

    save(train_data, out_dir / "train.spacy", out_dir / "train.json")
    save(dev_data, out_dir / "dev.spacy", out_dir / "dev.json")

    print(f"Saved {len(train_data)} training and {len(dev_data)} dev examples")

