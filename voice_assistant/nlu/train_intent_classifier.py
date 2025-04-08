import os
import json
import torch
import pandas as pd
import numpy as np
import evaluate
from datasets import Dataset, DatasetDict
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.model_selection import train_test_split

print(f"[INFO] Using device: {'GPU' if torch.cuda.is_available() else 'CPU'}")

# === Config ===
MODEL_DIR = "voice_assistant/models/intent_classifier"
DATA_PATH = "voice_assistant/data/intent_training_data.csv"

# === Load & prepare data ===
df = pd.read_csv(DATA_PATH)
label_list = df["label"].unique().tolist()
label_to_id = {label: i for i, label in enumerate(label_list)}
df["label_id"] = df["label"].map(label_to_id)

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
dataset = DatasetDict({
    "train": Dataset.from_pandas(train_df),
    "test": Dataset.from_pandas(test_df),
})

# === Tokenization ===
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def tokenize(example):
    return tokenizer(example["text"], padding="max_length", truncation=True)

dataset = dataset.map(tokenize)
dataset = dataset.remove_columns(["text", "label"])
dataset = dataset.rename_column("label_id", "labels")

# === Model ===
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(label_list)
)

# === Metrics ===
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

# === Training config ===
training_args = TrainingArguments(
    output_dir=MODEL_DIR,
    evaluation_strategy="epoch",
    save_strategy="epoch", 
    logging_strategy="epoch",
    num_train_epochs=20,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_dir="./logs",
    load_best_model_at_end=True,
    warmup_ratio=0.1,              
    lr_scheduler_type="linear",    
    metric_for_best_model="accuracy",
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# === Train ===
trainer.train()

# === Save final model/tokenizer/label map ===
model.save_pretrained(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)

with open(os.path.join(MODEL_DIR, "label_map.json"), "w") as f:
    json.dump(label_to_id, f)

print("Model, tokenizer, and label map saved.")
