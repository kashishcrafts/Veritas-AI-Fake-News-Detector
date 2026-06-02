import pandas as pd
import torch
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer
)

# =========================
# LOAD DATA
# =========================

fake = pd.read_csv("data/Fake.csv")
true = pd.read_csv("data/True.csv")

fake["label"] = 0
true["label"] = 1

df = pd.concat([fake, true], ignore_index=True)

print("Total rows:", len(df))

df = df[["text", "label"]]
df.dropna(inplace=True)

df["text"] = df["text"].str.lower()

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# =========================
# TOKENIZER
# =========================

tokenizer = DistilBertTokenizer.from_pretrained(
    "distilbert-base-uncased"
)

print("Tokenizer loaded successfully")

train_encodings = tokenizer(
    list(X_train),
    truncation=True,
    padding=True,
    max_length=128
)

test_encodings = tokenizer(
    list(X_test),
    truncation=True,
    padding=True,
    max_length=128
)

print("Tokenization completed")

# =========================
# DATASET CLASS
# =========================

class NewsDataset(torch.utils.data.Dataset):

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels.reset_index(drop=True)

    def __getitem__(self, idx):

        item = {
            key: torch.tensor(val[idx])
            for key, val in self.encodings.items()
        }

        item["labels"] = torch.tensor(
            self.labels[idx]
        )

        return item

    def __len__(self):
        return len(self.labels)

train_dataset = NewsDataset(
    train_encodings,
    y_train
)

test_dataset = NewsDataset(
    test_encodings,
    y_test
)

print("PyTorch datasets created")

# =========================
# LOAD FRESH DISTILBERT
# =========================

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)

print("Fresh DistilBERT model loaded")

# =========================
# METRICS
# =========================

def compute_metrics(pred):

    labels = pred.label_ids

    preds = np.argmax(
        pred.predictions,
        axis=1
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            labels,
            preds,
            average="binary"
        )
    )

    accuracy = accuracy_score(
        labels,
        preds
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# =========================
# TRAINING SETTINGS
# =========================

training_args = TrainingArguments(
    output_dir="./results",

    num_train_epochs=2,

    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,

    weight_decay=0.01,

    eval_strategy="epoch",
    save_strategy="epoch",

    logging_dir="./logs",
    logging_steps=100
)

print("Training arguments configured")

# =========================
# TRAINER
# =========================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

print("Trainer created successfully")

# =========================
# TRAIN MODEL
# =========================

print("\nStarting training...\n")

trainer.train()

# =========================
# EVALUATE
# =========================

results = trainer.evaluate()

print("\n========== FINAL RESULTS ==========")

print(
    f"Accuracy  : {results['eval_accuracy']:.4f}"
)

print(
    f"Precision : {results['eval_precision']:.4f}"
)

print(
    f"Recall    : {results['eval_recall']:.4f}"
)

print(
    f"F1 Score  : {results['eval_f1']:.4f}"
)

# =========================
# SAVE MODEL
# =========================

model.save_pretrained("saved_model")

tokenizer.save_pretrained("saved_model")

print("\nModel saved successfully!")

