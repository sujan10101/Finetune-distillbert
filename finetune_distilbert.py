import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")
np.random.seed(42)
torch.manual_seed(42)

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 512
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5
N_SPLITS = 5
DATASET_PATH = "dataset_clean.csv"
ORIGINAL_DATASET_PATH = "dataset_balanced.csv"  # used only for group IDs in CV
OUTPUT_DIR = "finetuned_distilbert"


class WikipediaDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="binary")
    tn, fp, *_ = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return acc, f1, fpr


def train_one_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="  Training", leave=False):
        optimizer.zero_grad()
        outputs = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            labels=batch["labels"].to(device),
        )
        outputs.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        total_loss += outputs.loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="  Evaluating", leave=False):
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            all_preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
            all_labels.extend(batch["labels"].numpy())
    return np.array(all_labels), np.array(all_preds)


def run_cv(texts, labels, groups):
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    gkf = GroupKFold(n_splits=N_SPLITS)
    texts, labels, groups = np.array(texts), np.array(labels), np.array(groups)
    fold_accs, fold_f1s, fold_fprs = [], [], []

    print(f"\n{'='*60}")
    print(f"Fine-Tuned {MODEL_NAME} — {N_SPLITS}-Fold GroupKFold CV")
    print(f"{'='*60}")

    for fold, (train_idx, val_idx) in enumerate(gkf.split(texts, labels, groups=groups), 1):
        print(f"\nFold {fold}/{N_SPLITS}")

        train_loader = DataLoader(
            WikipediaDataset(texts[train_idx].tolist(), labels[train_idx].tolist(), tokenizer),
            batch_size=BATCH_SIZE, shuffle=True,
        )
        val_loader = DataLoader(
            WikipediaDataset(texts[val_idx].tolist(), labels[val_idx].tolist(), tokenizer),
            batch_size=BATCH_SIZE,
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=2
        ).to(device)

        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
        total_steps = len(train_loader) * EPOCHS
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps,
        )

        best_f1, best_preds, best_labels_val = 0.0, None, None
        for epoch in range(1, EPOCHS + 1):
            loss = train_one_epoch(model, train_loader, optimizer, scheduler, device)
            y_true, y_pred = evaluate(model, val_loader, device)
            acc, f1, fpr = compute_metrics(y_true, y_pred)
            print(f"  Epoch {epoch}/{EPOCHS} — loss: {loss:.4f} | acc: {acc:.4f} | F1: {f1:.4f} | FPR: {fpr:.4f}")
            if f1 > best_f1:
                best_f1, best_preds, best_labels_val = f1, y_pred, y_true

        acc, f1, fpr = compute_metrics(best_labels_val, best_preds)
        fold_accs.append(acc)
        fold_f1s.append(f1)
        fold_fprs.append(fpr)
        print(f"  Best — Accuracy: {acc:.4f} | F1: {f1:.4f} | FPR: {fpr:.4f}")

        fold_dir = os.path.join(OUTPUT_DIR, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)
        model.save_pretrained(fold_dir)
        tokenizer.save_pretrained(fold_dir)

    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Accuracy : {np.mean(fold_accs):.4f} ± {np.std(fold_accs):.4f}")
    print(f"F1 Score : {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}")
    print(f"FPR      : {np.mean(fold_fprs):.4f} ± {np.std(fold_fprs):.4f}")

    pd.DataFrame({
        "fold": list(range(1, N_SPLITS + 1)) + ["Mean", "Std"],
        "Accuracy": fold_accs + [np.mean(fold_accs), np.std(fold_accs)],
        "F1_Score": fold_f1s + [np.mean(fold_f1s), np.std(fold_f1s)],
        "FPR": fold_fprs + [np.mean(fold_fprs), np.std(fold_fprs)],
    }).to_csv(os.path.join(OUTPUT_DIR, "cv_results.csv"), index=False)
    print(f"Results saved to {OUTPUT_DIR}/cv_results.csv")


def load_data(path, original_path):
    df = pd.read_csv(path).dropna(subset=["content"])
    # Load titles from original dataset as group IDs (not used as features)
    orig = pd.read_csv(original_path, usecols=["title"]).loc[df.index]
    groups = orig["title"].fillna("unknown").astype(str).tolist()
    print(f"Loaded {len(df)} samples | label dist: {df['is_ai_flagged'].value_counts().to_dict()}")
    return df["content"].astype(str).tolist(), df["is_ai_flagged"].astype(int).tolist(), groups


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    texts, labels, groups = load_data(DATASET_PATH, ORIGINAL_DATASET_PATH)
    run_cv(texts, labels, groups)
