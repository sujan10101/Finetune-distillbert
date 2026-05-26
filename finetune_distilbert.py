import os
import logging
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("finetune_distilbert.log"),
    ],
)
logger = logging.getLogger(__name__)

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
    avg_loss = total_loss / len(loader)
    logger.debug("Batch training complete — avg loss: %.4f", avg_loss)
    return avg_loss


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
    logger.info("Using device: %s", device)

    logger.info("Loading tokenizer: %s", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    gkf = GroupKFold(n_splits=N_SPLITS)
    texts, labels, groups = np.array(texts), np.array(labels), np.array(groups)
    fold_accs, fold_f1s, fold_fprs = [], [], []

    logger.info("Starting %d-fold GroupKFold CV with %s", N_SPLITS, MODEL_NAME)

    for fold, (train_idx, val_idx) in enumerate(gkf.split(texts, labels, groups=groups), 1):
        logger.info("--- Fold %d/%d | train=%d  val=%d ---", fold, N_SPLITS, len(train_idx), len(val_idx))

        train_loader = DataLoader(
            WikipediaDataset(texts[train_idx].tolist(), labels[train_idx].tolist(), tokenizer),
            batch_size=BATCH_SIZE, shuffle=True,
        )
        val_loader = DataLoader(
            WikipediaDataset(texts[val_idx].tolist(), labels[val_idx].tolist(), tokenizer),
            batch_size=BATCH_SIZE,
        )

        logger.info("Loading model: %s", MODEL_NAME)
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
        logger.info("Optimizer: AdamW | lr=%.0e | total_steps=%d | warmup=%d",
                    LEARNING_RATE, total_steps, int(0.1 * total_steps))

        best_f1, best_preds, best_labels_val = 0.0, None, None
        for epoch in range(1, EPOCHS + 1):
            logger.info("Epoch %d/%d — training...", epoch, EPOCHS)
            loss = train_one_epoch(model, train_loader, optimizer, scheduler, device)
            logger.info("Epoch %d/%d — evaluating...", epoch, EPOCHS)
            y_true, y_pred = evaluate(model, val_loader, device)
            acc, f1, fpr = compute_metrics(y_true, y_pred)
            logger.info("Epoch %d/%d — loss: %.4f | acc: %.4f | F1: %.4f | FPR: %.4f",
                        epoch, EPOCHS, loss, acc, f1, fpr)
            if f1 > best_f1:
                best_f1, best_preds, best_labels_val = f1, y_pred, y_true
                logger.info("New best F1: %.4f", best_f1)

        acc, f1, fpr = compute_metrics(best_labels_val, best_preds)
        fold_accs.append(acc)
        fold_f1s.append(f1)
        fold_fprs.append(fpr)
        logger.info("Fold %d best — Accuracy: %.4f | F1: %.4f | FPR: %.4f", fold, acc, f1, fpr)

        fold_dir = os.path.join(OUTPUT_DIR, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)
        model.save_pretrained(fold_dir)
        tokenizer.save_pretrained(fold_dir)
        logger.info("Model saved to %s", fold_dir)

    logger.info("=" * 60)
    logger.info("FINAL RESULTS")
    logger.info("Accuracy : %.4f ± %.4f", np.mean(fold_accs), np.std(fold_accs))
    logger.info("F1 Score : %.4f ± %.4f", np.mean(fold_f1s), np.std(fold_f1s))
    logger.info("FPR      : %.4f ± %.4f", np.mean(fold_fprs), np.std(fold_fprs))
    logger.info("=" * 60)

    results_path = os.path.join(OUTPUT_DIR, "cv_results.csv")
    pd.DataFrame({
        "fold": list(range(1, N_SPLITS + 1)) + ["Mean", "Std"],
        "Accuracy": fold_accs + [np.mean(fold_accs), np.std(fold_accs)],
        "F1_Score": fold_f1s + [np.mean(fold_f1s), np.std(fold_f1s)],
        "FPR": fold_fprs + [np.mean(fold_fprs), np.std(fold_fprs)],
    }).to_csv(results_path, index=False)
    logger.info("Results saved to %s", results_path)


def load_data(path, original_path):
    logger.info("Loading dataset from %s", path)
    df = pd.read_csv(path).dropna(subset=["content"])
    logger.info("Loading group IDs from %s", original_path)
    orig = pd.read_csv(original_path, usecols=["title"]).loc[df.index]
    groups = orig["title"].fillna("unknown").astype(str).tolist()
    logger.info("Loaded %d samples | label dist: %s", len(df), df["is_ai_flagged"].value_counts().to_dict())
    return df["content"].astype(str).tolist(), df["is_ai_flagged"].astype(int).tolist(), groups


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("finetune_distilbert.py started")
    logger.info("Config — model: %s | max_len: %d | batch: %d | epochs: %d | lr: %.0e | folds: %d",
                MODEL_NAME, MAX_LENGTH, BATCH_SIZE, EPOCHS, LEARNING_RATE, N_SPLITS)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    texts, labels, groups = load_data(DATASET_PATH, ORIGINAL_DATASET_PATH)
    run_cv(texts, labels, groups)
    logger.info("finetune_distilbert.py finished")
