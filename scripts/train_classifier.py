# scripts/train_classifier.py
from __future__ import annotations
import argparse
import os
import random
from dataclasses import dataclass
from typing import Dict, List
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch import amp
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import open_clip
import config
from experiment_tracker import ExperimentTracker

# Optional focal loss default: False (usefull for imbalanced dataset like mine)
USE_FOCAL_LOSS = True
FOCAL_GAMMA = 2.0


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.
    Based on: CE * (1 - p_t)^gamma with optional per-class alpha (weight).
    """

    def __init__(self, alpha: torch.Tensor | None = None, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.register_buffer("alpha", alpha if alpha is not None else None)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # CE per-sample (no reduction)
        ce = nn.functional.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        # p_t = prob of the true class
        pt = torch.softmax(logits, dim=1).gather(1, targets.unsqueeze(1)).squeeze(1).clamp_(1e-6, 1.0 - 1e-6)
        focal = (1.0 - pt).pow(self.gamma) * ce
        if self.reduction == "mean":
            return focal.mean()
        elif self.reduction == "sum":
            return focal.sum()
        return focal


# ========================== SEED AND DEVICE ===================================
def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_hardware_header() -> str:
    print("=" * 60)
    print("[Hardware Check]")
    print("=" * 60)
    if torch.cuda.is_available():
        print(f"Using device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        print("WARNING: CUDA not available — running on CPU (slow).")
    print("=" * 60)
    return "cuda" if torch.cuda.is_available() else "cpu"


# ======================DATASET======================================
CLINICAL_COLS = [
    "pause_count",
    "total_pause_duration",
    "speech_rate_wps",
    "type_token_ratio",
]


def _shrink_text(txt: str, max_words: int = 256) -> str:
    """Shorten transcript to avoid CLIP tokenizer truncation."""
    if not txt:
        return ""
    words = txt.split()
    return " ".join(words[:max_words])


class DementiaDataset(Dataset):
    """Loads spectrogram, transcript, and clinical features."""

    def __init__(self, df: pd.DataFrame, clip_preprocess, scaler: StandardScaler):
        self.df = df.reset_index(drop=True)
        self.preprocess = clip_preprocess
        self.scaler = scaler

        self.transcripts = []
        for path in self.df["transcript_path"]:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    txt = f.read()
            except Exception:
                txt = ""
            self.transcripts.append(_shrink_text(txt, max_words=256))

        features = self.df[CLINICAL_COLS].fillna(0).values.astype(np.float32)
        self.norm_feats = self.scaler.transform(features)
        self.image_paths = self.df["spectrogram_path"].tolist()
        self.labels = self.df["label"].astype(int).to_numpy()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img_tensor = self.preprocess(img)
        transcript = self.transcripts[idx]  # return raw text string (tokenize per batch)
        clinical = torch.from_numpy(self.norm_feats[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img_tensor, transcript, clinical, label


# ================================ MODELS ============================

class VisionHead(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 2),
        )

    def forward(self, x):
        return self.net(x)


class VisionClassifier(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model
        self.head = VisionHead(clip_model.visual.output_dim)

    def forward(self, images):
        features = self.clip_model.encode_image(images).to(torch.float32)
        return self.head(features)


class MultimodalClassifierBasic(nn.Module):
    """Spectrogram + transcript"""

    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model
        embed_dim = clip_model.visual.output_dim
        self.norm = nn.LayerNorm(embed_dim * 2)
        self.head = nn.Sequential(
            nn.Linear(embed_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 2),
        )

    def forward(self, images, text_tokens):
        img = self.clip_model.encode_image(images).to(torch.float32)
        txt = self.clip_model.encode_text(text_tokens).to(torch.float32)
        fused = self.norm(torch.cat([img, txt], dim=1))
        return self.head(fused)


class MultimodalClassifierFull(nn.Module):
    """Spectrogram + transcript + clinical"""

    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model
        embed_dim = clip_model.visual.output_dim
        self.proj_clin = nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.LayerNorm(64))
        self.norm = nn.LayerNorm(embed_dim * 2 + 64)
        self.head = nn.Sequential(
            nn.Linear(embed_dim * 2 + 64, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
        )

    def forward(self, images, text_tokens, clinical):
        img = self.clip_model.encode_image(images).to(torch.float32)
        txt = self.clip_model.encode_text(text_tokens).to(torch.float32)
        clin = self.proj_clin(clinical)
        fused = self.norm(torch.cat([img, txt, clin], dim=1))
        return self.head(fused)


# ================================== FINE TUNING ================================

def set_finetune(clip_model, finetune: bool):
    for p in clip_model.parameters():
        p.requires_grad = finetune


def set_partial_finetune_last_block(clip_model, k: int = 1):
    """
    Unfreeze the last k visual transformer blocks + LayerNorms.
    Controlled by config.PARTIAL_UNFREEZE_K.
    """
    for p in clip_model.parameters():
        p.requires_grad = False

    try:
        # Unfreeze last K residual blocks (gentle fine-tune)
        for block in clip_model.visual.transformer.resblocks[-k:]:
            for p in block.parameters():
                p.requires_grad = True
    except Exception:
        pass

    # Always unfreeze normalization layers (to allow stable adaptation)
    for m in clip_model.modules():
        if isinstance(m, torch.nn.LayerNorm):
            for p in m.parameters():
                p.requires_grad = True


def _count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ================================== TRAIN LOOP ==============================

@dataclass
class FoldConfig:
    mode: str
    epochs: int
    lr: float
    clip_lr_mult: float
    freeze_epochs: int
    use_class_weights: bool
    use_amp: bool
    label_smoothing: float
    num_workers: int


def run_fold(fold_idx, train_df, val_df, cfg: FoldConfig, tracker):
    print(f"\n--- Fold {fold_idx + 1}/{config.N_SPLITS} ---")
    print("Initializing model and data...", flush=True)

    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        config.CLIP_MODEL_NAME,
        pretrained=config.CLIP_PRETRAINED,
        device=config.DEVICE,
    )
    tokenizer = open_clip.get_tokenizer(config.CLIP_MODEL_NAME)

    scaler = StandardScaler().fit(train_df[CLINICAL_COLS].fillna(0).values)
    train_ds = DementiaDataset(train_df, preprocess, scaler)
    val_ds = DementiaDataset(val_df, preprocess, scaler)

    # WEIGHTED sampler (training only)
    class_counts = train_df["label"].value_counts()
    weights = train_df["label"].map({0: 1.0 / class_counts[0], 1: 1.0 / class_counts[1]}).values
    sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)

    # Build model
    if cfg.mode == "vision_only":
        model = VisionClassifier(clip_model)
    elif cfg.mode == "multimodal_basic":
        model = MultimodalClassifierBasic(clip_model)
    else:
        model = MultimodalClassifierFull(clip_model)
    model = model.to(config.DEVICE)

    # Freeze CLIP initially
    if cfg.freeze_epochs > 0:
        set_finetune(model.clip_model, False)
        print(f"Fine-tuning CLIP: No (frozen for {cfg.freeze_epochs} epochs)")
    else:
        set_finetune(model.clip_model, True)
        print("Fine-tuning CLIP: Yes (from start)")

    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Trainable params: {_count_trainable(model):,}")

    # === Loss function selection ===
    # Compute class weights tensor on device if requested
    alpha_tensor = None
    if cfg.use_class_weights:
        total = class_counts.sum()
        per_class = torch.tensor(
            [total / (2 * class_counts[0]), total / (2 * class_counts[1])],
            dtype=torch.float32,
            device=config.DEVICE,
        )
        alpha_tensor = per_class  # reuse for focal if enabled

    if USE_FOCAL_LOSS:
        # Focal loss ignores label_smoothing
        criterion = FocalLoss(alpha=alpha_tensor, gamma=FOCAL_GAMMA, reduction="mean")
    else:
        criterion = nn.CrossEntropyLoss(weight=alpha_tensor, label_smoothing=cfg.label_smoothing)

    # Optimizer & LR scheduling
    optimizer = optim.AdamW(
        [
            {"params": model.clip_model.parameters(), "lr": cfg.lr * cfg.clip_lr_mult},
            {"params": [p for n, p in model.named_parameters() if not n.startswith("clip_model.")], "lr": cfg.lr},
        ],
        weight_decay=0.03,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    scaler_amp = amp.GradScaler(device="cuda", enabled=cfg.use_amp)

    best_val = float("inf")
    patience = getattr(config, "EARLY_STOP_PATIENCE", 6)
    stale = 0

    for epoch in range(cfg.epochs):
        # Unfreeze schedule (partial finetune after freeze period)
        if cfg.freeze_epochs > 0 and epoch == cfg.freeze_epochs:
            set_partial_finetune_last_block(model.clip_model, k=config.PARTIAL_UNFREEZE_K)
            print(
                f"\nEpoch {epoch + 1}: Fine-tuning CLIP ENABLED — unfroze last {config.PARTIAL_UNFREEZE_K} block(s) + LayerNorms.")
            trainable_params = _count_trainable(model)
            print(f"Fine-tuning CLIP: YES — Trainable params now: {trainable_params:,}\n")

        # Train
        model.train()
        train_loss = 0.0
        for images, texts, clinical, labels in train_loader:
            images = images.to(config.DEVICE)
            clinical = clinical.to(config.DEVICE)
            labels = labels.to(config.DEVICE)

            if cfg.mode != "vision_only":
                text_tokens = tokenizer(list(texts)).to(config.DEVICE)

            optimizer.zero_grad(set_to_none=True)
            with amp.autocast(device_type="cuda", enabled=cfg.use_amp):
                if cfg.mode == "vision_only":
                    logits = model(images)
                elif cfg.mode == "multimodal_basic":
                    logits = model(images, text_tokens)
                else:
                    logits = model(images, text_tokens, clinical)
                loss = criterion(logits, labels)

            scaler_amp.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler_amp.step(optimizer)
            scaler_amp.update()
            train_loss += loss.item()

        avg_train = train_loss / max(len(train_loader), 1)

        # ---- Validation ----
        model.eval()
        val_loss, preds, probs, labels_all = 0.0, [], [], []
        with torch.no_grad():
            for images, texts, clinical, labels in val_loader:
                images = images.to(config.DEVICE)
                clinical = clinical.to(config.DEVICE)
                labels = labels.to(config.DEVICE)

                if cfg.mode != "vision_only":
                    text_tokens = tokenizer(list(texts)).to(config.DEVICE)

                with amp.autocast(device_type="cuda", enabled=cfg.use_amp):
                    if cfg.mode == "vision_only":
                        logits = model(images)
                    elif cfg.mode == "multimodal_basic":
                        logits = model(images, text_tokens)
                    else:
                        logits = model(images, text_tokens, clinical)
                    loss = criterion(logits, labels)

                val_loss += loss.item()
                p = torch.softmax(logits, dim=1)[:, 1]
                preds.extend(torch.argmax(logits, 1).cpu().tolist())
                probs.extend(p.cpu().tolist())
                labels_all.extend(labels.cpu().tolist())

        acc = accuracy_score(labels_all, preds)
        f1 = f1_score(labels_all, preds, average="weighted", zero_division=0)
        roc = roc_auc_score(labels_all, probs) if len(set(labels_all)) > 1 else 0.5
        avg_val = val_loss / max(len(val_loader), 1)

        print(
            f"Epoch {epoch + 1}/{cfg.epochs} - Train Loss: {avg_train:.4f}, Val Loss: {avg_val:.4f} | "
            f"Acc: {acc:.4f}, F1: {f1:.4f}, ROC-AUC: {roc:.4f}",
            flush=True,
        )

        tracker.log_message(
            f"[Fold {fold_idx + 1}][Epoch {epoch + 1}] train_loss={avg_train:.4f} "
            f"val_loss={avg_val:.4f} acc={acc:.4f} f1={f1:.4f} roc_auc={roc:.4f}"
        )

        if avg_val < best_val:
            best_val, stale = avg_val, 0
        else:
            stale += 1
            if stale >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
        scheduler.step()

    return {"acc": acc, "f1": f1, "roc_auc": roc}


# ================================ MAIN =============================================
def main():
    parser = argparse.ArgumentParser(description="Train Dementia CLIP classifier")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["vision_only", "multimodal_basic", "multimodal_full"])
    args = parser.parse_args()

    device = print_hardware_header()
    set_global_seed(config.RANDOM_STATE)
    print(f"\nStarting training: {args.mode} on {device}\n")

    df = pd.read_csv(config.METADATA_FILE)
    tracker = ExperimentTracker(experiment_name=f"CLIP_{config.CLIP_MODEL_NAME}_finetune", mode=args.mode)

    fold_cfg = FoldConfig(
        mode=args.mode,
        epochs=config.EPOCHS_MULTIMODAL if args.mode != "vision_only" else config.EPOCHS_VISION,
        lr=config.LR_MULTIMODAL if args.mode != "vision_only" else config.LR_VISION,
        clip_lr_mult=config.CLIP_LR_MULT,
        freeze_epochs=config.FREEZE_EPOCHS,
        use_class_weights=config.USE_CLASS_WEIGHTS,
        use_amp=(config.DEVICE == "cuda"),
        label_smoothing=0.0 if USE_FOCAL_LOSS else getattr(config, "LABEL_SMOOTHING", 0.0),
        num_workers=min(8, os.cpu_count() or 0),
    )

    skf = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.RANDOM_STATE)
    fold_metrics = []

    for i, (tr, va) in enumerate(skf.split(df, df["label"])):
        metrics = run_fold(i, df.iloc[tr], df.iloc[va], fold_cfg, tracker)
        fold_metrics.append(metrics)
        print(f"Fold {i + 1} - Acc: {metrics['acc']:.4f}, F1: {metrics['f1']:.4f}, ROC-AUC: {metrics['roc_auc']:.4f}")

    avg = pd.DataFrame(fold_metrics).mean()
    print("\n" + "=" * 50)
    print(f"Mode: {args.mode}")
    print(f"Avg Accuracy: {avg['acc']:.4f}, Avg F1: {avg['f1']:.4f}, Avg ROC-AUC: {avg['roc_auc']:.4f}")
    print("=" * 50)
    print("\nTraining complete.\n")


if __name__ == "__main__":
    main()
