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


# =========================
# Utility: Repro & Device
# =========================
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
    print("[Hardware Environment Check]")
    print("=" * 60)
    if torch.cuda.is_available():
        print("CUDA is available")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
        current = torch.cuda.current_device()
        print(f"Using device: {torch.cuda.get_device_name(current)}")
        print(f"PyTorch CUDA version: {torch.version.cuda}")
        print(f"cuDNN enabled: {torch.backends.cudnn.enabled}")
    else:
        print("CUDA not available -- running on CPU (this will be slow).")
        print("If you have an NVIDIA GPU, install a CUDA-enabled PyTorch build:")
        print("  pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu121")
        print("Check drivers with: nvidia-smi")
    print("=" * 60)
    return "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# Data
# =========================
CLINICAL_COLS = ["pause_count", "total_pause_duration", "pitch_variation", "type_token_ratio"]


def _shrink_text(txt: str, max_words: int = 256) -> str:
    """Keep early content; avoids CLIP tokenizer truncating everything."""
    if not txt:
        return ""
    words = txt.split()
    if len(words) <= max_words:
        return txt
    return " ".join(words[:max_words])


class DementiaDataset(Dataset):
    """
    - Pre-loads transcripts once and shortens before tokenization.
    - Normalizes clinical features with an externally-fitted scaler (no leakage).
    """

    def __init__(self, df: pd.DataFrame, clip_preprocess, scaler: StandardScaler):
        self.df = df.reset_index(drop=True).copy()
        self.preprocess = clip_preprocess
        self.scaler = scaler

        # Pre-load + shrink transcripts (prevents CLIP from discarding long tails)
        self.transcripts: List[str] = []
        for p in self.df["transcript_path"].tolist():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    txt = f.read()
            except Exception:
                txt = ""
            self.transcripts.append(_shrink_text(txt, max_words=256))

        # Pre-compute normalized clinical features
        feats = self.df[CLINICAL_COLS].fillna(0).to_numpy(dtype=np.float32)
        self.norm_feats = self.scaler.transform(feats).astype(np.float32)

        self.image_paths = self.df["spectrogram_path"].tolist()
        self.labels = self.df["label"].astype(int).to_numpy()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img_tensor = self.preprocess(img)
        transcript = self.transcripts[idx]
        clinical = torch.from_numpy(self.norm_feats[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img_tensor, transcript, clinical, label


# =========================
# Models
# =========================
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VisionClassifier(nn.Module):
    def __init__(self, clip_model: nn.Module):
        super().__init__()
        self.clip_model = clip_model
        embed_dim = clip_model.visual.output_dim
        self.head = VisionHead(embed_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        img_features = self.clip_model.encode_image(images).to(torch.float32)
        return self.head(img_features)


class MultimodalClassifierBasic(nn.Module):
    """Spectrogram + transcript"""
    def __init__(self, clip_model: nn.Module):
        super().__init__()
        self.clip_model = clip_model
        embed_dim = clip_model.visual.output_dim
        fused_dim = embed_dim * 2  # image + text
        self.norm = nn.LayerNorm(fused_dim)
        self.head = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 2),
        )

    def forward(self, images: torch.Tensor, texts_tokenized: torch.Tensor) -> torch.Tensor:
        img = self.clip_model.encode_image(images).to(torch.float32)
        txt = self.clip_model.encode_text(texts_tokenized).to(torch.float32)
        fused = self.norm(torch.cat([img, txt], dim=1))
        return self.head(fused)


class MultimodalClassifierFull(nn.Module):
    """Spectrogram + transcript + clinical (4-dim -> 64-d projection)"""
    def __init__(self, clip_model: nn.Module):
        super().__init__()
        self.clip_model = clip_model
        embed_dim = clip_model.visual.output_dim

        self.proj_clin = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
        )

        fused_dim = embed_dim * 2 + 64
        self.norm = nn.LayerNorm(fused_dim)
        self.head = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
        )

    def forward(
        self,
        images: torch.Tensor,
        texts_tokenized: torch.Tensor,
        clinical: torch.Tensor,
    ) -> torch.Tensor:
        img = self.clip_model.encode_image(images).to(torch.float32)
        txt = self.clip_model.encode_text(texts_tokenized).to(torch.float32)
        clin = self.proj_clin(clinical)
        fused = self.norm(torch.cat([img, txt, clin], dim=1))
        return self.head(fused)


# =========================
# Training helpers
# =========================
@dataclass
class FoldConfig:
    mode: str
    epochs: int
    lr: float
    clip_lr_mult: float
    freeze_epochs: int
    use_class_weights: bool
    use_amp: bool
    num_workers: int
    label_smoothing: float


def set_finetune(clip_model: nn.Module, finetune: bool) -> None:
    for p in clip_model.parameters():
        p.requires_grad = finetune


def set_partial_finetune_last_block(clip_model: nn.Module) -> None:
    """Unfreeze last visual transformer block (+ all LayerNorms)."""
    # freeze everything
    for p in clip_model.parameters():
        p.requires_grad = False

    # unfreeze last visual block
    try:
        last_block = clip_model.visual.transformer.resblocks[-1]
        for p in last_block.parameters():
            p.requires_grad = True
    except Exception:
        pass

    # always unfreeze LayerNorms in the model (helps adapt stats)
    for m in clip_model.modules():
        if isinstance(m, torch.nn.LayerNorm):
            for p in m.parameters():
                p.requires_grad = True


def set_partial_finetune_text_last_block(clip_model: nn.Module) -> None:
    """(Optional) Unfreeze last text transformer block."""
    try:
        last_tblock = clip_model.transformer.resblocks[-1]
        for p in last_tblock.parameters():
            p.requires_grad = True
    except Exception:
        pass


def build_model(mode: str, clip_model: nn.Module) -> nn.Module:
    if mode == "vision_only":
        return VisionClassifier(clip_model)
    if mode == "multimodal_basic":
        return MultimodalClassifierBasic(clip_model)
    return MultimodalClassifierFull(clip_model)


def tokenize_texts(tokenizer, texts: List[str]) -> torch.Tensor:
    # Batch-tokenize a list of strings to a single tensor (CPU)
    return tokenizer(texts)


def _count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_fold(
    fold_idx: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    fold_cfg: FoldConfig,
    tracker: ExperimentTracker,
) -> Dict[str, float]:
    print(f"\n--- Fold {fold_idx + 1}/{config.N_SPLITS} ---", flush=True)

    print("Creating CLIP model and preprocess...", flush=True)
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai", device=config.DEVICE, force_quick_gelu=True
    )
    print("CLIP ready.", flush=True)

    print("Getting tokenizer...", flush=True)
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    print("Tokenizer ready.", flush=True)

    print("Fitting clinical scaler on train split...", flush=True)
    scaler = StandardScaler().fit(train_df[CLINICAL_COLS].fillna(0).values)
    print("Scaler ready.", flush=True)

    print("Building datasets (reads images & transcripts)...", flush=True)
    train_ds = DementiaDataset(train_df, preprocess, scaler)
    val_ds = DementiaDataset(val_df, preprocess, scaler)
    print(f"Datasets ready: {len(train_ds)} train / {len(val_ds)} val", flush=True)

    print("Tokenizing transcripts (batch)...", flush=True)
    train_tokenized = tokenize_texts(tokenizer, train_ds.transcripts).to(config.DEVICE)
    val_tokenized = tokenize_texts(tokenizer, val_ds.transcripts).to(config.DEVICE)
    print("Tokenization done.", flush=True)

    print("Preparing weighted sampler and dataloaders...", flush=True)
    class_counts = train_df["label"].value_counts()
    sample_weights = train_df["label"].map({0: 1.0 / class_counts.get(0, 1), 1: 1.0 / class_counts.get(1, 1)}).values
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    batch_size = min(config.BATCH_SIZE, max(1, len(train_ds)))
    num_workers = fold_cfg.num_workers
    pin_mem = config.DEVICE == "cuda"

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_mem,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_mem,
        persistent_workers=num_workers > 0,
    )
    print("Dataloaders ready. Starting training...", flush=True)

    # Model
    model = build_model(fold_cfg.mode, clip_model).to(config.DEVICE)

    # Optional compile for speed (PyTorch 2+)
    if getattr(config, "USE_COMPILE", False):
        try:
            model = torch.compile(model)  # type: ignore[attr-defined]
            print("Model compiled with torch.compile")
        except Exception:
            print("torch.compile not available; continuing without compilation.")

    # Loss (weights + label smoothing)
    if fold_cfg.use_class_weights and len(class_counts) == 2:
        total = class_counts.sum()
        w = torch.tensor(
            [
                total / (2.0 * class_counts.get(0, 1)),
                total / (2.0 * class_counts.get(1, 1)),
            ],
            dtype=torch.float32,
            device=config.DEVICE,
        )
        criterion = nn.CrossEntropyLoss(weight=w, label_smoothing=fold_cfg.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=fold_cfg.label_smoothing)

    # Optimizer: tiny LR for CLIP backbone
    head_params = [p for n, p in model.named_parameters() if not n.startswith("clip_model.")]
    optimizer = optim.AdamW(
        [
            {"params": model.clip_model.parameters(), "lr": fold_cfg.lr * fold_cfg.clip_lr_mult},
            {"params": head_params, "lr": fold_cfg.lr},
        ],
        weight_decay=0.02,
    )

    # Scheduler: cosine over epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, fold_cfg.epochs))

    # AMP
    scaler_amp = amp.GradScaler(device="cuda", enabled=fold_cfg.use_amp)

    # Freeze-then-partial-unfreeze CLIP
    if fold_cfg.freeze_epochs > 0:
        set_finetune(model.clip_model, False)
        print(f"Fine-tuning CLIP: No (frozen for {fold_cfg.freeze_epochs} epochs)")
    else:
        set_finetune(model.clip_model, True)
        print("Fine-tuning CLIP: Yes (from start)")

    total_trainable = _count_trainable_params(model)
    print(f"Batch size: {batch_size}")
    print(f"Using AMP: {fold_cfg.use_amp}")
    print(f"Trainable parameters: {total_trainable}")
    print(f"Feature cols: {CLINICAL_COLS}")

    best_val = float("inf")
    patience = getattr(config, "EARLY_STOP_PATIENCE", 5)
    stale = 0

    for epoch in range(fold_cfg.epochs):
        # Warmup boundary: partial unfreeze
        if fold_cfg.freeze_epochs > 0 and epoch == fold_cfg.freeze_epochs:
            if getattr(config, "PARTIAL_UNFREEZE_VISUAL_LAST_BLOCK", True):
                set_partial_finetune_last_block(model.clip_model)
                print(f"Epoch {epoch+1}: Partially unfroze CLIP visual (last block + LayerNorms).")
            else:
                set_finetune(model.clip_model, True)
                print(f"Epoch {epoch+1}: Unfroze all CLIP parameters.")
            if getattr(config, "PARTIAL_UNFREEZE_TEXT_LAST_BLOCK", False):
                set_partial_finetune_text_last_block(model.clip_model)
                print(f"Epoch {epoch+1}: Partially unfroze CLIP text (last block).")

        # ----- Train -----
        model.train()
        train_loss = 0.0

        for batch_idx, (images, texts, clinical, labels) in enumerate(train_loader):
            images = images.to(config.DEVICE, non_blocking=True)
            clinical = clinical.to(config.DEVICE, non_blocking=True)
            labels = labels.to(config.DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with amp.autocast(device_type="cuda", enabled=fold_cfg.use_amp):
                if fold_cfg.mode == "vision_only":
                    logits = model(images=images)
                elif fold_cfg.mode == "multimodal_basic":
                    tkn = train_tokenized[batch_idx * batch_size : batch_idx * batch_size + images.size(0)]
                    logits = model(images=images, texts_tokenized=tkn)
                else:
                    tkn = train_tokenized[batch_idx * batch_size : batch_idx * batch_size + images.size(0)]
                    logits = model(images=images, texts_tokenized=tkn, clinical=clinical)
                loss = criterion(logits, labels)

            scaler_amp.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler_amp.step(optimizer)
            scaler_amp.update()
            train_loss += loss.item()

        avg_train = train_loss / max(1, len(train_loader))

        # ----- Validate -----
        model.eval()
        val_loss = 0.0
        all_preds: List[int] = []
        all_labels: List[int] = []
        all_probs: List[float] = []

        with torch.no_grad():
            for batch_idx, (images, texts, clinical, labels) in enumerate(val_loader):
                images = images.to(config.DEVICE, non_blocking=True)
                clinical = clinical.to(config.DEVICE, non_blocking=True)
                labels = labels.to(config.DEVICE, non_blocking=True)

                with amp.autocast(device_type="cuda", enabled=fold_cfg.use_amp):
                    if fold_cfg.mode == "vision_only":
                        logits = model(images=images)
                    elif fold_cfg.mode == "multimodal_basic":
                        tkn = val_tokenized[batch_idx * batch_size : batch_idx * batch_size + images.size(0)]
                        logits = model(images=images, texts_tokenized=tkn)
                    else:
                        tkn = val_tokenized[batch_idx * batch_size : batch_idx * batch_size + images.size(0)]
                        logits = model(images=images, texts_tokenized=tkn, clinical=clinical)
                    loss = criterion(logits, labels)

                val_loss += loss.item()
                probs = torch.softmax(logits, dim=1)[:, 1]
                preds = torch.argmax(logits, dim=1)

                all_preds.extend(preds.detach().cpu().tolist())
                all_labels.extend(labels.detach().cpu().tolist())
                all_probs.extend(probs.detach().cpu().tolist())

        avg_val = val_loss / max(1, len(val_loader))
        acc = accuracy_score(all_labels, all_preds) if all_labels else float("nan")
        f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0) if all_labels else float("nan")
        try:
            roc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) == 2 else float("nan")
        except ValueError:
            roc = float("nan")

        print(
            f"Epoch {epoch + 1}/{fold_cfg.epochs} - "
            f"Train Loss: {avg_train:.4f}, Val Loss: {avg_val:.4f} | "
            f"Acc: {acc:.4f}, F1: {f1:.4f}, ROC-AUC: {roc:.4f}",
            flush=True,
        )
        tracker.log_message(
            f"[Fold {fold_idx+1}][Epoch {epoch+1}] train_loss={avg_train:.4f} "
            f"val_loss={avg_val:.4f} acc={acc:.4f} f1={f1:.4f} roc_auc={roc:.4f}"
        )

        # Early stopping by val loss
        if avg_val + 1e-6 < best_val:
            best_val = avg_val
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        scheduler.step()

    # Final metrics from last validation pass
    return {"acc": acc, "f1": f1, "roc_auc": roc}


# =========================
# Main
# =========================
def main() -> None:
    parser = argparse.ArgumentParser(description="Train Dementia Classification with CLIP (ViT-B/32)")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["vision_only", "multimodal_basic", "multimodal_full"],
        help="vision_only: spectrogram only | multimodal_basic: spectrogram+text | multimodal_full: spectrogram+text+clinical",
    )
    args = parser.parse_args()

    device_used = print_hardware_header()
    set_global_seed(config.RANDOM_STATE)

    print(f"\nStarting training for mode: {args.mode} on {device_used}")
    print("CLIP Model: ViT-B-32")
    print("CLIP Pretrained: openai")
    print(f"Freeze epochs (CLIP): {getattr(config, 'FREEZE_EPOCHS', 3)}")

    if not config.METADATA_FILE.exists():
        raise FileNotFoundError(f"Metadata file not found: {config.METADATA_FILE}. Run preprocessing first.")

    df = pd.read_csv(config.METADATA_FILE)
    if df.empty:
        raise ValueError("Metadata file is empty. Run preprocessing first.")

    print(f"Loaded {len(df)} samples from {config.METADATA_FILE}")
    print("\nClass distribution:")
    print(df["label"].value_counts())

    tracker = ExperimentTracker(
        experiment_name=f"CLIP_ViT-B-32_finetune",
        mode=args.mode,
    )
    tracker.log_config(
        model_name=("VisionClassifier" if args.mode == "vision_only" else f"MultimodalClassifier_{args.mode}"),
        pretrained="openai",
        clip_model_name="ViT-B-32",
        other_config={
            "total_samples": int(len(df)),
            "num_control": int(df["label"].value_counts().get(0, 0)),
            "num_dementia": int(df["label"].value_counts().get(1, 0)),
            "freeze_epochs": getattr(config, "FREEZE_EPOCHS", 3),
            "use_class_weights": getattr(config, "USE_CLASS_WEIGHTS", True),
        },
    )
    tracker.log_message(f"Experiment started: {tracker.experiment_id}")
    tracker.log_message(f"Mode: {args.mode}")
    tracker.log_message(f"Total samples: {len(df)}")

    # Fold configuration (gentle fine-tune defaults)
    epochs = config.EPOCHS_VISION if args.mode == "vision_only" else config.EPOCHS_MULTIMODAL
    lr = config.LR_VISION if args.mode == "vision_only" else config.LR_MULTIMODAL
    fold_cfg = FoldConfig(
        mode=args.mode,
        epochs=epochs,
        lr=lr,
        clip_lr_mult=getattr(config, "CLIP_LR_MULT", 0.03),
        freeze_epochs=getattr(config, "FREEZE_EPOCHS", 3) if getattr(config, "FREEZE_CLIP", True) else 0,
        use_class_weights=getattr(config, "USE_CLASS_WEIGHTS", True),
        use_amp=(config.DEVICE == "cuda"),
        num_workers=min(8, os.cpu_count() or 0),
        label_smoothing=getattr(config, "LABEL_SMOOTHING", 0.05),
    )

    skf = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.RANDOM_STATE)
    fold_metrics: List[Dict[str, float]] = []

    for i, (tr_idx, va_idx) in enumerate(skf.split(df, df["label"])):
        train_df, val_df = df.iloc[tr_idx], df.iloc[va_idx]
        metrics = run_fold(i, train_df, val_df, fold_cfg, tracker)
        fold_metrics.append(metrics)
        print(f"Fold {i + 1} - Acc: {metrics['acc']:.4f}, F1: {metrics['f1']:.4f}, ROC-AUC: {metrics['roc_auc']:.4f}")
        tracker.log_fold_results(i, metrics)
        tracker.log_message(
            f"Fold {i + 1} complete - Acc: {metrics['acc']:.4f}, F1: {metrics['f1']:.4f}, ROC-AUC: {metrics['roc_auc']:.4f}"
        )

    avg = pd.DataFrame(fold_metrics).mean(numeric_only=True)

    tracker.log_average_metrics(avg)
    tracker.copy_metadata()
    tracker.save_results()

    print("\n" + "=" * 50)
    print("--- Cross-Validation Summary ---")
    print(f"Mode: {args.mode}")
    print(f"Average Accuracy: {avg.get('acc', float('nan')):.4f}")
    print(f"Average F1-Score: {avg.get('f1', float('nan')):.4f}")
    print(f"Average ROC-AUC: {avg.get('roc_auc', float('nan')):.4f}")
    print("=" * 50)

    print(tracker.get_summary())
    tracker.log_message("Experiment completed successfully")
    print(f"\nAll results saved to: {tracker.exp_dir}")


if __name__ == "__main__":
    main()
