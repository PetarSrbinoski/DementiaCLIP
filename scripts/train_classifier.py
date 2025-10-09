# scripts/train_classifier.py
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm
import open_clip
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from scripts import config


# PyTorch Dataset
class DementiaDataset(Dataset):
    def __init__(self, metadata_df, clip_processor):
        self.df = metadata_df
        self.processor = clip_processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row['spectrogram_path']).convert("RGB")
        image_tensor = self.processor(image)
        with open(row['transcript_path'], 'r', encoding='utf-8') as f:
            transcript = f.read()
        label = torch.tensor(row['label'], dtype=torch.long)
        return image_tensor, transcript, label


# Models
class VisionClassifier(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model
        self.head = nn.Linear(clip_model.visual.output_dim, 2)

    def forward(self, images, **kwargs):
        with torch.no_grad():
            img_features = self.clip_model.encode_image(images).to(torch.float32)
        return self.head(img_features)


class MultimodalClassifier(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model
        fused_dim = clip_model.visual.output_dim + clip_model.text.output_dim
        self.head = nn.Sequential(
            nn.Linear(fused_dim, fused_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fused_dim // 2, 2)
        )

    def forward(self, images, texts, tokenizer):
        with torch.no_grad():
            img_features = self.clip_model.encode_image(images).to(torch.float32)
            text_tokens = tokenizer(texts).to(config.DEVICE)
            text_features = self.clip_model.encode_text(text_tokens).to(torch.float32)
        fused_features = torch.cat((img_features, text_features), dim=1)
        return self.head(fused_features)


# Training and Eval
def run_fold(fold, train_df, val_df, clip_model, preprocess, tokenizer, mode):
    print(f"\n--- Fold {fold + 1}/{config.N_SPLITS} ---")

    train_dataset = DementiaDataset(train_df, preprocess)
    val_dataset = DementiaDataset(val_df, preprocess)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4)

    model = VisionClassifier(clip_model).to(config.DEVICE) if mode == 'vision_only' else MultimodalClassifier(
        clip_model).to(config.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.head.parameters(), lr=config.LEARNING_RATE)

    for epoch in range(config.EPOCHS):
        model.train()
        for images, texts, labels in train_loader:
            images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
            optimizer.zero_grad()
            logits = model(images=images, texts=texts, tokenizer=tokenizer) if mode == 'multimodal' else model(
                images=images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}/{config.EPOCHS} complete.")

    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for images, texts, labels in val_loader:
            images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
            logits = model(images=images, texts=texts, tokenizer=tokenizer) if mode == 'multimodal' else model(
                images=images)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return {
        'acc': accuracy_score(all_labels, all_preds),
        'f1': f1_score(all_labels, all_preds, average='weighted', zero_division=0),
        'roc_auc': roc_auc_score(all_labels, all_probs)
    }


# Main
def main(args):
    print(f"Starting training for mode: {args.mode} on {config.DEVICE}")
    df = pd.read_csv(config.METADATA_FILE)

    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        config.CLIP_MODEL_NAME, pretrained=config.CLIP_PRETRAINED, device=config.DEVICE
    )
    tokenizer = open_clip.get_tokenizer(config.CLIP_MODEL_NAME)

    for param in clip_model.parameters(): param.requires_grad = False

    skf = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.RANDOM_STATE)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['label'])):
        train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]
        metrics = run_fold(fold, train_df, val_df, clip_model, preprocess, tokenizer, args.mode)
        fold_metrics.append(metrics)

    avg_metrics = pd.DataFrame(fold_metrics).mean()
    print("\n--- Cross-Validation Summary ---")
    print(f"Mode: {args.mode}")
    print(f"Average Accuracy: {avg_metrics['acc']:.4f}")
    print(f"Average F1-Score: {avg_metrics['f1']:.4f}")
    print(f"Average ROC-AUC: {avg_metrics['roc_auc']:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Dementia Classification Model")
    parser.add_argument('--mode', type=str, required=True, choices=['vision_only', 'multimodal'], help='Training mode')
    args = parser.parse_args()
    main(args)