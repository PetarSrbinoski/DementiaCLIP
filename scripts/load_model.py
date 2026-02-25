import torch
import open_clip
import config
from train_classifier import (
    VisionClassifier,
    MultimodalClassifierBasic,
    MultimodalClassifierFull,
    CLINICAL_COLS,
)
from sklearn.preprocessing import StandardScaler


def load_model(checkpoint_path, device="cuda"):
    ckpt = torch.load(checkpoint_path, map_location=device)

    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        ckpt["clip_model_name"],
        pretrained=ckpt["clip_pretrained"],
        device=device,
    )

    mode = ckpt["mode"]

    if mode == "vision_only":
        model = VisionClassifier(clip_model)
    elif mode == "multimodal_basic":
        model = MultimodalClassifierBasic(clip_model)
    else:
        model = MultimodalClassifierFull(clip_model)

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    scaler = StandardScaler()
    scaler.mean_ = ckpt["scaler_mean"]
    scaler.scale_ = ckpt["scaler_scale"]

    tokenizer = open_clip.get_tokenizer(ckpt["clip_model_name"])

    return model, preprocess, tokenizer, scaler
