# config.py
from pathlib import Path
import torch

# =========================
# Paths
# =========================
BASE_DIR = Path(__file__).parent.parent

DATA_DIR = BASE_DIR / "Data"

# Audio/text (Cookie task)
AUDIO_CONTROL_DIR   = DATA_DIR / "audio" / "Control"   / "cookie" / "raw"
AUDIO_DEMENTIA_DIR  = DATA_DIR / "audio" / "Dementia"  / "cookie" / "raw"
TEXT_CONTROL_DIR    = DATA_DIR / "text"  / "Control"   / "cookie"
TEXT_DEMENTIA_DIR   = DATA_DIR / "text"  / "Dementia"  / "cookie"

# Outputs
OUTPUTS_DIR      = BASE_DIR / "Outputs"
MODEL_SAVE_DIR   = OUTPUTS_DIR / "models"
LOG_DIR          = OUTPUTS_DIR / "logs"
SPECTROGRAM_DIR  = OUTPUTS_DIR / "spectrograms"
TRANSCRIPT_DIR   = OUTPUTS_DIR / "transcripts"
METADATA_FILE    = OUTPUTS_DIR / "metadata.csv"

# =========================
# Device
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# CLIP backbone
# (Switched to ViT-B/32 + LAION2B, tends to adapt better on small data)
# =========================
CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"  # was "openai"

# =========================
# Audio / Spectrogram (speech-centric)
# =========================
SR           = 16000
N_MELS       = 80      # speech-friendly
N_FFT        = 400     # ~25 ms @16k
HOP_LENGTH   = 160     # ~10 ms @16k
TRIM_TOP_DB  = 25
IMG_SIZE     = (224, 224)

# =========================
# Training
# =========================
BATCH_SIZE        = 16
N_SPLITS          = 5
RANDOM_STATE      = 42

# Epochs
EPOCHS_VISION     = 16
EPOCHS_MULTIMODAL = 20

# Learning rates
LR_VISION         = 7e-4          # head LR for vision-only
LR_MULTIMODAL     = 1e-3          # ↑ bumped per your request

# Fine-tuning policy
FREEZE_CLIP       = True          # start frozen for stability
FREEZE_EPOCHS     = 5             # was 3; small data benefits from longer warmup
CLIP_LR_MULT      = 0.1           # CLIP gets smaller LR than the head

# Regularization / optimization
USE_CLASS_WEIGHTS = True
LABEL_SMOOTHING   = 0.05          # gentle; helps small datasets
EARLY_STOP_PATIENCE = 8           # allow more epochs to improve before stopping
