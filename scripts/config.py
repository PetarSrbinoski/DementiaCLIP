# config.py
from pathlib import Path
import torch

# ============================================================
# Paths
# ============================================================
BASE_DIR = Path(__file__).parent.parent

DATA_DIR = BASE_DIR / "Data"

AUDIO_CONTROL_DIR = DATA_DIR / "audio" / "Control" / "cookie" / "raw"
AUDIO_DEMENTIA_DIR = DATA_DIR / "audio" / "Dementia" / "cookie" / "raw"

TEXT_CONTROL_DIR = DATA_DIR / "text" / "Control" / "cookie"
TEXT_DEMENTIA_DIR = DATA_DIR / "text" / "Dementia" / "cookie"

OUTPUTS_DIR = BASE_DIR / "Outputs"
MODEL_SAVE_DIR = OUTPUTS_DIR / "models"
LOG_DIR = OUTPUTS_DIR / "logs"
SPECTROGRAM_DIR = OUTPUTS_DIR / "spectrograms"
TRANSCRIPT_DIR = OUTPUTS_DIR / "transcripts"
METADATA_FILE = OUTPUTS_DIR / "metadata.csv"

for _p in (OUTPUTS_DIR, MODEL_SAVE_DIR, LOG_DIR, SPECTROGRAM_DIR, TRANSCRIPT_DIR):
    _p.mkdir(parents=True, exist_ok=True)

# ============================================================
# Device
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# CLIP backbone
# ============================================================
CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "openai"

# ============================================================
# Audio / Spectrogram
# ============================================================
SR = 16000
N_MELS = 128          # richer mel features for small data
N_FFT = 2048
HOP_LENGTH = 512
TRIM_TOP_DB = 20
IMG_SIZE = (224, 224)

# ============================================================
# Training hyperparameters
# ============================================================
BATCH_SIZE = 16

# Fine-tuning friendly defaults
EPOCHS_VISION = 16
EPOCHS_MULTIMODAL = 20

# Head LRs (CLIP LR is scaled via CLIP_LR_MULT)
LR_VISION = 3e-4
LR_MULTIMODAL = 3e-4

# Cross-validation
N_SPLITS = 5
RANDOM_STATE = 42

# ============================================================
# Fine-tuning control / regularization
# ============================================================
# Warmup: keep CLIP frozen at start; then gently unfreeze part of it
FREEZE_CLIP = True
FREEZE_EPOCHS = 3

# Tiny LR for CLIP (relative to head LR). e.g. 0.03 -> 3% of head LR.
CLIP_LR_MULT = 0.03

EARLY_STOP_PATIENCE = 8
USE_CLASS_WEIGHTS = True
LABEL_SMOOTHING = 0.05

# Partial unfreeze options
PARTIAL_UNFREEZE_VISUAL_LAST_BLOCK = True   # unfreeze last visual block + LayerNorms
PARTIAL_UNFREEZE_TEXT_LAST_BLOCK = False    # usually keep text frozen on small data

# Optional: PyTorch 2.x compilation (minor speedup after warmup)
USE_COMPILE = False
