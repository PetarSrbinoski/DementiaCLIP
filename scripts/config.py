# config.py
from pathlib import Path
import torch

#========================= PATHS ================================
BASE_DIR = Path(__file__).parent.parent

DATA_DIR = BASE_DIR / "Data"

# Audio/text
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

#==================== DEVICE =================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#===================== CLIP MODEL ==============================

CLIP_MODEL_NAME = "ViT-B-16"
CLIP_PRETRAINED = "laion2b_s34b_b88k"

#========================== AUDIO / SPECTROGRAMS ======================================

SR           = 16000
N_MELS       = 80
N_FFT        = 400
HOP_LENGTH   = 160
TRIM_TOP_DB  = 25
IMG_SIZE     = (224, 224)  # open_clip transform will handle final resize/crop


#======================== TRAINING PARAMS ======================================

BATCH_SIZE        = 16 #16/32
N_SPLITS          = 5
RANDOM_STATE      = 42

# Epochs
EPOCHS_VISION     = 16
EPOCHS_MULTIMODAL = 20

# Learning rates (conservative for small data)
LR_VISION         = 3e-4
LR_MULTIMODAL     = 3e-4

# Fine-tuning policy
FREEZE_CLIP          = True     # start frozen for stability
FREEZE_EPOCHS        = 8       # longer warmup worked best
CLIP_LR_MULT         = 0.05 #0.05    # CLIP gets 20x smaller LR than the heads
PARTIAL_UNFREEZE_K   = 2        # unfreeze last K visual blocks after warmup

# Regularization / optimization
USE_CLASS_WEIGHTS   = True
LABEL_SMOOTHING     = 0.05
EARLY_STOP_PATIENCE = 8 #6
WEIGHT_DECAY        = 0.03

USE_EXTRA_CLINICAL = True #false
