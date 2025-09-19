import torch
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

# Directories
DATA_DIR = BASE_DIR / "Data"

# Audio paths for the cookie task
AUDIO_CONTROL_DIR = DATA_DIR / "audio" / "Control" / "cookie" / "raw"
AUDIO_DEMENTIA_DIR = DATA_DIR / "audio" / "Dementia" / "cookie" / "raw"

# Text paths for the cookie task
TEXT_CONTROL_DIR = DATA_DIR / "text" / "Control" / "cookie"
TEXT_DEMENTIA_DIR = DATA_DIR / "text" / "Dementia" / "cookie"
# Output paths
OUTPUTS_DIR = BASE_DIR / "Outputs"
MODEL_SAVE_DIR = OUTPUTS_DIR / "models"
LOG_DIR = OUTPUTS_DIR / "logs"
SPECTROGRAM_DIR = OUTPUTS_DIR / "spectrograms"
TRANSCRIPT_DIR = OUTPUTS_DIR / "transcripts"
METADATA_FILE = OUTPUTS_DIR / "metadata.csv"


# Device config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLIP_MODEL_NAME = 'ViT-B-32'
CLIP_PRETRAINED = 'laion2b_s34b_b79k'

# Spectrogram generation
SR = 16000
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
TRIM_TOP_DB = 25
IMG_SIZE = (224, 224)

# Training parameters
BATCH_SIZE = 16
EPOCHS_VISION = 15
EPOCHS_MULTIMODAL = 25
LR_VISION = 1e-4
LR_MULTIMODAL = 1e-4
N_SPLITS = 5
RANDOM_STATE = 42

# Multimodal model
CLS_LOSS_WEIGHT = 0.6
