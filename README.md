# 🧠 DementiaCLIP
### Fine-tuning OpenCLIP for Multimodal Dementia Classification from Speech

Python 3.10+ | PyTorch | OpenCLIP 

---

## 📘 Overview

**DementiaCLIP** is a research framework exploring how **multimodal deep learning** can assist early dementia detection from spontaneous speech.  
It fine-tunes **OpenCLIP (ViT-B-16 pretrained on LAION2B)** to combine **audio**, **text**, and **visual** (spectrogram) signals from the classic *Cookie Theft* picture-description task.

Each modality captures complementary aspects of cognitive decline:  
- 🎧 **Audio** – Acoustic correlates from speech rendered as Mel-spectrogram images.  
- 💬 **Text** – Transcribed language patterns tokenized for CLIP text encoder.  
- 👁️ **Vision** – Spectrograms through CLIP’s visual encoder.

These are fused into a shared representation for binary classification (dementia vs control).

---

## ⚙️ Key Features

- **Multimodal Fusion** – Spectrogram (**vision**) + transcript (**text**) + optional **clinical** features.  
- **OpenCLIP Backbone** – `ViT-B-16` with `laion2b_s34b_b88k` weights.  
- **Staged Fine-Tuning** – CLIP fully frozen for warmup, then **partial unfreeze of last visual block**.  
- **Focal Loss (γ=1) + Class Weights** – Better focus on hard/minority cases.  
- **Weighted Sampling & Label Smoothing (configurable)** – Robust on small clinical datasets.  
- **Warmup-Cosine LR Scheduler** – Linear warmup then cosine decay; preserves LR ratios.  
- **Grad Clipping (‖g‖≤1.0) & AMP** – Stable and fast training on GPU.  
- **Early Stopping (AUC-based)** – Stops when ROC-AUC plateaus.  
- **5-Fold Stratified CV** – Reliable performance estimation.  
- **Automatic Experiment Tracking** – Metrics, configs, checkpoints, and plots.

---

## 🏁 Latest Results (5-Fold CV, RTX 4050)

**Run:** `2025-10-17_234935` with `ViT-B-16 | laion2b_s34b_b88k`, **partial unfreeze after 10 epochs**, warmup-cosine schedule, focal loss (γ=1), grad clip 1.0.

| Model   | Accuracy |   F1   | ROC-AUC |
|:--------------------|:-------:|:------:|:------:|
| Vision Only | 0.3891  | 0.3346 | 0.5206 |
| Multimodal Basic| 0.7542  | 0.7576 | 0.8313 |
| **Multimodal Full** | **0.8155** | **0.8177** | **0.8767** |

> **Note:** On this dataset, a realistic ceiling is **~0.82–0.85** accuracy (F1 similar) and **~0.88–0.91** ROC-AUC.

---

## 🚀 Getting Started

### Requirements
- Python **3.10+**
- CUDA-enabled GPU recommended

### Installation
```bash
git clone https://github.com/your-username/DementiaCLIP.git
cd DementiaCLIP
pip install -r requirements.txt
```

---

## ⚙️ Configuration

All parameters live in **`config.py`**.

| Parameter | Default | Description |
|---|---:|---|
| `CLIP_MODEL_NAME` | `"ViT-B-16"` | OpenCLIP architecture |
| `CLIP_PRETRAINED` | `"laion2b_s34b_b88k"` | Pretrained weights |
| `DEVICE` | auto (`cuda` if available) | Device selection |
| `BATCH_SIZE` | `16` | Samples per batch |
| `N_SPLITS` | `5` | Stratified CV folds |
| `RANDOM_STATE` | `42` | Reproducibility |
| `EPOCHS_VISION` | `20` | Vision-only epochs |
| `EPOCHS_MULTIMODAL` | `30` | Multimodal epochs |
| `LR_VISION` | `2e-4` | Vision mode base LR |
| `LR_MULTIMODAL` | `2e-4` | Multimodal base LR |
| `FREEZE_CLIP` | `True` | Start with CLIP frozen |
| `FREEZE_EPOCHS` | `10` | Warmup epochs (frozen) |
| `CLIP_LR_MULT` | `0.02` | Multiplier for CLIP params |
| `PARTIAL_UNFREEZE_K` | `1` | Unfreeze last K visual blocks |
| `USE_CLASS_WEIGHTS` | `True` | CE/Focal α by class freq |
| `LABEL_SMOOTHING` | `0.01` | Used only if CE is active |
| `EARLY_STOP_PATIENCE` | `8` | AUC-based early stop patience |
| `WEIGHT_DECAY` | `0.03` | AdamW weight decay |
| `USE_EXTRA_CLINICAL` | `True` | Adds extra clinical feats |
| `USE_SCHEDULER` | `True` | Enable LR scheduling |
| `SCHEDULER_TYPE` | `"warmup_cosine"` | LR scheduler |
| `WARMUP_EPOCHS` | `4` | Linear warmup length |
| `MIN_LR_FACTOR` | `0.05` | Cosine floor (× base LR) |
| `GRAD_CLIP_NORM` | `1.0` | Global grad-norm cap |

> **Loss:** The training script enables **Focal Loss (γ=1)** by default (`USE_FOCAL_LOSS=True`). When Focal Loss is active, `LABEL_SMOOTHING` is ignored. Class weights (α) are still used.

---

## 🧹 Data & Preprocessing

Configured in `config.py`:

- **Audio/Text** under `Data/`  
- **Outputs**: models, logs, spectrograms, transcripts, and metadata CSV under `Outputs/`

Run preprocessing:
```bash
python preprocess_data.py
```

Spectrograms are resized to **224×224**, audio is standardized (e.g., 16 kHz, 80 Mel bins), transcripts are trimmed to **≤256 words** to avoid CLIP tokenizer truncation.

**Clinical features** (used when `USE_EXTRA_CLINICAL=True`):  
Base: `pause_count`, `total_pause_duration`, `speech_rate_wps`, `type_token_ratio`  
Extra: `avg_pause_duration`, `mlu_words`

---

## 🧠 Training

### Train one mode
```bash
python scripts/train_classifier.py --mode multimodal_full
# or: --mode vision_only | multimodal_basic
```

Artifacts (checkpoints, logs, metrics, plots) are saved in:
```
Outputs/experiments/
```

### Run all modes (benchmark)
```bash
python run_all_models.py
```

---

## 📈 Experiment Tracking

`experiment_tracker.py` logs for every fold/epoch:
- Train/val loss, **Accuracy**, **F1**, **ROC-AUC**
- LR schedule steps
- Best checkpoints and diagnostic plots (loss curves, ROC, CM)

---

## 🧩 Modules

- **`config.py`** – Centralized paths & hyperparameters.  
- **`preprocess_data.py`** – Builds spectrograms, cleans transcripts, compiles metadata.  
- **`scripts/train_classifier.py`** – Training loop with staged unfreeze, focal loss, scheduler, AMP, grad-clip, early stop.  
- **`run_all_models.py`** – Orchestrates 5-fold CV for all modes and aggregates results.  
- **`experiment_tracker.py`** – Filesystem-based experiment logging & plotting.

---

## 🔬 Why This Setup Works

### Overfitting Prevention
- **Warmup then partial unfreeze (K=1):** Stabilizes heads first, then gently adapts CLIP.  
- **AdamW + weight decay (0.03):** Regularizes large weights.  
- **Grad clipping (1.0):** Prevents rare exploding steps after unfreeze.  
- **AUC-based early stopping:** Optimizes for ranking quality, not just accuracy.

### Imbalance Handling
- **WeightedRandomSampler** balances batches.  
- **Focal Loss (γ=1) + α (class weights):** Emphasizes hard/minority samples; more stable than plain CE on small n.

### Optimization Details
- **Warmup-Cosine:** Linear warmup (4 epochs) → cosine decay to **5%** of base LR; maintains CLIP/head LR ratio (`CLIP_LR_MULT=0.02`).  
- **AMP:** Faster training and lower VRAM usage on RTX 4050.

---

## 💡 Training Notes

- Texts are truncated to **≤256 words** to minimize CLIP tokenizer truncation artifacts.  
- Clinical features are **standardized per fold** using `StandardScaler` fit on the training split only.  
- After warmup, the **CLIP LR is halved at unfreeze** and only the **last visual block + LayerNorms** are trainable.  
- Typical early stopping for multimodal occurs around **epochs 14–20**.

---

   

## 🧑‍💻 Author

**Petar Srbinoski**  
Faculty of Computer Science and Engineering (FINKI), UKIM  
petar.srbinoski@gmail.com
