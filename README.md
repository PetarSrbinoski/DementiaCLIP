# 🧠 DementiaCLIP  
### Fine-tuning OpenCLIP for Multimodal Dementia Classification from Speech  

Python 3.10 + | PyTorch | OpenCLIP | MIT License  

---

## 📘 Overview  

**DementiaCLIP** is a research framework exploring how **multimodal deep learning** can assist early dementia detection from spontaneous speech.  
It fine-tunes **OpenCLIP (ViT-B/16 pretrained on LAION2B)** to combine **audio**, **text**, and **visual** (spectrogram) signals from the classic *Cookie Theft* picture-description task.  

Each modality captures complementary aspects of cognitive decline:  
- 🎧 **Audio** – Acoustic features extracted from patient speech.  
- 💬 **Text** – Transcribed language patterns converted to text embeddings.  
- 👁️ **Vision** – Spectrograms processed through CLIP’s visual encoder.  

These are fused into a shared representation for binary classification (dementia vs control).  

---

## ⚙️ Key Features  

 - **Multimodal Fusion** – Audio + Text + Spectrogram features.  
 - **OpenCLIP Backbone** – `ViT-B/16` trained on `laion2b_s34b_b88k`.  
 - **Gradual Unfreezing** – CLIP frozen for 8 epochs, then partially unfrozen.  
 - **Weighted Sampling & Label Smoothing** – Stabilizes small clinical datasets.  
 - **Mixed-Precision (AMP)** – 30–40 % faster GPU training.  
 - **Early Stopping + Weight Decay** – Prevents overfitting.  
 - **5-Fold Stratified CV** – Robust performance estimation.  
 - **Automatic Experiment Tracking** – Logs metrics, configs, checkpoints, plots.  

---

## Getting Started

### Requirements

-   Python 3.10+
-   A CUDA-enabled GPU is highly recommended for training.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/DementiaCLIP.git](https://github.com/your-username/DementiaCLIP.git)
    cd DementiaCLIP
    ```

2.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## ⚙️ Configuration  

All parameters are defined in **`config.py`**.

| Parameter | Default | Description |
|------------|----------|-------------|
| `CLIP_MODEL_NAME` | `'ViT-B/16'` | OpenCLIP architecture |
| `CLIP_PRETRAINED` | `'laion2b_s34b_b88k'` | Pretrained weights |
| `FREEZE_EPOCHS` | `8` | Epochs to keep CLIP frozen |
| `LR_VISION` | `5e-4` | Vision encoder LR |
| `LR_MULTIMODAL` | `5e-4` | Multimodal classifier LR |
| `CLIP_LR_MULT` | `0.05` | CLIP fine-tuning multiplier |
| `WEIGHT_DECAY` | `0.03` | Regularization |
| `BATCH_SIZE` | `32` | Samples per batch |
| `NUM_FOLDS` | `5` | CV folds |
| `DEVICE` | `cuda` | Auto GPU detection |

---

## 🧹 Preprocessing  

```bash
python preprocess_data.py
````
## 🧠 Training

### 🔹 Train a Single Model

    python train_classifier.py --mode multimodal_full

Available modes:
- vision_only
- multimodal_basic
- multimodal_full (⭐ best performing)

All model checkpoints, logs, metrics, and visualizations are saved under:
`Outputs/experiments/`

---

### 🔹 Run All Models

To benchmark all configurations sequentially:

    python run_all_models.py

This script executes 5-fold cross-validation for each mode and logs:
- Accuracy
- F1-Score
- ROC-AUC
- Confusion matrices
- Training/validation curves

---

## 📈 Experiment Tracking

`experiment_tracker.py` automatically handles experiment logging and output management:
- Records metrics for each epoch and fold
- Saves best model checkpoints
- Generates plots (loss, ROC, confusion matrices)

This ensures complete reproducibility and transparent reporting of results.

---

## 🏆 Best Results so Far (5-Fold CV on RTX 4050 GPU)

| Model              | Accuracy | F1    | ROC-AUC |
|:-------------------|:--------:|:-----:|:-------:|
| Vision Only        | 0.420    | 0.388 | 0.517   |
| Multimodal Basic   | 0.741    | 0.745 | 0.819   |
| **Multimodal Full**| **0.816**| **0.819** | **0.909** |

**Note:** Best model uses partial unfreeze after 8 epochs.  
Realistic accuracy ceiling for this dataset: 0.82–0.85 (F1 ≈ same, ROC-AUC ≈ 0.88–0.91).

---

## 🧩 Modules Explained

### config.py
Central configuration file defining all paths, hyperparameters, and training constants.
Used across every script for reproducibility and consistent experiment setup.

### preprocess_data.py
Handles data preparation:
- Loads raw audio and text from `Data/`
- Extracts MFCCs and Mel-spectrograms
- Cleans and tokenizes transcripts
- Builds stratified train/val/test splits per fold.
Why: Ensures a unified, reproducible preprocessing pipeline across all runs.

### train_classifier.py
Core training script:
- Initializes OpenCLIP (ViT-B/16) backbone
- Applies FREEZE_CLIP then partial unfreeze after 8 epochs
- Uses weighted sampling, label smoothing, AMP, and early stopping
- Logs per-epoch metrics and saves best checkpoints.
Why: Provides flexible multimodal training for comparative experiments.

### run_all_models.py
Pipeline orchestrator:
- Iterates through all model modes
- Invokes `train_classifier.py` automatically
- Collects metrics and aggregates results into summary files
Why: Enables easy end-to-end benchmarking without manual intervention.

### experiment_tracker.py
Experiment management and logging utility:
- Records performance metrics, losses, and config hashes
- Saves checkpoints, CSVs, and plots
- Supports post-analysis and reproducibility
Why: Provides structured tracking for every run and fold.

---

## 🧠 Preventing Overfitting & Handling Class Imbalance

The dataset is relatively small and unbalanced (fewer dementia than control samples).  
To ensure generalization and stability, several complementary strategies are used:

### Overfitting Prevention
- **Early Stopping:** Stops training once validation loss stops improving, preventing the model from memorizing training data.
- **Weight Decay (0.03):** Adds L2 regularization to reduce overfitting by penalizing large weights.
- **Gradual Unfreeze:** Keeps CLIP frozen for 8 epochs, allowing the classifier to learn before fine-tuning the encoder—this stabilizes gradients and avoids catastrophic forgetting.
- **Label Smoothing (0.1):** Reduces overconfidence in predictions and encourages better generalization.
- **Cross-Validation (5-Fold):** Ensures robustness by averaging performance across multiple splits.

### Class Imbalance Handling
- **WeightedRandomSampler:** Balances class frequencies by sampling underrepresented classes more often.
- **Focal Loss:** Focuses training on hard-to-classify samples by down-weighting easy examples, helping the model learn from minority-class errors more effectively.
  - **Why Focal Loss:** Standard cross-entropy loss treats all examples equally, which can bias the model toward the majority class. Focal Loss dynamically scales the loss to emphasize difficult, misclassified samples—crucial when detecting dementia cases that are rarer in the dataset.

Together, these techniques create a model that remains stable during training, generalizes well on unseen data, and maintains sensitivity to minority-class (dementia) predictions.

---

## 💡 Training Notes

- Automatic Mixed Precision (AMP) speeds up training by ≈ 30%
- WeightedRandomSampler balances dementia vs control data
- Early Stopping on validation loss prevents overfitting
- Label Smoothing = 0.1 stabilizes gradients
- Gradual Unfreeze after epoch 8 yields best F1 ≈ 0.82

---

## 🧑‍💻 Author

**Petar Srbinoski**  
Faculty of Computer Science and Engineering (FINKI), UKIM  
petar.srbinoski@gmail.com
