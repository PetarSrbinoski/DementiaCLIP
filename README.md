# DementiaCLIP
### Fine-tuning CLIP for Multimodal Dementia Classification from Speech

Python 3.10+ | PyTorch | MIT License

---

## Overview

**DementiaCLIP** is a pioneering research project exploring the application of vision-language models for early dementia detection. By fine-tuning **CLIP** (`ViT-B-32`), this work aims to identify subtle indicators of cognitive decline from multimodal data captured during the *Cookie Theft* picture description task.

The model fuses three critical modalities to create a comprehensive patient representation:

-   **Audio**: Acoustic features are extracted directly from the patient's speech signal.
-   **Text**: Transcripts of the speech are converted into rich text embeddings.
-   **Vision**: CLIP's visual encoder is applied to spectrograms of the audio, treating sound as an image.

These combined features are fed into a classifier trained to distinguish between participants with dementia and healthy controls, paving the way for more accessible and objective diagnostic tools.

---

## Key Features

-   **Advanced Multimodal Fusion**: Integrates audio, text, and visual (spectrogram) features for robust classification.
-   **State-of-the-Art Backbone**: Leverages the power of OpenAI's pre-trained CLIP model.
-   **Automated & Efficient**: Features automatic GPU detection and mixed-precision (AMP) training for maximum performance.
-   **Smart Training Strategies**: Implements gradual unfreezing of the CLIP encoder and early stopping to prevent overfitting.
-   **Handles Imbalance**: Uses weighted sampling to effectively manage class imbalance in clinical datasets.
-   **Rigorous Validation**: Employs a 5-fold stratified cross-validation scheme for reliable performance evaluation.
-   **Modular Architecture**: Designed for easy extension and experimentation with different fusion techniques.

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

## Configuration

All model and training parameters can be customized in the `configs/config.py` file.

| Parameter             | Default      | Description                                                 |
| --------------------- | ------------ | ----------------------------------------------------------- |
| `CLIP_MODEL_NAME`     | `ViT-B-32`   | The base CLIP model architecture to use as a backbone.      |
| `CLIP_PRETRAINED`     | `openai`     | The set of pre-trained weights to load for the CLIP model.  |
| `FREEZE_CLIP`         | `True`       | If `True`, freezes the CLIP encoder for the first 3 epochs. |
| `EPOCHS_MULTIMODAL`   | `30`         | The total number of training epochs for the multimodal model. |
| `LR_MULTIMODAL`       | `3e-4`       | The learning rate for the multimodal classifier.            |
| `BATCH_SIZE`          | `16`         | The batch size to be used per GPU during training.          |

---

## Training

You can either run the full experimental pipeline or train a single model configuration.

### 1. Run All Models

To train all defined model configurations (vision-only, multimodal-basic, and multimodal-full) and generate comparative results, execute the main script:

```bash
python scripts/run_all_models.py
```
### 2. Train a Single Model
To train a specific model, use the train_classifier.py script with the desired --mode argument.

***Example (training the full multimodal model):***

```bash
python scripts/train_classifier.py --mode multimodal_full
```
All training logs, performance metrics (Accuracy, F1-Score, ROC-AUC), and model checkpoints are saved automatically to the Outputs/experiments/. directory.

## Author
### Petar Srbinoski ***Faculty of Computer Science and Engineering (FINKI), UKIM***

***petar.srbinoski@gmail.com***