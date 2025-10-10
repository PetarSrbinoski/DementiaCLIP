# scripts/preprocess_data.py
import re
import pylangacq
import pandas as pd
import librosa
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
import nltk
import warnings

import config

warnings.filterwarnings('ignore')


# --- Feature Extraction Functions ---
def extract_demographics(cha_path: Path):
    try:
        reader = pylangacq.read_chat(str(cha_path))
        headers = reader.headers()

        id_header = headers.get("ID", "") if isinstance(headers, dict) else ""
        if not id_header:
            return None, None

        parts = id_header.split('|')
        if len(parts) >= 5:
            age_str = parts[3].replace(';', '.')
            age = int(float(age_str)) if age_str and age_str != '.' else None
            sex = parts[4] if parts[4] else None
            return age, sex
    except Exception:
        return None, None
    return None, None


def extract_linguistic_features(transcript: str):
    try:
        if not transcript or len(transcript.strip()) == 0:
            return 0.0

        # Simple tokenization to avoid external dependencies
        words = transcript.lower().split()
        if not words:
            return 0.0

        unique_words = set(words)
        return float(len(unique_words)) / float(len(words))
    except Exception:
        return 0.0


def extract_acoustic_features(audio_waveform: np.ndarray, sr: int):
    """
    Acoustic proxies using librosa (no Praat).
    Returns: pause_count, total_pause_duration (sec), pitch_variation_proxy
    """
    try:
        if audio_waveform is None or len(audio_waveform) == 0:
            return 0, 0.0, 0.0

        # Zero-crossing rate as crude VOX proxy
        zcr = librosa.feature.zero_crossing_rate(audio_waveform)[0]
        pause_count = int(np.sum(zcr < np.mean(zcr) * 0.5))

        # RMS via mel-spectrogram to estimate low-energy regions as pauses
        S = librosa.feature.melspectrogram(y=audio_waveform, sr=sr)
        log_S = librosa.power_to_db(S, ref=np.max)
        mean_energy = float(np.mean(log_S))
        # Count frames below (mean - 10dB)
        frames_below = np.sum(log_S < (mean_energy - 10))
        total_pause_duration = float(frames_below) * (config.HOP_LENGTH / sr)

        # Pitch variation proxy = std of spectral centroid
        D = librosa.stft(audio_waveform, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH)
        magnitude = np.abs(D)
        spec_centroid = librosa.feature.spectral_centroid(S=magnitude, sr=sr)
        f0_stddev = float(np.std(spec_centroid))

        return int(pause_count), float(total_pause_duration), float(f0_stddev)
    except Exception:
        return 0, 0.0, 0.0


# --- Helper Functions ---
def extract_clean_transcript(cha_path: Path):
    try:
        reader = pylangacq.read_chat(str(cha_path))
        words = reader.words(participants="PAR")
        if not words:
            return ""
        transcript = " ".join(words)
        transcript = re.sub(r'\[.*?\]', '', transcript)
        transcript = re.sub(r'&\w+', '', transcript)
        return re.sub(r'\s+', ' ', transcript).strip()
    except Exception:
        return ""


def create_spectrogram(audio_waveform: np.ndarray, sr: int, save_path: str):
    try:
        if audio_waveform is None or len(audio_waveform) == 0:
            return False

        y_trimmed, _ = librosa.effects.trim(audio_waveform, top_db=config.TRIM_TOP_DB)
        if len(y_trimmed) == 0:
            return False

        mel_spec = librosa.feature.melspectrogram(
            y=y_trimmed, sr=sr, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH, n_mels=config.N_MELS
        )
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        # Normalize to [0,1] and stack to 3 channels (RGB)
        mn, mx = np.min(log_mel_spec), np.max(log_mel_spec)
        denom = (mx - mn) + 1e-8
        norm_spec = (log_mel_spec - mn) / denom
        spec_rgb = np.stack([norm_spec] * 3, axis=-1)  # (H, W, 3), already "RGB"

        # Save (no cmap since we already have RGB)
        plt.imsave(save_path, spec_rgb, origin='lower')
        return True
    except Exception:
        return False


# --- Main Preprocessing Logic ---
def main():
    print("Starting data preprocessing with feature extraction...")
    for path in [config.OUTPUTS_DIR, config.SPECTROGRAM_DIR, config.TRANSCRIPT_DIR, config.MODEL_SAVE_DIR]:
        path.mkdir(exist_ok=True, parents=True)

    metadata_list = []
    data_sources = [
        {"text_dir": config.TEXT_CONTROL_DIR, "audio_dir": config.AUDIO_CONTROL_DIR, "label": 0, "group": "Control"},
        {"text_dir": config.TEXT_DEMENTIA_DIR, "audio_dir": config.AUDIO_DEMENTIA_DIR, "label": 1, "group": "Dementia"}
    ]

    for source in data_sources:
        group_name, group_label = source["group"], source["label"]
        print(f"\nProcessing group: {group_name}")

        # Verify directories exist
        if not source["text_dir"].exists():
            print(f"ERROR: Text directory not found: {source['text_dir']}")
            continue
        if not source["audio_dir"].exists():
            print(f"ERROR: Audio directory not found: {source['audio_dir']}")
            continue

        participant_files = defaultdict(list)
        cha_files = list(source["text_dir"].glob('*-*.cha'))
        print(f"Found {len(cha_files)} .cha files")

        for cha_path in cha_files:
            participant_id = cha_path.stem.split('-')[0]
            participant_files[participant_id].append(cha_path)

        print(f"Found {len(participant_files)} unique participants")

        for pid, cha_paths in tqdm(participant_files.items(), desc=f"Processing {group_name} participants"):
            cha_paths.sort()

            # Concatenate transcripts
            transcripts = [extract_clean_transcript(p) for p in cha_paths]
            full_transcript = " ".join([t for t in transcripts if t])
            if not full_transcript.strip():
                continue

            # Concatenate audio across files
            audio_segments = []
            for path in cha_paths:
                # .mp3 or .wav
                audio_path = source["audio_dir"] / f"{path.stem}.mp3"
                if not audio_path.exists():
                    audio_path = source["audio_dir"] / f"{path.stem}.wav"

                if audio_path.exists():
                    try:
                        y, _ = librosa.load(str(audio_path), sr=config.SR, mono=True)
                        audio_segments.append(y)
                    except Exception:
                        pass

            if not audio_segments:
                continue
            full_audio_waveform = np.concatenate(audio_segments, dtype=np.float32)

            # Demographics and features
            age, sex = extract_demographics(cha_paths[0])
            ttr = extract_linguistic_features(full_transcript)
            pause_count, pause_duration, f0_stddev = extract_acoustic_features(full_audio_waveform, config.SR)

            # Save transcript
            transcript_path = config.TRANSCRIPT_DIR / f"{pid}.txt"
            transcript_path.write_text(full_transcript, encoding='utf-8')

            # Save spectrogram
            spectrogram_path = config.SPECTROGRAM_DIR / f"{pid}.png"
            ok = create_spectrogram(full_audio_waveform, config.SR, str(spectrogram_path))
            if not ok:
                continue

            metadata_list.append({
                "participant_id": pid,
                "label": group_label,
                "age": age,
                "sex": sex,
                "type_token_ratio": ttr,
                "pause_count": pause_count,
                "total_pause_duration": pause_duration,
                "pitch_variation": f0_stddev,
                "transcript_path": str(transcript_path),
                "spectrogram_path": str(spectrogram_path)
            })

    if metadata_list:
        df = pd.DataFrame(metadata_list)
        print(f"\nTotal samples collected: {len(df)}")
        print(f"Samples after processing: {len(df)}")
        df.to_csv(config.METADATA_FILE, index=False)
        print(f"\n✓ Preprocessing complete!")
        print(f"✓ Metadata saved to {config.METADATA_FILE}")
        print(f"✓ Spectrograms saved to {config.SPECTROGRAM_DIR}")
        print(f"✓ Transcripts saved to {config.TRANSCRIPT_DIR}")
        print(f"\nData distribution:")
        print(df['label'].value_counts())
    else:
        print("\nERROR: No metadata was collected. Check your data directories and file formats.")


if __name__ == "__main__":
    # Download required NLTK data (best effort)
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        try:
            nltk.download('punkt_tab', quiet=True)
        except Exception:
            pass

    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        try:
            nltk.download('punkt', quiet=True)
        except Exception:
            pass

    main()
