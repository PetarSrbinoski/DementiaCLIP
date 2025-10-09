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
import parselmouth
from parselmouth.praat import call

from scripts import config


# Extract features
def extract_demographics(cha_path):
    try:
        reader = pylangacq.read_chat(cha_path)
        id_header = reader.headers().get("ID", "")
        parts = id_header.split('|')
        if len(parts) >= 5:
            age_str = parts[3].replace(';', '.')
            age = int(float(age_str)) if age_str and age_str != '.' else None
            sex = parts[4] if parts[4] else None
            return age, sex
    except Exception:
        return None, None
    return None, None


def extract_linguistic_features(transcript):
    try:
        tokens = nltk.word_tokenize(transcript.lower())
        if not tokens: return 0.0
        unique_tokens = set(tokens)
        return len(unique_tokens) / len(tokens)
    except Exception:
        return 0.0


def extract_acoustic_features(audio_waveform, sr):
    try:
        sound = parselmouth.Sound(audio_waveform, sampling_frequency=sr)
        intensity = sound.to_intensity()
        silences = call(intensity, "Get intervals where...", 0.0, 0.0, "less than", -25, "equalTo", "silent")
        pause_count = call(silences, "Get number of intervals")
        total_pause_duration = call(silences, "Get total duration")
        pitch = sound.to_pitch()
        f0_stddev = call(pitch, "Get standard deviation", 0, 0, "Hertz")
        return pause_count, total_pause_duration, f0_stddev
    except Exception:
        return 0, 0.0, 0.0


# Helper
def extract_clean_transcript(cha_path):
    try:
        reader = pylangacq.read_chat(cha_path)
        words = reader.words(participants="PAR")
        transcript = " ".join(words)
        transcript = re.sub(r'\[.*?\]', '', transcript)
        transcript = re.sub(r'&\w+', '', transcript)
        return re.sub(r'\s+', ' ', transcript).strip()
    except Exception:
        return ""


def create_spectrogram(audio_waveform, sr, save_path):
    try:
        y_trimmed, _ = librosa.effects.trim(audio_waveform, top_db=config.TRIM_TOP_DB)
        mel_spec = librosa.feature.melspectrogram(y=y_trimmed, sr=sr, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH,
                                                  n_mels=config.N_MELS)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        norm_spec = (log_mel_spec - log_mel_spec.min()) / (log_mel_spec.max() - log_mel_spec.min())
        spec_rgb = np.stack([norm_spec] * 3, axis=-1)
        plt.imsave(save_path, spec_rgb, cmap='viridis', origin='lower')
        return True
    except Exception:
        return False


# Main
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

        participant_files = defaultdict(list)
        for cha_path in source["text_dir"].glob('*-*.cha'):
            participant_id = cha_path.stem.split('-')[0]
            participant_files[participant_id].append(cha_path)

        for pid, cha_paths in tqdm(participant_files.items(), desc=f"Processing {group_name} participants"):
            cha_paths.sort()

            full_transcript = " ".join([extract_clean_transcript(p) for p in cha_paths])
            concatenated_audio = []
            for path in cha_paths:
                wav_path = source["audio_dir"] / f"{path.stem}.wav"
                if wav_path.exists():
                    y, _ = librosa.load(wav_path, sr=config.SR)
                    concatenated_audio.append(y)
            if not concatenated_audio: continue
            full_audio_waveform = np.concatenate(concatenated_audio)

            age, sex = extract_demographics(cha_paths[0])
            ttr = extract_linguistic_features(full_transcript)
            pause_count, pause_duration, f0_stddev = extract_acoustic_features(full_audio_waveform, config.SR)

            transcript_path = config.TRANSCRIPT_DIR / f"{pid}.txt"
            transcript_path.write_text(full_transcript, encoding='utf-8')
            spectrogram_path = config.SPECTROGRAM_DIR / f"{pid}.png"
            if create_spectrogram(full_audio_waveform, config.SR, str(spectrogram_path)):
                metadata_list.append({
                    "participant_id": pid, "label": group_label, "age": age, "sex": sex,
                    "type_token_ratio": ttr, "pause_count": pause_count,
                    "total_pause_duration": pause_duration, "pitch_variation": f0_stddev,
                    "transcript_path": str(transcript_path), "spectrogram_path": str(spectrogram_path)
                })

    df = pd.DataFrame(metadata_list).dropna().reset_index(drop=True)
    df.to_csv(config.METADATA_FILE, index=False)
    print(f"\nPreprocessing complete. Enriched metadata saved to {config.METADATA_FILE}")


if __name__ == "__main__":
    try:
        nltk.data.find('tokenizers/punkt')
    except nltk.downloader.DownloadError:
        nltk.download('punkt')
    main()