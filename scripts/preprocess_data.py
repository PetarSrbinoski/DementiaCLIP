from scripts import config
import re
import pylangacq
import pandas as pd
import librosa
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torchaudio import transforms as T
from pathlib import Path
from collections import defaultdict


def extract_clean_transcript(cha_path):
    try:
        reader = pylangacq.read_chat(cha_path)
        words = reader.words(participants="PAR")
        transcript = " ".join(words)
        transcript = re.sub(r'\[\s*\+\s*.*?\]', '', transcript)
        transcript = re.sub(r'&\w+', '', transcript)
        return re.sub(r'\s+', ' ', transcript).strip()
    except Exception as e:
        print(f"Error processing transcript for {cha_path}: {e}")
        return ""


def extract_mmse(cha_path):
    try:
        reader = pylangacq.read_chat(cha_path)
        id_header = reader.headers().get("ID", "")
        if not id_header or len(parts := id_header.split('|')) < 7:
            return np.nan
        mmse_str = parts[6].strip()
        return int(mmse_str) if mmse_str.isdigit() else np.nan
    except Exception:
        return np.nan


def create_spectrogram(audio_waveform, sr, save_path):
    try:
        y_trimmed, _ = librosa.effects.trim(audio_waveform, top_db=config.TRIM_TOP_DB)
        mel_spec = librosa.feature.melspectrogram(y=y_trimmed, sr=sr, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH,
                                                  n_mels=config.N_MELS)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        log_mel_spec_tensor = torch.from_numpy(log_mel_spec).unsqueeze(0)
        spec_transform = torch.nn.Sequential(T.FrequencyMasking(freq_mask_param=15), T.TimeMasking(time_mask_param=35))
        log_mel_spec = spec_transform(log_mel_spec_tensor).squeeze(0).numpy()
        norm_spec = (log_mel_spec - log_mel_spec.min()) / (log_mel_spec.max() - log_mel_spec.min())
        spec_rgb = np.stack([norm_spec] * 3, axis=-1)
        plt.imsave(save_path, spec_rgb, cmap='viridis', origin='lower')
        return True
    except Exception as e:
        print(f"Error creating spectrogram: {e}")
        return False


def main():
    print("Starting preprocessing for split files\n")
    for path in [config.OUTPUTS_DIR, config.SPECTROGRAM_DIR, config.TRANSCRIPT_DIR, config.MODEL_SAVE_DIR,
                 config.LOG_DIR]:
        path.mkdir(exist_ok=True)

    metadata_list = []

    data_sources = [
        {"text_dir": config.TEXT_CONTROL_DIR, "audio_dir": config.AUDIO_CONTROL_DIR, "label": 0, "group": "Control"},
        {"text_dir": config.TEXT_DEMENTIA_DIR, "audio_dir": config.AUDIO_DEMENTIA_DIR, "label": 1, "group": "Dementia"}
    ]

    for source in data_sources:
        gr_name = source["group"]
        print(f"rocessing group: {gr_name} \n")

        # group all file parts by id
        participant_files = defaultdict(list)
        all_cha_files = source["text_dir"].glob('*-*.cha')
        for cha_path in all_cha_files:
            participant_id = cha_path.stem.split('-')[0]
            participant_files[participant_id].append(cha_path)

        # process each participant

        for pid, cha_paths in participant_files.items():
            cha_paths.sort()

            # join the transcript parts
            full_transcript = " ".join([extract_clean_transcript(p) for p in cha_paths])

            # join the audio parts
            concat_audio = []
            for path in cha_paths:
                wav_path = source["audio_dir"] / f"{path.stem}.wav"
                if wav_path.exists():
                    y, _ = librosa.load(wav_path, sr=config.SR)
                    concat_audio.append(y)
                else:
                    print(f"Missing audio part {wav_path.name}")
            if not concat_audio:
                print(f"No audio found for participant {pid}.")
                continue

            full_audio_wav = np.concatenate(concat_audio)

            # get metadata
            mmse = extract_mmse(cha_paths[0])
            if pd.isna(mmse):
                print(f"Could not extract valid MMSE for {pid}.")
                continue

            # save the transcript
            transcript_path = config.TRANSCRIPT_DIR / f"{pid}.txt"
            transcript_path.write_text(full_transcript, encoding='utf-8')

            # make and save spectrogram from full audio file
            spectrogram_path = config.SPECTROGRAM_DIR / f"{pid}.png"
            success = create_spectrogram(full_audio_wav, config.SR, str(spectrogram_path))

            if success:
                metadata_list.append({
                    "participant_id": pid,
                    "label": source["lavel"],
                    "mmse": mmse,
                    "transcript_path": str(transcript_path),
                    "spectrogram_path": str(spectrogram_path)
                })

            #make and save final metadata csv
            df = pd.DataFrame(metadata_list)
            df['mmse'] = df['mmse'].astype(int)

            df.to_csv(config.METADATA_FILE, index=False)
            print(f"\nPreprocessing complete. Metadata saved to {config.METADATA_FILE}")
            print(f"Found {len(df)} complete participant sessions.")


if __name__ == '__main__':
    main()
