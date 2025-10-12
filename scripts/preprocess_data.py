# scripts/preprocess_data.py
from __future__ import annotations

import re
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

import config

warnings.filterwarnings("ignore")

# ==================================Helpers: Files & constants ============================================================
PAR_TAG = "*PAR:"
INV_TAG = "*INV:"
TIMESTAMP_RE = re.compile(r"\x15(\d+)_(\d+)\x15")
BRACKETS_RE = re.compile(r"\[.*?\]")
AMP_CODE_RE = re.compile(r"&[-=+][\w:()]+")        # &=clears:throat, &-uh, &+ha
OPTIONAL_LETTER_RE = re.compile(r"(\w)\((\w)\)")
MULTI_WS_RE = re.compile(r"\s+")
WORD_RE = re.compile(r"[a-z']+")

# Minimal English stopword list (kept small and stable)
STOPWORDS = {
    "the","a","an","and","or","but","in","on","at","to","of","for","with","that","there",
    "be","is","are","was","were","am","i","you","he","she","it","we","they","this","those",
    "these","as","by","from","about","over","under","into","out","up","down","not","no"
}

PRONOUNS = {
    "i","you","he","she","it","we","they","me","him","her","us","them","my","your","his",
    "hers","our","their","mine","yours","ours","theirs","myself","yourself","himself",
    "herself","itself","ourselves","yourselves","themselves"
}

DISFLUENCY_TOKENS = {"uh", "um", "erm", "hmm"}  # we also map &-uh/&+ha/etc. → "uh"
REPAIR_MARKERS = ["[/]", "[//]"]


#========================== TEXT CLEANING ==============================================
def clean_chat_text(raw: str) -> str:
    """
    Make a clean text for feature extraction (keep disfluency tokens as 'uh' for counting).
    - Remove investigator lines, timestamps, bracketed annotations.
    - Normalize CHAT & codes (&-uh, &=clears:throat) to 'uh'.
    - Normalize optional letters: windo(w) -> window, (o)kay -> okay
    """
    s = raw

    # Remove %mor, %gra, %wor tiers completely
    s = "\n".join(
        line for line in s.splitlines()
        if not line.startswith("%mor:") and not line.startswith("%gra:") and not line.startswith("%wor:")
    )

    # Keep only participant utterances and @ headers
    lines = []
    for line in s.splitlines():
        if line.startswith(PAR_TAG):
            lines.append(line[len(PAR_TAG):].strip())
    s = "\n".join(lines)

    # Normalize CHAT "amp" codes to 'uh' (e.g., &=clears:throat, &-uh, &+ha)
    s = AMP_CODE_RE.sub(" uh ", s)

    # Remove timestamps like 10719_13061
    s = TIMESTAMP_RE.sub(" ", s)

    # Remove bracketed annotations
    s = BRACKETS_RE.sub(" ", s)

    # Normalize optional letters in words: windo(w) -> window
    s = OPTIONAL_LETTER_RE.sub(r"\1\2", s)
    s = s.replace("(o)kay", "okay")

    # Collapse underscores and whitespace
    s = re.sub(r"[_]+", " ", s)
    s = MULTI_WS_RE.sub(" ", s).strip()
    return s


def captionize_for_clip(clean_text: str, max_words: int = 80) -> str:
    """
    Make a short, CLIP-friendly caption:
    - Lowercase tokens
    - Remove stopwords and disfluency tokens
    - Keep up to max_words content tokens
    """
    tokens = [t for t in WORD_RE.findall(clean_text.lower()) if t not in STOPWORDS and t not in DISFLUENCY_TOKENS]
    if not tokens:
        return ""
    return " ".join(tokens[:max_words])


# ============================= Timestamp / pause metrics=============================
def extract_utterance_timestamps(raw: str) -> List[Tuple[int, int]]:
    """
    Return list of (start_ms, end_ms) for *PAR: utterances, using end-of-line timestamps like s_e.
    """
    times: List[Tuple[int, int]] = []
    for line in raw.splitlines():
        if line.startswith(PAR_TAG):
            for m in TIMESTAMP_RE.finditer(line):
                try:
                    s = int(m.group(1))
                    e = int(m.group(2))
                    if e >= s:
                        times.append((s, e))
                except Exception:
                    pass
    times.sort(key=lambda x: x[0])
    return times


def pause_stats_from_utterances(utt_times: List[Tuple[int, int]], min_pause_ms: int = 250) -> Tuple[int, float, float]:
    """
    Compute pause stats from gaps between utterances.
    Returns: pause_count, total_pause_sec, avg_pause_sec
    """
    if len(utt_times) < 2:
        return 0, 0.0, 0.0
    gaps = []
    for i in range(len(utt_times) - 1):
        prev_end = utt_times[i][1]
        next_start = utt_times[i + 1][0]
        gap = next_start - prev_end
        if gap >= min_pause_ms:
            gaps.append(gap)
    if not gaps:
        return 0, 0.0, 0.0
    total_sec = sum(gaps) / 1000.0
    avg_sec = total_sec / len(gaps)
    return len(gaps), total_sec, avg_sec


# ===================== Text features= ===============
def basic_text_features(clean_text: str, raw_text_for_repairs: str, utt_count: int, utt_times: List[Tuple[int, int]]) -> Dict[str, float]:
    tokens = WORD_RE.findall(clean_text.lower())

    total_words = len(tokens)
    unique_words = len(set(tokens))
    ttr = (unique_words / total_words) if total_words > 0 else 0.0

    # Pronoun ratio
    pronouns = sum(1 for t in tokens if t in PRONOUNS)
    pronoun_ratio = (pronouns / total_words) if total_words > 0 else 0.0

    # Content ratio = share of tokens not in STOPWORDS
    content_tokens = [t for t in tokens if t not in STOPWORDS]
    content_ratio = (len(content_tokens) / total_words) if total_words > 0 else 0.0

    # Disfluency rate (tokens already normalized to "uh")
    disfluencies = sum(1 for t in tokens if t in DISFLUENCY_TOKENS or t == "uh")
    disfluency_rate = (disfluencies / total_words) if total_words > 0 else 0.0

    # Repair rate from original raw (before cleaning) using markers
    repair_count = 0
    for rm in REPAIR_MARKERS:
        repair_count += raw_text_for_repairs.count(rm)
    repair_rate = (repair_count / max(total_words, 1)) if total_words > 0 else 0.0

    # Speech rate and MLU using utterance counts and time span
    if utt_times:
        total_span_sec = (utt_times[-1][1] - utt_times[0][0]) / 1000.0
    else:
        total_span_sec = 0.0
    speech_rate = (total_words / total_span_sec) if total_span_sec > 0 else 0.0
    mlu_words = (total_words / utt_count) if utt_count > 0 else 0.0

    return {
        "type_token_ratio": float(ttr),
        "pronoun_ratio": float(pronoun_ratio),
        "content_ratio": float(content_ratio),
        "disfluency_rate": float(disfluency_rate),
        "repair_rate": float(repair_rate),
        "speech_rate_wps": float(speech_rate),
        "mlu_words": float(mlu_words),
        "total_words": int(total_words),
        "utterances": int(utt_count),
        "total_span_sec": float(total_span_sec),
    }


#============================== MAKE SPECTROGRAMS =======================================
def make_mel_spectrogram_image(y: np.ndarray, sr: int, save_path: Path) -> bool:
    """
    Speech-centric mel-spectrogram:
      n_fft=400 (25ms), hop=160 (10ms), n_mels=80, fmin=50, fmax=8000
      Log-mel + simple CMVN per file -> scale to 0..1 -> save RGB
    """
    try:
        if y.size == 0:
            return False

        # Trim leading/trailing silence (gentle)
        y_trim, _ = librosa.effects.trim(y, top_db=config.TRIM_TOP_DB)

        if y_trim.size == 0:
            return False

        mel = librosa.feature.melspectrogram(
            y=y_trim,
            sr=sr,
            n_fft=400,
            hop_length=160,
            n_mels=80,
            fmin=50,
            fmax=8000,
            power=2.0,
        )
        log_mel = librosa.power_to_db(mel, ref=np.max)

        # Simple CMVN (per file)
        mu = np.mean(log_mel, axis=None)
        sigma = np.std(log_mel, axis=None) + 1e-8
        log_mel = (log_mel - mu) / sigma

        # Scale to 0..1 for image
        min_v, max_v = log_mel.min(), log_mel.max()
        spec = (log_mel - min_v) / (max_v - min_v + 1e-8)

        # Prepare 3-channel RGB
        spec_rgb = np.stack([spec, spec, spec], axis=-1)

        # Save safely
        plt.imsave(save_path, spec_rgb, origin="lower")
        return True
    except Exception:
        return False


def load_audio_for_paths(paths: List[Path], sr: int) -> np.ndarray:
    """
    Concatenate audio across available .mp3 or .wav files for a participant.
    """
    chunks = []
    for p in paths:
        audio_path_mp3 = p.with_suffix(".mp3")
        audio_path_wav = p.with_suffix(".wav")
        audio_file = audio_path_mp3 if audio_path_mp3.exists() else audio_path_wav
        if audio_file.exists():
            try:
                y, _ = librosa.load(str(audio_file), sr=sr)
                if y.size > 0:
                    chunks.append(y)
            except Exception:
                pass
    if not chunks:
        return np.array([], dtype=np.float32)
    return np.concatenate(chunks).astype(np.float32)


#========================================= MAIN ========================================
def main():
    print("Starting preprocessing (CHAT → features + spectrograms)...")

    # Ensure output folders
    for p in [config.OUTPUTS_DIR, config.SPECTROGRAM_DIR, config.TRANSCRIPT_DIR, config.MODEL_SAVE_DIR]:
        p.mkdir(parents=True, exist_ok=True)

    data_sources = [
        {"text_dir": config.TEXT_CONTROL_DIR, "audio_dir": config.AUDIO_CONTROL_DIR, "label": 0, "group": "Control"},
        {"text_dir": config.TEXT_DEMENTIA_DIR, "audio_dir": config.AUDIO_DEMENTIA_DIR, "label": 1, "group": "Dementia"},
    ]

    rows: List[Dict] = []

    for src in data_sources:
        tdir: Path = src["text_dir"]
        adir: Path = src["audio_dir"]
        label = src["label"]
        group = src["group"]

        if not tdir.exists():
            print(f"[WARN] Missing text dir: {tdir}")
            continue
        if not adir.exists():
            print(f"[WARN] Missing audio dir: {adir}")
            continue

        # Group CHAT files per participant (expects <pid>-*.cha)
        participant_files: Dict[str, List[Path]] = defaultdict(list)
        cha_files = sorted(tdir.glob("*.cha"))
        for cha in cha_files:
            # pid inferred from "<pid>-something.cha"
            pid = cha.stem.split("-")[0]
            participant_files[pid].append(cha)

        print(f"\nGroup: {group} | Participants: {len(participant_files)}")

        for pid, files in tqdm(participant_files.items(), desc=f"Processing {group}"):
            # Combine raw text for this participant
            raw_texts = []
            for cha in files:
                try:
                    raw_texts.append(cha.read_text(encoding="utf-8", errors="ignore"))
                except Exception:
                    pass
            if not raw_texts:
                continue
            raw_combined = "\n".join(raw_texts)

            # Utterance timestamps and counts
            utt_times = extract_utterance_timestamps(raw_combined)
            utt_count = sum(1 for line in raw_combined.splitlines() if line.startswith(PAR_TAG))

            # Clean text for features and caption for CLIP
            text_features = clean_chat_text(raw_combined)   # retains "uh" tokens
            text_clip = captionize_for_clip(text_features, max_words=80)

            if not text_clip:
                # Skip if we have no usable text for CLIP (rare)
                continue

            # Pause metrics from utterance gaps
            pause_count, total_pause_sec, avg_pause_sec = pause_stats_from_utterances(utt_times, min_pause_ms=250)

            # Core text features
            tf = basic_text_features(
                clean_text=text_features,
                raw_text_for_repairs=raw_combined,
                utt_count=utt_count,
                utt_times=utt_times,
            )

            # Load & concatenate audio for all files (try .mp3 then .wav for each)
            audio_wave = load_audio_for_paths([adir / f.stem for f in files], sr=config.SR)
            if audio_wave.size == 0:
                # No audio found — skip participant
                continue

            # Save CLIP caption as transcript file (the model expects a readable text file)
            transcript_path = (config.TRANSCRIPT_DIR / f"{pid}.txt")
            transcript_path.write_text(text_clip, encoding="utf-8")

            # Save spectrogram
            spectrogram_path = (config.SPECTROGRAM_DIR / f"{pid}.png")
            ok = make_mel_spectrogram_image(audio_wave, config.SR, spectrogram_path)
            if not ok:
                continue

            # Demographics (try to parse @ID line; keep None if unavailable)
            age = None
            sex = None
            # Look for @ID participant header lines
            for line in raw_combined.splitlines():
                if line.startswith("@ID:") and "|PAR|" in line:
                    parts = line.split("|")
                    # Typical: @ID: eng|Pitt|PAR|67;00.|male|...
                    try:
                        age_str = parts[3].replace(";", ".")
                        age = int(float(age_str)) if age_str and age_str != "." else None
                    except Exception:
                        age = None
                    try:
                        sex = parts[4].strip() if parts[4].strip() else None
                    except Exception:
                        sex = None
                    break

            # Build row
            row = {
                "participant_id": pid,
                "group": group,
                "label": label,
                "age": age,
                "sex": sex,
                # Paths
                "transcript_path": str(transcript_path),
                "spectrogram_path": str(spectrogram_path),
                # Pause metrics (true gaps from timestamps)
                "pause_count": int(pause_count),
                "total_pause_duration": float(total_pause_sec),
                "avg_pause_duration": float(avg_pause_sec),
                # Text features
                "type_token_ratio": tf["type_token_ratio"],
                "pronoun_ratio": tf["pronoun_ratio"],
                "content_ratio": tf["content_ratio"],
                "disfluency_rate": tf["disfluency_rate"],
                "repair_rate": tf["repair_rate"],
                "speech_rate_wps": tf["speech_rate_wps"],
                "mlu_words": tf["mlu_words"],
                "total_words": tf["total_words"],
                "utterances": tf["utterances"],
                "total_span_sec": tf["total_span_sec"],
            }
            rows.append(row)

    if not rows:
        print("\nERROR: No samples were collected. Check your folders and file formats.")
        return

    df = pd.DataFrame(rows)

    # Final sanity & save
    print(f"\nCollected {len(df)} participants.")
    print("Saving metadata to:", config.METADATA_FILE)
    df.to_csv(config.METADATA_FILE, index=False)

    print("\nDone.")
    print("  - Spectrograms:", config.SPECTROGRAM_DIR)
    print("  - Transcripts:", config.TRANSCRIPT_DIR)
    print("  - Metadata:", config.METADATA_FILE)
    print("\nFeature columns available:")
    print(sorted([
        "pause_count", "total_pause_duration", "avg_pause_duration",
        "type_token_ratio", "pronoun_ratio", "content_ratio",
        "disfluency_rate", "repair_rate", "speech_rate_wps",
        "mlu_words", "total_words", "utterances", "total_span_sec",
    ]))


if __name__ == "__main__":
    main()
