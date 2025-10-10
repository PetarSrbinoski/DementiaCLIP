# scripts/run_all_models.py
from __future__ import annotations

import subprocess
import sys
import os
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PYTHON = sys.executable
TRAIN_SCRIPT = BASE_DIR / "scripts" / "train_classifier.py"
OUTPUTS_DIR = BASE_DIR / "Outputs"
EXPERIMENTS_DIR = OUTPUTS_DIR / "experiments"
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

def run_mode(mode: str) -> str:
    print("\n=== Running mode:", mode, "===")
    cmd = [PYTHON, "-u", str(TRAIN_SCRIPT), "--mode", mode]  # unbuffered
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"  # force unbuffered
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, env=env
    )
    lines = []
    assert proc.stdout is not None
    for line in proc.stdout:
        try:
            print(line, end="")
        except UnicodeEncodeError:
            print(line.encode(sys.stdout.encoding, errors="replace").decode(sys.stdout.encoding), end="")
        lines.append(line)
    proc.wait()
    return "".join(lines)

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = EXPERIMENTS_DIR / f"run_all_{timestamp}.txt"

    modes = ["vision_only", "multimodal_basic", "multimodal_full"]
    all_logs = [f"Run timestamp: {timestamp}\nBase dir: {BASE_DIR}\nPython: {PYTHON}\n"]

    for mode in modes:
        all_logs.append(f"\n\n===== MODE: {mode} =====\n")
        all_logs.append(run_mode(mode))

    log_path.write_text("".join(all_logs), encoding="utf-8")
    print("\nCombined log saved to:", log_path)

if __name__ == "__main__":
    main()
