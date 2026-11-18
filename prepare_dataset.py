import os
import shutil
from pathlib import Path

# Project folders
BASE_DIR = Path(__file__).resolve().parent

AUDIO_DIR = BASE_DIR / "raw_data" / "train" / "flac"
PROTOCOL_FILE = (
    BASE_DIR / "raw_data" / "protocols"
    / "ASVspoof2019.LA.cm.train.trn.txt"
)

BONAFIDE_DIR = BASE_DIR / "dataset" / "train" / "bonafide"
SPOOF_DIR = BASE_DIR / "dataset" / "train" / "spoof"


def main():
    # Check required folders and files
    if not AUDIO_DIR.exists():
        print("ERROR: Audio folder not found:", AUDIO_DIR)
        return

    if not PROTOCOL_FILE.exists():
        print("ERROR: Protocol file not found:", PROTOCOL_FILE)
        return

    # Find all audio recordings
    audio_files = list(AUDIO_DIR.rglob("*.flac"))
    audio_files += list(AUDIO_DIR.rglob("*.wav"))

    audio_lookup = {file.stem: file for file in audio_files}

    print("Audio files found:", len(audio_files))

    BONAFIDE_DIR.mkdir(parents=True, exist_ok=True)
    SPOOF_DIR.mkdir(parents=True, exist_ok=True)

    bonafide_count = 0
    spoof_count = 0
    missing_count = 0

    # Read protocol and organise recordings
    with open(PROTOCOL_FILE, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split()

            if len(parts) < 2:
                continue

            audio_id = parts[1]
            label = parts[-1].lower()

            if label not in ("bonafide", "spoof"):
                continue

            source = audio_lookup.get(Path(audio_id).stem)

            if source is None:
                missing_count += 1
                continue

            if label == "bonafide":
                destination = BONAFIDE_DIR / source.name
                bonafide_count += 1
            else:
                destination = SPOOF_DIR / source.name
                spoof_count += 1

            # Move instead of copying to save storage
            shutil.move(str(source), str(destination))

    print("\nDataset preparation completed!")
    print("Bonafide files:", bonafide_count)
    print("Spoof files:", spoof_count)
    print("Missing files:", missing_count)


if __name__ == "__main__":
    main()