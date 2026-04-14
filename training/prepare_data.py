import csv, soundfile as sf, io
import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path("/Users/maclcolmolagundoye/INSPIRE_EDGE/chopstix-tts/data/processed")
LIBRITTS_DIR = Path("/Users/maclcolmolagundoye/INSPIRE_EDGE/chopstix-tts/data/raw/libritts_r")
AUDIO_DIR = OUTPUT_DIR / "audio"
OUTPUT_DIR.mkdir(exist_ok=True)
AUDIO_DIR.mkdir(exist_ok=True)

MIN_DURATION = 1.0
MAX_DURATION = 30.0
rows = []

parquet_files = list(LIBRITTS_DIR.glob("**/*.parquet"))
print(f"Found {len(parquet_files)} parquet files")

for i, pq_file in enumerate(parquet_files):
    print(f"Processing {i+1}/{len(parquet_files)}: {pq_file.name}")
    df = pd.read_parquet(pq_file)
    for _, row in df.iterrows():
        try:
            audio_bytes = row["audio"]["bytes"]
            text = row["text_normalized"].strip()
            speaker_id = str(row["speaker_id"])
            audio_buf = io.BytesIO(audio_bytes)
            audio_data, sr = sf.read(audio_buf)
            duration = len(audio_data) / sr
            if not (MIN_DURATION <= duration <= MAX_DURATION):
                continue
            filename = f"{speaker_id}_{len(rows):08d}.wav"
            out_path = AUDIO_DIR / filename
            sf.write(str(out_path), audio_data, sr)
            rows.append({
                "speaker_id": speaker_id,
                "text": text,
                "audio_path": str(out_path),
                "duration": round(duration, 3)
            })
        except Exception as e:
            continue

print(f"Total samples: {len(rows)}")
split = int(len(rows) * 0.9)
for name, subset in [("train", rows[:split]), ("val", rows[split:])]:
    with open(OUTPUT_DIR / f"{name}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["speaker_id","text","audio_path","duration"])
        writer.writeheader()
        writer.writerows(subset)
    print(f"{name}: {len(subset)} samples")
