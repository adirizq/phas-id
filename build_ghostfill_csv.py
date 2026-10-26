import csv
import json
from pathlib import Path

GHOSTFILL_DIR = Path("dataset/ghostfill")
OUTPUT_CSV = GHOSTFILL_DIR / "raw.csv"

rows = []

for model_dir in sorted(GHOSTFILL_DIR.iterdir()):
    if not model_dir.is_dir():
        continue
    jsonl_path = model_dir / "dataset.jsonl"
    if not jsonl_path.exists():
        continue

    model = model_dir.name

    with open(jsonl_path) as f:
        for line in f:
            doc = json.loads(line)
            author_id = doc["blogspot_author_id"]
            doc_index = doc["doc_index"]
            session_id = f"{author_id}_{doc_index}_{model}"

            for sent_id, sent in enumerate(doc["sentences"], start=1):
                rows.append({
                    "session_id": session_id,
                    "sent_id": sent_id,
                    "sentence_text": sent["text"],
                    "word_count": len(sent["text"].split()),
                    "train_ix": "test",
                    "label": 0 if sent["label"] == "human" else 1,
                    "essay_type": "blog",
                    "worker_id": author_id,
                    "model": model,
                })

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"Written {len(rows)} rows to {OUTPUT_CSV}")
