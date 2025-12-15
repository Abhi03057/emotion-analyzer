from datasets import load_dataset
import pandas as pd
import os

print("Downloading GoEmotions dataset...")

# Try possible dataset names
try:
    ds = load_dataset("go_emotions")
except Exception:
    ds = load_dataset("mrm8488/goemotions")

print("Dataset loaded successfully.")

rows = []
print("Processing dataset splits...")
for split in ds.keys():
    print(f"  Processing split: {split}, rows = {len(ds[split])}")
    for ex in ds[split]:
        text = ex.get("text") or ex.get("sentence") or ""
        labels = ex.get("labels") or []
        if isinstance(labels, int):
            labels = [labels]
        rows.append({
            "split": split,
            "text": text,
            "go_label_indices": labels
        })

os.makedirs("data/text", exist_ok=True)
out_path = "data/text/goemotions_raw.csv"

df = pd.DataFrame(rows)
df.to_csv(out_path, index=False)

print("Saved:", out_path)
print("Total rows:", len(df))
