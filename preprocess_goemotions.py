import pandas as pd
import os
from collections import Counter

IN_PATH = "data/text/goemotions_raw.csv"
OUT_DIR = "data/text"
OUT_PATH = os.path.join(OUT_DIR, "goemotions_7class.csv")

# 1) GoEmotions label names
go_labels = [
    "admiration","amusement","anger","annoyance","approval","caring","confusion",
    "curiosity","desire","disappointment","disapproval","disgust","embarrassment",
    "excitement","fear","gratitude","grief","joy","love","nervousness","optimism",
    "pride","realization","relief","remorse","sadness","surprise","neutral"
]

# 2) 27 → 7 mapping
mapping = {
    "admiration": "happy",
    "amusement": "happy",
    "anger": "angry",
    "annoyance": "angry",
    "approval": "happy",
    "caring": "neutral",
    "confusion": "neutral",
    "curiosity": "neutral",
    "desire": "neutral",
    "disappointment": "angry",
    "disapproval": "angry",
    "disgust": "disgust",
    "embarrassment": "neutral",
    "excitement": "happy",
    "fear": "fear",
    "gratitude": "happy",
    "grief": "sad",
    "joy": "happy",
    "love": "happy",
    "nervousness": "fear",
    "optimism": "happy",
    "pride": "happy",
    "realization": "surprise",
    "relief": "happy",
    "remorse": "sad",
    "sadness": "sad",
    "surprise": "surprise",
    "neutral": "neutral"
}

target_labels = ["angry","disgust","fear","happy","sad","surprise","neutral"]
label_to_id = {lab: i for i, lab in enumerate(target_labels)}

def map_go_to_target(go_indices):
    if not go_indices:
        return "neutral", label_to_id["neutral"]
    
    counts = Counter()
    for idx in go_indices:
        go_name = go_labels[idx]
        tgt = mapping[go_name]
        counts[tgt] += 1
    
    # pick highest count → if tie, prefer non-neutral → then alphabetical
    def sort_key(item):
        label, cnt = item
        neutral_penalty = 1 if label == "neutral" else 0
        return (-cnt, neutral_penalty, label)
    
    best = sorted(counts.items(), key=sort_key)[0][0]
    return best, label_to_id[best]

def normalize(text):
    if not isinstance(text, str):
        return ""
    t = text.strip().lower()
    return t

def main():
    if not os.path.exists(IN_PATH):
        print("Raw file missing:", IN_PATH)
        return
    
    df = pd.read_csv(IN_PATH)
    print("Loaded rows:", len(df))

    rows = []
    dropped = 0

    for _, row in df.iterrows():
        text = normalize(row["text"])
        if len(text) < 2:
            dropped += 1
            continue

        labels = row["go_label_indices"]
        if isinstance(labels, str):
            labels = labels.strip("[]")
            labels = [int(x) for x in labels.split(",") if x.strip().isdigit()]
        
        target_label, target_id = map_go_to_target(labels)

        rows.append({
            "text": text,
            "mapped_label": target_label,
            "mapped_id": target_id
        })
    
    out_df = pd.DataFrame(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    out_df.to_csv(OUT_PATH, index=False)

    print("Saved:", OUT_PATH)
    print("Final rows:", len(out_df))
    print("Dropped short/empty:", dropped)
    print("\nClass distribution:")
    print(out_df["mapped_label"].value_counts())

if __name__ == "__main__":
    main()
