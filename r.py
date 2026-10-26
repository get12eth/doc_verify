from collections import Counter
from datasets import load_dataset

ds = load_dataset("rakshitdabral/Signature-Verification-Dataset")
print(ds)                                          # splits, columns, row counts

for name, split in ds.items():
    print(f"\n== {name}: {len(split)} rows")
    print("features:", split.features)
    lab = split.features.get("label")
    if hasattr(lab, "names"):
        print("class names (first 10):", lab.names[:10], "... total", len(lab.names))
    counts = Counter(split["label"])
    print("distinct labels:", len(counts), "| images per label min/max:", min(counts.values()), max(counts.values()))
    print("first row, non-image fields:", {k: v for k, v in split[0].items() if k != "image"})

#do the splits share labels? If yes, they were split by image, not by signer
sets = {n: set(s["label"]) for n, s in ds.items()}
names = list(sets)
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        print(f"labels shared by {names[i]} and {names[j]}:", len(sets[names[i]] & sets[names[j]]))