import json
from pathlib import Path
from collections import defaultdict

DATA_PATH = Path("data/soccernet")

counter = defaultdict(int)

for label_file in DATA_PATH.rglob("Labels-v2.json"):
    with open(label_file) as f:
        labels = json.load(f)
    for a in labels["annotations"]:
        counter[a["label"]] += 1

print("Total par classe sur tout le dataset :")
for label, count in sorted(counter.items(), key=lambda x: x[1]):
    print(f"  {label:<25} {count}")