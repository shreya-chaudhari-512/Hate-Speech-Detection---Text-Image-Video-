import os
import csv

BASE_DIR = "dataset/images"
OUTPUT_CSV = "dataset/labels.csv"

LABEL_MAP = {
    "hate/text_based": 1,
    "hate/symbol_based": 1,
    "non_hate/text_based": 0,
    "non_hate/symbol_based": 0
}

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png")

rows = []

for folder, label in LABEL_MAP.items():
    folder_path = os.path.join(BASE_DIR, folder)

    if not os.path.exists(folder_path):
        continue

    for img_name in os.listdir(folder_path):
        if img_name.lower().endswith(VALID_EXTENSIONS):
            img_path = os.path.join(folder, img_name)
            rows.append([img_path, label])

# Write CSV
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image_path", "label"])
    writer.writerows(rows)

print(f"✅ labels.csv created with {len(rows)} entries")
