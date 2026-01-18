import os
from tqdm import tqdm

BASE_DIR = "dataset/images"

CATEGORIES = {
    "hate/text_based": "hate_txt",
    "hate/symbol_based": "hate_sym",
    "non_hate/text_based": "nonhate_txt",
    "non_hate/symbol_based": "nonhate_sym"
}

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png")

for folder, prefix in CATEGORIES.items():
    folder_path = os.path.join(BASE_DIR, folder)

    if not os.path.exists(folder_path):
        print(f"Skipping missing folder: {folder_path}")
        continue

    images = [f for f in os.listdir(folder_path) if f.lower().endswith(VALID_EXTENSIONS)]
    images.sort()

    print(f"\nRenaming images in: {folder_path}")

    for idx, image_name in enumerate(tqdm(images, desc=f"{prefix}", unit="image"), start=1):
        ext = os.path.splitext(image_name)[1]
        new_name = f"{prefix}_{idx:03d}{ext}"

        old_path = os.path.join(folder_path, image_name)
        new_path = os.path.join(folder_path, new_name)

        if old_path != new_path:
            os.rename(old_path, new_path)

    print(f"✅ Renamed {len(images)} images in {folder_path}")

print("\n🎉 All folders processed successfully.")
