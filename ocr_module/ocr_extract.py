import os
import cv2
import easyocr

IMAGE_TEXT_DIR = "dataset/images/image_text"

# Initialize OCR reader (English only)
reader = easyocr.Reader(['en'], gpu=False)

def extract_text(image_path):
    img = cv2.imread(image_path)
    results = reader.readtext(img)
    
    text = " ".join([res[1] for res in results])
    return text.strip()

for label in ["hate_text", "non_hate_text"]:
    folder = os.path.join(IMAGE_TEXT_DIR, label)

    print(f"\n📂 Processing {label}")

    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)
        text = extract_text(img_path)

        print(f"\n🖼️ {img_name}")
        print("📝 Extracted Text:")
        print(text)
