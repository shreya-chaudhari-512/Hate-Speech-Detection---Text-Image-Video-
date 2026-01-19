import cv2
import easyocr

reader = easyocr.Reader(['en'], gpu=False)

def extract_text(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return ""

    results = reader.readtext(img)
    text = " ".join([res[1] for res in results])
    return text.strip()
