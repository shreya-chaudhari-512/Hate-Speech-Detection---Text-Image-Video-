from ocr_module.ocr_utils import extract_text
from image_model.predict_image import predict_symbol
from text_model.predict_text import predict_text

TEXT_THRESHOLD = 5  # characters

def predict(image_path):
    extracted_text = extract_text(image_path)

    if len(extracted_text) < TEXT_THRESHOLD:
        print("🖼️ No text detected → Using IMAGE model")
        return predict_symbol(image_path)

    else:
        print("📝 Text detected → Using TEXT model")
        return predict_text(extracted_text)

if __name__ == "__main__":
    img_path = "test_images/input.jpg"

    result = predict(img_path)
    print("✅ FINAL PREDICTION:", result)
