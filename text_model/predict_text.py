def predict_text(text):
    hate_words = ["kill", "hate", "die", "destroy"]

    for word in hate_words:
        if word in text.lower():
            return "HATE"

    return "NON-HATE"
