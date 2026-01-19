import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ---------------- CONFIG ----------------
DATA_PATH = "dataset/text/texts.csv"
MODEL_DIR = "text_model"

os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------- LOAD DATA ----------------
df = pd.read_csv(DATA_PATH)

# Safety check
assert "text" in df.columns and "label" in df.columns, \
    "CSV must contain 'text' and 'label' columns"

X = df["text"].astype(str)
y = df["label"].astype(int)

# ---------------- TRAIN / VAL SPLIT ----------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------- TF-IDF ----------------
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    max_features=5000,
    ngram_range=(1, 2)
)

X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)

# ---------------- MODEL ----------------
model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

model.fit(X_train_vec, y_train)

# ---------------- EVALUATION ----------------
y_pred = model.predict(X_val_vec)

acc = accuracy_score(y_val, y_pred)
cm = confusion_matrix(y_val, y_pred)

print("\n📊 Validation Accuracy:", round(acc, 3))
print("📉 Confusion Matrix:")
print(cm)
print("\n📄 Classification Report:")
print(classification_report(y_val, y_pred))

# ---------------- SAVE MODEL ----------------
joblib.dump(model, os.path.join(MODEL_DIR, "text_model.pkl"))
joblib.dump(vectorizer, os.path.join(MODEL_DIR, "vectorizer.pkl"))

print("\n💾 Text model saved:")
print("   - text_model/text_model.pkl")
print("   - text_model/vectorizer.pkl")
