import pandas as pd
from sklearn.model_selection import train_test_split

LABELS_CSV = "dataset/labels.csv"
TRAIN_CSV = "dataset/train.csv"
VAL_CSV = "dataset/val.csv"

# Load labels
df = pd.read_csv(LABELS_CSV)

# Stratified split (VERY IMPORTANT)
train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

# Save splits
train_df.to_csv(TRAIN_CSV, index=False)
val_df.to_csv(VAL_CSV, index=False)

print(f"✅ Train samples: {len(train_df)}")
print(f"✅ Val samples: {len(val_df)}")
