from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


# ==========================================
# 1. PROJECT PATH
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "data_file.csv"


# ==========================================
# 2. LOAD DATASET
# ==========================================

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

print("Original dataset shape:", df.shape)


# ==========================================
# 3. REMOVE IDENTIFIER COLUMNS
# ==========================================

identifier_columns = [
    "FileName",
    "md5Hash"
]

df = df.drop(columns=identifier_columns)

print("\nAfter removing identifier columns:")
print(df.shape)


# ==========================================
# 4. CHECK MISSING VALUES
# ==========================================

print("\nMissing values:")

print(df.isnull().sum().sum())


# ==========================================
# 5. REMOVE DUPLICATES
# ==========================================

duplicates = df.duplicated().sum()

print("\nDuplicate rows:", duplicates)

if duplicates > 0:

    df = df.drop_duplicates()

    print(
        "Shape after removing duplicates:",
        df.shape
    )


# ==========================================
# 6. SEPARATE FEATURES AND TARGET
# ==========================================

X = df.drop(columns=["Benign"])

y = df["Benign"]


print("\n========== FEATURES ==========")

for feature in X.columns:
    print(feature)


print("\nNumber of features:", X.shape[1])


# ==========================================
# 7. TARGET DISTRIBUTION
# ==========================================

print("\n========== TARGET DISTRIBUTION ==========")

print(y.value_counts())

print("\nTarget percentages:")

print(
    y.value_counts(normalize=True) * 100
)


# ==========================================
# 8. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


# ==========================================
# 9. DISPLAY SPLIT INFORMATION
# ==========================================

print("\n========== DATA SPLIT ==========")

print(
    "Training samples:",
    X_train.shape[0]
)

print(
    "Testing samples:",
    X_test.shape[0]
)

print(
    "Training features:",
    X_train.shape[1]
)

print(
    "Testing features:",
    X_test.shape[1]
)


# ==========================================
# 10. CLASS DISTRIBUTION AFTER SPLIT
# ==========================================
    
print("\nTraining target distribution:")

print(y_train.value_counts())


print("\nTesting target distribution:")

print(y_test.value_counts())


print("\nPreprocessing completed successfully.")