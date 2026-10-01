from pathlib import Path
import pandas as pd
import joblib
import time

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# ==========================================
# PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "data_file.csv"
MODEL_PATH = PROJECT_ROOT / "models"

MODEL_PATH.mkdir(exist_ok=True)


# ==========================================
# LOAD DATASET
# ==========================================

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

print("Original dataset shape:", df.shape)


# ==========================================
# REMOVE IDENTIFIER COLUMNS
# ==========================================

identifier_columns = ["FileName", "md5Hash"]

df = df.drop(columns=identifier_columns)

print("After removing identifier columns:", df.shape)


# ==========================================
# REMOVE DUPLICATES
# ==========================================

duplicates = df.duplicated().sum()

print("Duplicate rows found:", duplicates)

if duplicates > 0:
    df = df.drop_duplicates()

print("Final dataset shape:", df.shape)


# ==========================================
# SEPARATE FEATURES AND TARGET
# ==========================================

X = df.drop(columns=["Benign"])
y = df["Benign"]

print("\nNumber of features:", X.shape[1])
print("Number of samples:", X.shape[0])


# ==========================================
# TRAIN-TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n========== DATA SPLIT ==========")

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# ==========================================
# DECISION TREE
# ==========================================

print("\n========== DECISION TREE ==========")

start_time = time.time()

decision_tree = DecisionTreeClassifier(
    random_state=42
)

decision_tree.fit(X_train, y_train)

dt_training_time = time.time() - start_time

print("Decision Tree training completed.")
print("Training time:", round(dt_training_time, 4), "seconds")


# ==========================================
# SAVE DECISION TREE
# ==========================================

dt_model_path = MODEL_PATH / "decision_tree.pkl"

joblib.dump(decision_tree, dt_model_path)

print("Decision Tree saved to:")
print(dt_model_path)


# ==========================================
# RANDOM FOREST
# ==========================================

print("\n========== RANDOM FOREST ==========")

start_time = time.time()

random_forest = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

random_forest.fit(X_train, y_train)

rf_training_time = time.time() - start_time

print("Random Forest training completed.")
print("Training time:", round(rf_training_time, 4), "seconds")


# ==========================================
# SAVE RANDOM FOREST
# ==========================================

rf_model_path = MODEL_PATH / "random_forest.pkl"

joblib.dump(random_forest, rf_model_path)

print("Random Forest saved to:")
print(rf_model_path)


# ==========================================
# SAVE TEST DATA
# ==========================================

test_data = X_test.copy()
test_data["Benign"] = y_test.values

test_data_path = PROJECT_ROOT / "data" / "test_data.csv"

test_data.to_csv(test_data_path, index=False)

print("\nTest data saved to:")
print(test_data_path)


# ==========================================
# COMPLETION
# ==========================================

print("\n========================================")
print("MODEL TRAINING COMPLETED SUCCESSFULLY")
print("========================================")