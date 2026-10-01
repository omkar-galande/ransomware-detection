from pathlib import Path
import pandas as pd


# ==========================================
# 1. FIND PROJECT DIRECTORY
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "data_file.csv"


# ==========================================
# 2. CHECK DATASET EXISTS
# ==========================================

print("Dataset path:")
print(DATA_PATH)

if not DATA_PATH.exists():
    print("\nERROR: Dataset file not found!")
    print("Please check the filename and location.")
    exit()

print("\nDataset found successfully!")


# ==========================================
# 3. LOAD DATASET
# ==========================================

df = pd.read_csv(DATA_PATH)


# ==========================================
# 4. DATASET SHAPE
# ==========================================

print("\n========== DATASET SHAPE ==========")
print(df.shape)


# ==========================================
# 5. COLUMN NAMES
# ==========================================

print("\n========== COLUMN NAMES ==========")

for column in df.columns:
    print(column)


# ==========================================
# 6. FIRST 5 ROWS
# ==========================================

print("\n========== FIRST 5 ROWS ==========")
print(df.head())


# ==========================================
# 7. DATA TYPES
# ==========================================

print("\n========== DATA TYPES ==========")
print(df.dtypes)


# ==========================================
# 8. DATASET INFORMATION
# ==========================================

print("\n========== DATASET INFO ==========")
df.info()


# ==========================================
# 9. MISSING VALUES
# ==========================================

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())


# ==========================================
# 10. DUPLICATES
# ==========================================

print("\n========== DUPLICATE ROWS ==========")
print(df.duplicated().sum())


# ==========================================
# 11. STATISTICAL SUMMARY
# ==========================================

print("\n========== STATISTICAL SUMMARY ==========")
print(df.describe())


# ==========================================
# 12. TARGET COLUMN
# ==========================================

if "Benign" in df.columns:

    print("\n========== TARGET DISTRIBUTION ==========")
    print(df["Benign"].value_counts())

    print("\n========== TARGET DISTRIBUTION (%) ==========")
    print(
        df["Benign"]
        .value_counts(normalize=True)
        .mul(100)
    )

else:

    print("\nWARNING: 'Benign' column was not found.")