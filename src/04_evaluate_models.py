from pathlib import Path
import pandas as pd
import joblib
import time

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ==========================================
# PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "test_data.csv"
MODEL_PATH = PROJECT_ROOT / "models"


# ==========================================
# LOAD TEST DATA
# ==========================================

print("Loading test data...")

test_data = pd.read_csv(DATA_PATH)

X_test = test_data.drop(columns=["Benign"])
y_test = test_data["Benign"]

print("Test samples:", X_test.shape[0])
print("Test features:", X_test.shape[1])


# ==========================================
# LOAD MODELS
# ==========================================

print("\nLoading models...")

decision_tree = joblib.load(
    MODEL_PATH / "decision_tree.pkl"
)

random_forest = joblib.load(
    MODEL_PATH / "random_forest.pkl"
)

print("Decision Tree loaded.")
print("Random Forest loaded.")


# ==========================================
# FUNCTION TO EVALUATE MODEL
# ==========================================

def evaluate_model(model, model_name):

    print("\n")
    print("=" * 60)
    print(model_name)
    print("=" * 60)

    # --------------------------------------
    # Prediction
    # --------------------------------------

    start_time = time.time()

    y_pred = model.predict(X_test)

    prediction_time = time.time() - start_time

    # --------------------------------------
    # Metrics
    # --------------------------------------

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        pos_label=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        pos_label=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        pos_label=0
    )

    # --------------------------------------
    # Confusion Matrix
    # --------------------------------------

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    # --------------------------------------
    # False Positive Rate
    # --------------------------------------

    if (fp + tn) > 0:
        false_positive_rate = fp / (fp + tn)
    else:
        false_positive_rate = 0

    # --------------------------------------
    # Results
    # --------------------------------------

    print("\nConfusion Matrix:")
    print(cm)

    print("\nTN:", tn)
    print("FP:", fp)
    print("FN:", fn)
    print("TP:", tp)

    print("\nAccuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1-score:", round(f1, 4))
    print("False Positive Rate:", round(false_positive_rate, 4))

    print(
        "Prediction time:",
        round(prediction_time, 6),
        "seconds"
    )

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred
        )
    )

    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1,
        "False_Positive_Rate": false_positive_rate,
        "Prediction_Time": prediction_time
    }


# ==========================================
# EVALUATE BOTH MODELS
# ==========================================

decision_tree_results = evaluate_model(
    decision_tree,
    "Decision Tree"
)

random_forest_results = evaluate_model(
    random_forest,
    "Random Forest"
)


# ==========================================
# COMPARISON TABLE
# ==========================================

results = pd.DataFrame([
    decision_tree_results,
    random_forest_results
])

print("\n")
print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(results.to_string(index=False))


# ==========================================
# SAVE RESULTS
# ==========================================

RESULTS_PATH = PROJECT_ROOT / "results"

RESULTS_PATH.mkdir(exist_ok=True)

results_file = RESULTS_PATH / "model_comparison.csv"

results.to_csv(
    results_file,
    index=False
)

print("\nResults saved to:")
print(results_file)

print("\n========================================")
print("MODEL EVALUATION COMPLETED")
print("========================================")