
"""
Support Ticket Intelligence Platform
Stage 7: Baseline Model Training

Models:
    1. Majority Classifier
    2. Multinomial Naive Bayes

Targets:
    1. Category
    2. Priority
    3. Assigned Team
"""

import json
import joblib
import pandas as pd

from pathlib import Path

from scipy.sparse import load_npz

from sklearn.naive_bayes import MultinomialNB
from sklearn.dummy import DummyClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)


# =====================================================
# 1. PROJECT PATHS
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "processed"
FEATURE_DIR = DATA_DIR / "features"

MODEL_DIR = PROJECT_ROOT / "artifacts" / "models"
REPORT_DIR = PROJECT_ROOT / "reports" / "model_evaluation"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================
# 2. LOAD DATA
# =====================================================

train_df = pd.read_csv(DATA_DIR / "train.csv")
test_df = pd.read_csv(DATA_DIR / "test.csv")

X_train = load_npz(
    FEATURE_DIR / "X_train_tfidf.npz"
)

X_test = load_npz(
    FEATURE_DIR / "X_test_tfidf.npz"
)

print("=" * 65)
print("SUPPORT TICKET CLASSIFICATION")
print("=" * 65)

print("\nTraining features:", X_train.shape)
print("Testing features:", X_test.shape)


# =====================================================
# 3. VALIDATE DATA
# =====================================================

if X_train.shape[0] != len(train_df):
    raise ValueError("Training feature and label counts differ.")

if X_test.shape[0] != len(test_df):
    raise ValueError("Testing feature and label counts differ.")

if X_train.shape[1] != X_test.shape[1]:
    raise ValueError("Training and testing feature counts differ.")


# =====================================================
# 4. DEFINE TARGETS
# =====================================================

TARGETS = [
    "category",
    "priority",
    "assigned_team"
]


# =====================================================
# 5. DEFINE EVALUATION FUNCTION
# =====================================================

def evaluate_model(model, X, y, model_name, target):

    predictions = model.predict(X)

    accuracy = accuracy_score(y, predictions)

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            y,
            predictions,
            average="macro",
            zero_division=0
        )
    )

    report = classification_report(
        y,
        predictions,
        output_dict=True,
        zero_division=0
    )

    labels = sorted(set(y))

    matrix = confusion_matrix(
        y,
        predictions,
        labels=labels
    )

    print(f"\n{model_name} - {target.upper()}")
    print("-" * 50)

    print("Accuracy:", round(accuracy, 4))
    print("Macro Precision:", round(precision, 4))
    print("Macro Recall:", round(recall, 4))
    print("Macro F1:", round(f1, 4))

    print("\nClassification Report:")

    print(
        classification_report(
            y,
            predictions,
            zero_division=0
        )
    )

    return {
        "model": model_name,
        "target": target,
        "accuracy": accuracy,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_labels": labels
    }


# =====================================================
# 6. TRAIN AND EVALUATE MODELS
# =====================================================

all_results = []

for target in TARGETS:

    print("\n" + "=" * 65)
    print(f"TRAINING TARGET: {target.upper()}")
    print("=" * 65)

    y_train = train_df[target]
    y_test = test_df[target]

    # -------------------------------------------------
    # A. MAJORITY CLASS BASELINE
    # -------------------------------------------------

    majority_model = DummyClassifier(
        strategy="most_frequent"
    )

    majority_model.fit(
        X_train,
        y_train
    )

    majority_results = evaluate_model(
        majority_model,
        X_test,
        y_test,
        "Majority Baseline",
        target
    )

    all_results.append(majority_results)

    # -------------------------------------------------
    # B. MULTINOMIAL NAIVE BAYES
    # -------------------------------------------------

    nb_model = MultinomialNB(
        alpha=1.0
    )

    nb_model.fit(
        X_train,
        y_train
    )

    nb_results = evaluate_model(
        nb_model,
        X_test,
        y_test,
        "Multinomial Naive Bayes",
        target
    )

    all_results.append(nb_results)

    # -------------------------------------------------
    # C. SAVE TRAINED MODEL
    # -------------------------------------------------

    model_path = MODEL_DIR / f"{target}_naive_bayes.joblib"

    joblib.dump(
        nb_model,
        model_path
    )

    print("\nModel saved:", model_path)


# =====================================================
# 7. SAVE EVALUATION REPORTS
# =====================================================

summary = []

for result in all_results:

    summary.append({
        "model": result["model"],
        "target": result["target"],
        "accuracy": result["accuracy"],
        "macro_precision": result["macro_precision"],
        "macro_recall": result["macro_recall"],
        "macro_f1": result["macro_f1"]
    })

summary_df = pd.DataFrame(summary)

summary_path = REPORT_DIR / "baseline_results.csv"

summary_df.to_csv(
    summary_path,
    index=False
)

# Save detailed reports and confusion matrices
detailed_results = []

for index, result in enumerate(all_results, start=1):

    detailed_results.append({
        "model": result["model"],
        "target": result["target"],
        "classification_report": result["classification_report"],
        "confusion_matrix": result["confusion_matrix"],
        "confusion_matrix_labels": result["confusion_matrix_labels"]
    })

with open(
    REPORT_DIR / "detailed_results.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        detailed_results,
        file,
        indent=4
    )


# =====================================================
# 8. FINAL SUMMARY
# =====================================================

print("\n" + "=" * 65)
print("BASELINE MODEL COMPARISON")
print("=" * 65)

print(
    summary_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\nResults saved to:", summary_path)

print("\nBASELINE TRAINING COMPLETED!")
