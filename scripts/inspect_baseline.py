
import json
import joblib
import pandas as pd

from pathlib import Path
from sklearn.metrics import classification_report


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "model_evaluation"
    / "detailed_results.json"
)

MODEL_DIR = PROJECT_ROOT / "artifacts" / "models"

VECTOR_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "tfidf_vectorizer.joblib"
)


# =====================================================
# 1. LOAD EVALUATION REPORTS
# =====================================================

with open(REPORT_PATH, "r", encoding="utf-8") as file:
    results = json.load(file)


# =====================================================
# 2. INSPECT CLASS-WISE PERFORMANCE
# =====================================================

for result in results:

    if result["model"] != "Multinomial Naive Bayes":
        continue

    print("\n" + "=" * 65)
    print("TARGET:", result["target"].upper())
    print("=" * 65)

    report = result["classification_report"]

    class_results = []

    for label, metrics in report.items():

        if label in ["accuracy", "macro avg", "weighted avg"]:
            continue

        class_results.append({
            "class": label,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1-score"],
            "support": metrics["support"]
        })

    class_df = pd.DataFrame(class_results)

    print(
        class_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    print("\nConfusion Matrix:")
    print(pd.DataFrame(
        result["confusion_matrix"],
        index=result["confusion_matrix_labels"],
        columns=result["confusion_matrix_labels"]
    ))


# =====================================================
# 3. INSPECT IMPORTANT WORDS PER CATEGORY
# =====================================================

vectorizer = joblib.load(VECTOR_PATH)

feature_names = vectorizer.get_feature_names_out()

for target in ["category", "priority", "assigned_team"]:

    model_path = MODEL_DIR / f"{target}_naive_bayes.joblib"

    model = joblib.load(model_path)

    print("\n" + "=" * 65)
    print("IMPORTANT WORDS:", target.upper())
    print("=" * 65)

    for class_index, class_name in enumerate(model.classes_):

        top_indices = model.feature_log_prob_[class_index].argsort()[-15:][::-1]

        top_words = feature_names[top_indices]

        print(f"\n{class_name}:")
        print(", ".join(top_words))

import numpy as np
from scipy.special import logsumexp

vectorizer = joblib.load(VECTOR_PATH)
feature_names = vectorizer.get_feature_names_out()

for target in ["category", "priority", "assigned_team"]:

    model = joblib.load(
        MODEL_DIR / f"{target}_naive_bayes.joblib"
    )

    print("\n" + "=" * 70)
    print(f"DISTINCTIVE WORDS: {target.upper()}")
    print("=" * 70)

    log_probs = model.feature_log_prob_

    for class_index, class_name in enumerate(model.classes_):

        other_indices = [
            i for i in range(len(model.classes_))
            if i != class_index
        ]

        other_log_probs = log_probs[other_indices]

        # Average log probability across all other classes
        other_average = (
            logsumexp(other_log_probs, axis=0)
            - np.log(len(other_indices))
        )

        # How much more characteristic is a word for this class?
        scores = log_probs[class_index] - other_average

        top_indices = np.argsort(scores)[-15:][::-1]

        print(f"\n{class_name}:")
        for index in top_indices:
            print(
                f"{feature_names[index]:25s} "
                f"Score: {scores[index]:.4f}"
            )