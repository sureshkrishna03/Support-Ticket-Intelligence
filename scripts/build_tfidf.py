
import pandas as pd
import joblib

from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz


# =====================================================
# 1. PROJECT PATHS
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"

ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
FEATURE_DIR = PROJECT_ROOT / "data" / "processed" / "features"

ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
FEATURE_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================
# 2. LOAD TRAINING AND TEST DATA
# =====================================================

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

print("=" * 60)
print("TF-IDF FEATURE EXTRACTION")
print("=" * 60)

print("\nTraining shape:", train_df.shape)
print("Testing shape:", test_df.shape)


# =====================================================
# 3. VALIDATE TEXT DATA
# =====================================================

TEXT_COLUMN = "cleaned_description"

for name, dataset in [
    ("Training", train_df),
    ("Testing", test_df)
]:

    if TEXT_COLUMN not in dataset.columns:
        raise ValueError(
            f"{TEXT_COLUMN} missing in {name} dataset"
        )

    if dataset[TEXT_COLUMN].isna().any():
        raise ValueError(
            f"Missing text found in {name} dataset"
        )

    if dataset[TEXT_COLUMN].str.strip().eq("").any():
        raise ValueError(
            f"Empty text found in {name} dataset"
        )


# =====================================================
# 4. INITIALIZE TF-IDF VECTORIZER
# =====================================================

vectorizer = TfidfVectorizer(
    lowercase=False,
    max_features=10000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)


# =====================================================
# 5. FIT ON TRAINING DATA ONLY
# =====================================================

X_train = vectorizer.fit_transform(
    train_df[TEXT_COLUMN]
)

print("\nTF-IDF fitted on training data.")


# =====================================================
# 6. TRANSFORM TEST DATA
# =====================================================

X_test = vectorizer.transform(
    test_df[TEXT_COLUMN]
)

print("Test data transformed successfully.")


# =====================================================
# 7. CHECK FEATURE DIMENSIONS
# =====================================================

print("\nTraining feature matrix:", X_train.shape)
print("Testing feature matrix:", X_test.shape)

print(
    "\nNumber of learned vocabulary terms:",
    len(vectorizer.vocabulary_)
)


# =====================================================
# 8. INSPECT SAMPLE VOCABULARY
# =====================================================

feature_names = vectorizer.get_feature_names_out()

print("\nSample learned features:")

for feature in feature_names[:30]:
    print(feature)


# =====================================================
# 9. SAVE TF-IDF ARTIFACTS
# =====================================================

joblib.dump(
    vectorizer,
    ARTIFACT_DIR / "tfidf_vectorizer.joblib"
)

save_npz(
    FEATURE_DIR / "X_train_tfidf.npz",
    X_train
)

save_npz(
    FEATURE_DIR / "X_test_tfidf.npz",
    X_test
)


# =====================================================
# 10. FINAL VALIDATION
# =====================================================

assert X_train.shape[1] == X_test.shape[1]

assert X_train.shape[0] == len(train_df)

assert X_test.shape[0] == len(test_df)

print("\n" + "=" * 60)
print("TF-IDF FEATURE EXTRACTION COMPLETED")
print("=" * 60)

print("\nSaved files:")
print("1. artifacts/tfidf_vectorizer.joblib")
print("2. data/processed/features/X_train_tfidf.npz")
print("3. data/processed/features/X_test_tfidf.npz")
