
import pandas as pd

from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit


# =====================================================
# 1. PROJECT PATHS
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "support_tickets_text_processed.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"


# =====================================================
# 2. LOAD PROCESSED DATASET
# =====================================================

df = pd.read_csv(INPUT_PATH)

print("=" * 60)
print("TRAIN / TEST SPLITTING")
print("=" * 60)

print("\nDataset shape:", df.shape)


# =====================================================
# 3. VALIDATE REQUIRED COLUMNS
# =====================================================

required_columns = [
    "cleaned_description",
    "category",
    "priority",
    "assigned_team"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

if df["cleaned_description"].isna().any():
    raise ValueError("Missing cleaned descriptions found.")

if df["cleaned_description"].str.strip().eq("").any():
    raise ValueError("Empty cleaned descriptions found.")


# =====================================================
# 4. CREATE GROUP IDENTIFIERS
# =====================================================

# Identical descriptions receive the same group ID.
groups = pd.factorize(
    df["cleaned_description"]
)[0]

print("\nUnique description groups:", len(set(groups)))


# =====================================================
# 5. GROUP-BASED TRAIN / TEST SPLIT
# =====================================================

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_indices, test_indices = next(
    splitter.split(
        df,
        groups=groups
    )
)

train_df = df.iloc[train_indices].copy()
test_df = df.iloc[test_indices].copy()


# =====================================================
# 6. CHECK DATASET SIZES
# =====================================================

print("\nTraining dataset:", train_df.shape)
print("Testing dataset:", test_df.shape)

print(
    "\nTraining percentage:",
    round(len(train_df) / len(df) * 100, 2),
    "%"
)

print(
    "Testing percentage:",
    round(len(test_df) / len(df) * 100, 2),
    "%"
)


# =====================================================
# 7. VERIFY NO GROUP LEAKAGE
# =====================================================

train_groups = set(groups[train_indices])
test_groups = set(groups[test_indices])

overlap = train_groups.intersection(test_groups)

print("\nOverlapping description groups:", len(overlap))

assert len(overlap) == 0, (
    "Data leakage detected!"
)


# =====================================================
# 8. CHECK TARGET DISTRIBUTIONS
# =====================================================

targets = [
    "category",
    "priority",
    "assigned_team"
]

for target in targets:

    print(f"\n{target.upper()} DISTRIBUTION")

    print("\nTraining:")
    print(
        train_df[target].value_counts(normalize=True).round(3)
    )

    print("\nTesting:")
    print(
        test_df[target].value_counts(normalize=True).round(3)
    )


# =====================================================
# 9. SAVE TRAIN AND TEST DATASETS
# =====================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

train_df.to_csv(
    OUTPUT_DIR / "train.csv",
    index=False
)

test_df.to_csv(
    OUTPUT_DIR / "test.csv",
    index=False
)

print("\nTrain and test datasets saved successfully!")

print("Training file:", OUTPUT_DIR / "train.csv")
print("Testing file:", OUTPUT_DIR / "test.csv")
