import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Add src folder to Python's module search path
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))

import pandas as pd

from features.text_processing import process_dataframe

# =====================================================
# 1. LOAD CLEAN DATASET
# =====================================================

input_path = PROJECT_ROOT / "data" / "processed" / "support_tickets_clean.csv"

df = pd.read_csv(input_path)

print("Original dataset shape:", df.shape)


# =====================================================
# 2. APPLY TEXT PREPROCESSING
# =====================================================

processed_df = process_dataframe(
    df=df,
    text_column="issue_description",
    output_column="cleaned_description",
    remove_stop_words=False,
    stemming=False,
    lemmatization=False
)


# =====================================================
# 3. VALIDATE PROCESSED TEXT
# =====================================================

print("\nProcessed dataset shape:", processed_df.shape)

print("\nOriginal vs Cleaned Text:")

for index in range(5):
    print("\nOriginal:")
    print(processed_df["issue_description"].iloc[index])

    print("\nCleaned:")
    print(processed_df["cleaned_description"].iloc[index])

    print("-" * 60)


# =====================================================
# 4. CHECK EMPTY DESCRIPTIONS
# =====================================================

empty_descriptions = (
    processed_df["cleaned_description"].str.strip() == ""
).sum()

print("\nEmpty cleaned descriptions:", empty_descriptions)


# =====================================================
# 5. CHECK DUPLICATE DESCRIPTIONS
# =====================================================

duplicate_descriptions = (
    processed_df["cleaned_description"].duplicated().sum()
)

print("Duplicate cleaned descriptions:", duplicate_descriptions)


# =====================================================
# 6. CHECK TEXT LENGTH
# =====================================================

processed_df["original_length"] = (
    processed_df["issue_description"].str.len()
)

processed_df["cleaned_length"] = (
    processed_df["cleaned_description"].str.len()
)

print("\nAverage original text length:")
print(processed_df["original_length"].mean())

print("\nAverage cleaned text length:")
print(processed_df["cleaned_length"].mean())


# =====================================================
# 7. SAVE PROCESSED DATASET
# =====================================================

output_path = PROJECT_ROOT / "data" / "processed" / "support_tickets_text_processed.csv"

processed_df.to_csv(output_path, index=False)

print("\nProcessed dataset saved successfully!")
print("Output:", output_path)
