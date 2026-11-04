
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# 0. LOAD DATA
# =========================================================

DATA_PATH = Path("data/raw/support_tickets_sample.csv")
REPORT_DIR = Path("reports/eda")

REPORT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("SUPPORT TICKET INTELLIGENCE - EXPLORATORY DATA ANALYSIS")
print("=" * 60)


# =========================================================
# 1. DATASET OVERVIEW
# =========================================================

print("\n1. DATASET OVERVIEW")

print("Shape:", df.shape)
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nDataset information:")
df.info()

print("\nMemory usage (MB):")
print(round(df.memory_usage(deep=True).sum() / (1024 ** 2), 2))


# =========================================================
# 2. DATA QUALITY - DUPLICATES
# =========================================================

print("\n2. DUPLICATE ANALYSIS")

print("Duplicate complete rows:", df.duplicated().sum())
print("Duplicate ticket IDs:", df["ticket_id"].duplicated().sum())
print("Duplicate descriptions:", df["issue_description"].duplicated().sum())

# Save duplicate descriptions for investigation
duplicate_descriptions = df[
    df["issue_description"].duplicated(keep=False)
].sort_values("issue_description")

duplicate_descriptions.to_csv(
    REPORT_DIR / "duplicate_descriptions.csv",
    index=False
)


# =========================================================
# 3. MISSING VALUE ANALYSIS
# =========================================================

print("\n3. MISSING VALUE ANALYSIS")

missing_count = df.isnull().sum()

missing_report = pd.DataFrame({
    "missing_count": missing_count,
    "missing_percentage": (missing_count / len(df) * 100).round(2)
})

missing_report = missing_report.sort_values(
    "missing_percentage",
    ascending=False
)

print(missing_report)

missing_report.to_csv(REPORT_DIR / "missing_values.csv")


# =========================================================
# 4. DESCRIPTIVE STATISTICS
# =========================================================

print("\n4. DESCRIPTIVE STATISTICS")

print("\nNumerical columns:")
print(df.describe().T)

print("\nCategorical columns:")
print(df.describe(include="object").T)

df.describe().T.to_csv(REPORT_DIR / "numerical_statistics.csv")
df.describe(include="object").T.to_csv(
    REPORT_DIR / "categorical_statistics.csv"
)


# =========================================================
# 5. NUMERICAL FEATURE ANALYSIS
# =========================================================

print("\n5. NUMERICAL FEATURE ANALYSIS")

numeric_columns = df.select_dtypes(include="number").columns

for column in numeric_columns:
    print(f"\n{column}")
    print("Minimum:", df[column].min())
    print("Maximum:", df[column].max())
    print("Mean:", round(df[column].mean(), 2))
    print("Median:", round(df[column].median(), 2))
    print("Standard deviation:", round(df[column].std(), 2))

# Histograms
df[numeric_columns].hist(
    figsize=(16, 12),
    bins=20
)

plt.suptitle("Numerical Feature Distributions")
plt.tight_layout()
plt.savefig(REPORT_DIR / "numerical_distributions.png")
plt.close()


# =========================================================
# 6. CATEGORICAL FEATURE ANALYSIS
# =========================================================

print("\n6. CATEGORICAL FEATURE ANALYSIS")

categorical_columns = df.select_dtypes(
    include=["object", "str"]
).columns

for column in categorical_columns:
    print(f"\n--- {column} ---")
    print("Unique values:", df[column].nunique())
    print(df[column].value_counts(dropna=False).head(15))


# =========================================================
# 7. TARGET VARIABLE ANALYSIS
# =========================================================

print("\n7. TARGET VARIABLE ANALYSIS")

target_columns = [
    "category",
    "priority",
    "assigned_team"
]

for column in target_columns:

    counts = df[column].value_counts()

    percentages = (
        df[column]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    print(f"\n--- {column.upper()} ---")
    print(pd.DataFrame({
        "count": counts,
        "percentage": percentages
    }))

    counts.plot(
        kind="bar",
        figsize=(9, 5),
        title=f"{column} Distribution"
    )

    plt.ylabel("Number of tickets")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / f"{column}_distribution.png")
    plt.close()


# =========================================================
# 8. TEXT FEATURE ANALYSIS
# =========================================================

print("\n8. TEXT FEATURE ANALYSIS")

df["description_length"] = (
    df["issue_description"].fillna("").str.len()
)

df["word_count"] = (
    df["issue_description"].fillna("").str.split().str.len()
)

print("\nDescription length statistics:")
print(df["description_length"].describe())

print("\nWord count statistics:")
print(df["word_count"].describe())

print("\nShortest descriptions:")
print(
    df.nsmallest(5, "description_length")[
        ["issue_description", "description_length"]
    ]
)

print("\nLongest descriptions:")
print(
    df.nlargest(5, "description_length")[
        ["issue_description", "description_length"]
    ]
)

df[["description_length", "word_count"]].hist(
    figsize=(10, 4),
    bins=20
)

plt.suptitle("Ticket Text Length Distributions")
plt.tight_layout()
plt.savefig(REPORT_DIR / "text_distributions.png")
plt.close()


# =========================================================
# 9. DESCRIPTION AND LABEL CONSISTENCY
# =========================================================

print("\n9. DESCRIPTION AND LABEL CONSISTENCY")

for target in target_columns:

    consistency = (
        df.groupby("issue_description")[target]
        .nunique()
    )

    print(f"\nTarget: {target}")
    print("Descriptions with one label:", (consistency == 1).sum())
    print("Descriptions with multiple labels:", (consistency > 1).sum())

    conflicts = consistency[consistency > 1]

    conflicts.to_csv(
        REPORT_DIR / f"{target}_description_conflicts.csv"
    )


# =========================================================
# 10. FEATURE RELATIONSHIPS
# =========================================================

print("\n10. FEATURE RELATIONSHIPS")

# Check issue_type_id to category relationship
issue_type_mapping = (
    df.groupby("issue_type_id")["category"]
    .nunique()
)

print("\nIssue types mapped to one category:",
      (issue_type_mapping == 1).sum())

print("Issue types mapped to multiple categories:",
      (issue_type_mapping > 1).sum())

print("\nIssue type vs category:")
print(pd.crosstab(df["issue_type_id"], df["category"]))

# Numeric correlations are exploratory, not proof of causation
correlation = df.select_dtypes(include="number").corr()

correlation.to_csv(REPORT_DIR / "numeric_correlations.csv")

plt.figure(figsize=(12, 9))
plt.imshow(correlation, aspect="auto")
plt.colorbar(label="Correlation")
plt.xticks(
    range(len(correlation.columns)),
    correlation.columns,
    rotation=90
)
plt.yticks(
    range(len(correlation.columns)),
    correlation.columns
)
plt.title("Numerical Feature Correlation")
plt.tight_layout()
plt.savefig(REPORT_DIR / "correlation_matrix.png")
plt.close()


# =========================================================
# 11. DATE ANALYSIS
# =========================================================

print("\n11. DATE ANALYSIS")

date_columns = [
    "created_at",
    "first_resolved_at",
    "resolved_at",
    "closed_at"
]

for column in date_columns:

    dates = pd.to_datetime(df[column], errors="coerce")

    print(f"\n{column}")
    print("Valid dates:", dates.notna().sum())
    print("Invalid or missing dates:", dates.isna().sum())

    if dates.notna().any():
        print("Earliest:", dates.min())
        print("Latest:", dates.max())


# =========================================================
# 12. POTENTIAL DATA LEAKAGE INVESTIGATION
# =========================================================

print("\n12. POTENTIAL DATA LEAKAGE INVESTIGATION")

potential_leakage_columns = [
    "category",
    "priority",
    "assigned_team",
    "assigned_to",
    "resolved_at",
    "closed_at",
    "resolved_by",
    "resolution_notes",
    "resolution_time_hours",
    "customer_satisfaction_score",
    "sla_breached"
]

print("\nReview these columns before selecting model features:")

for column in potential_leakage_columns:
    if column in df.columns:
        print(f"{column}: {df[column].nunique()} unique values")


# =========================================================
# 13. FINAL SUMMARY
# =========================================================

print("\n" + "=" * 60)
print("EDA COMPLETED")
print("=" * 60)

print("Total records:", len(df))
print("Total columns:", len(df.columns))
print("Duplicate rows:", df.duplicated().sum())
print("Missing cells:", df.isnull().sum().sum())

print("\nReports saved in:", REPORT_DIR)
