
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score


# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

DATA_PATH = Path("data/raw/support_tickets_sample.csv")
REPORT_DIR = Path("reports")

REPORT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)

TARGETS = ["category", "priority", "assigned_team"]

print("=" * 70)
print("SUPPORT TICKET INTELLIGENCE - COMPLETE FEATURE ANALYSIS")
print("=" * 70)

print(f"\nDataset shape: {df.shape}")
print(f"Total rows: {len(df)}")
print(f"Total columns: {len(df.columns)}")


# --------------------------------------------------
# 2. DEFINE FEATURE GROUPS
# --------------------------------------------------

# Potentially available when a ticket is created
APPROVED_FEATURES = [
    "issue_description",
    "product",
    "channel",
    "region",
    "language",
    "customer_age",
    "subscription_type",
    "customer_tenure_months",
    "previous_tickets",
    "customer_segment",
    "created_at",
]

# Need investigation before using
INVESTIGATE_FEATURES = [
    "issue_complexity_score",
    "issue_type_id",
    "reopen_count",
    "sla_target_hours",
]

# Exclude from initial prediction models
EXCLUDED_FEATURES = [
    "ticket_id",
    "customer_name",
    "customer_email",
    "customer_gender",
    "assigned_to",
    "status",
    "escalated",
    "escalation_reason",
    "first_response_time_hours",
    "resolution_time_hours",
    "paused_hours",
    "resolution_wallclock_hours",
    "first_resolved_at",
    "resolved_at",
    "closed_at",
    "resolved_by",
    "resolution_notes",
    "sla_breached",
    "sla_breach_margin",
    "customer_satisfaction_score",
]


# --------------------------------------------------
# 3. BASIC FEATURE INFORMATION
# --------------------------------------------------

feature_info = []

for col in df.columns:

    if col in TARGETS:
        continue

    missing_count = df[col].isna().sum()
    missing_pct = (missing_count / len(df)) * 100

    unique_count = df[col].nunique(dropna=True)
    unique_pct = (unique_count / len(df)) * 100

    if col in APPROVED_FEATURES:
        status = "Potentially usable"
        availability_score = 1.0
        leakage_score = 1.0

    elif col in INVESTIGATE_FEATURES:
        status = "Needs investigation"
        availability_score = 0.5
        leakage_score = 0.5

    else:
        status = "Exclude initially"
        availability_score = 0.0
        leakage_score = 0.0

    feature_info.append({
        "feature": col,
        "data_type": str(df[col].dtype),
        "missing_count": missing_count,
        "missing_percentage": round(missing_pct, 2),
        "unique_values": unique_count,
        "unique_percentage": round(unique_pct, 2),
        "selection_status": status,
        "availability_factor": availability_score,
        "leakage_safety_factor": leakage_score,
    })

feature_df = pd.DataFrame(feature_info)


# --------------------------------------------------
# 4. FEATURE ASSOCIATION WITH TARGETS
# --------------------------------------------------

def calculate_nmi(feature, target):

    temp = pd.DataFrame({
        "feature": feature,
        "target": target
    }).dropna()

    if len(temp) < 2:
        return np.nan

    # Convert numerical features into quantile groups
    if pd.api.types.is_numeric_dtype(temp["feature"]):
        try:
            temp["feature"] = pd.qcut(
                temp["feature"],
                q=10,
                duplicates="drop"
            )
        except ValueError:
            temp["feature"] = temp["feature"].astype(str)

    # Convert values to strings for consistent comparison
    feature_values = temp["feature"].astype(str)
    target_values = temp["target"].astype(str)

    return normalized_mutual_info_score(
        feature_values,
        target_values
    )


association_results = []

for feature in df.columns:

    if feature in TARGETS:
        continue

    for target in TARGETS:

        # Exact descriptions have very high cardinality.
        # Use word count and character count as simple proxies.
        if feature == "issue_description":

            text = df[feature].fillna("").astype(str)

            proxy_features = {
                "description_word_count": text.str.split().str.len(),
                "description_character_count": text.str.len(),
            }

            for proxy_name, proxy_values in proxy_features.items():

                score = calculate_nmi(
                    proxy_values,
                    df[target]
                )

                association_results.append({
                    "feature": proxy_name,
                    "original_feature": feature,
                    "target": target,
                    "nmi_score": round(score, 4),
                    "note": "Text-length proxy only; not full text relevance"
                })

        else:

            score = calculate_nmi(
                df[feature],
                df[target]
            )

            association_results.append({
                "feature": feature,
                "original_feature": feature,
                "target": target,
                "nmi_score": round(score, 4),
                "note": "Exploratory association, not proof of causation"
            })


association_df = pd.DataFrame(association_results)


# --------------------------------------------------
# 5. CALCULATE FEATURE SCORES
# --------------------------------------------------

# Maximum association across the three prediction targets.
# This is a screening score, not a model-performance score.

max_association = (
    association_df
    .groupby("original_feature")["nmi_score"]
    .max()
    .to_dict()
)

feature_df["relevance_nmi"] = (
    feature_df["feature"]
    .map(max_association)
    .fillna(0)
)

# Score components:
# Relevance: 40 points
# Availability at prediction time: 30 points
# Leakage safety: 30 points

feature_df["relevance_points"] = (
    feature_df["relevance_nmi"] * 40
)

feature_df["availability_points"] = (
    feature_df["availability_factor"] * 30
)

feature_df["leakage_safety_points"] = (
    feature_df["leakage_safety_factor"] * 30
)

feature_df["feature_score"] = (
    feature_df["relevance_points"]
    + feature_df["availability_points"]
    + feature_df["leakage_safety_points"]
).round(2)

feature_df = feature_df.sort_values(
    "feature_score",
    ascending=False
)


# --------------------------------------------------
# 6. TARGET DISTRIBUTION
# --------------------------------------------------

print("\n\nTARGET DISTRIBUTIONS")

for target in TARGETS:

    print("\n" + "-" * 50)
    print(f"TARGET: {target.upper()}")

    print(
        df[target]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\nPercentage distribution:")

    print(
        (df[target].value_counts(normalize=True) * 100)
        .round(2)
        .to_string()
    )


# --------------------------------------------------
# 7. MISSING VALUE REPORT
# --------------------------------------------------

missing_report = (
    df.isnull()
    .sum()
    .to_frame("missing_count")
)

missing_report["missing_percentage"] = (
    missing_report["missing_count"] / len(df) * 100
).round(2)

missing_report = missing_report.sort_values(
    "missing_percentage",
    ascending=False
)

print("\n\nMISSING VALUE ANALYSIS")
print(missing_report.to_string())


# --------------------------------------------------
# 8. CATEGORICAL FEATURE ANALYSIS
# --------------------------------------------------

categorical_columns = df.select_dtypes(
    include=["object", "str"]
).columns.tolist()

print("\n\nCATEGORICAL FEATURE ANALYSIS")

for col in categorical_columns:

    if col in TARGETS:
        continue

    print("\n" + "-" * 50)
    print(f"FEATURE: {col}")
    print(f"Unique values: {df[col].nunique()}")

    print(
        df[col]
        .value_counts(dropna=False)
        .head(10)
        .to_string()
    )


# --------------------------------------------------
# 9. NUMERICAL FEATURE ANALYSIS
# --------------------------------------------------

numerical_columns = df.select_dtypes(
    include=np.number
).columns.tolist()

print("\n\nNUMERICAL FEATURE ANALYSIS")

print(
    df[numerical_columns]
    .describe()
    .round(2)
    .to_string()
)


# --------------------------------------------------
# 10. DESCRIPTION DUPLICATE ANALYSIS
# --------------------------------------------------

if "issue_description" in df.columns:

    print("\n\nDESCRIPTION DUPLICATE ANALYSIS")

    print(
        "Total descriptions:",
        len(df)
    )

    print(
        "Unique descriptions:",
        df["issue_description"].nunique()
    )

    print(
        "Duplicate descriptions:",
        df["issue_description"].duplicated().sum()
    )

    for target in TARGETS:

        mapping = (
            df.groupby("issue_description")[target]
            .nunique()
        )

        print(
            f"\nDescriptions mapping to multiple {target} labels:",
            (mapping > 1).sum()
        )


# --------------------------------------------------
# 11. ISSUE TYPE ID LEAKAGE CHECK
# --------------------------------------------------

if "issue_type_id" in df.columns:

    print("\n\nISSUE_TYPE_ID LEAKAGE CHECK")

    for target in TARGETS:

        mapping = (
            df.groupby("issue_type_id")[target]
            .nunique()
        )

        print(
            f"Unique issue_type_id values mapping to multiple "
            f"{target} labels: {(mapping > 1).sum()}"
        )


# --------------------------------------------------
# 12. SAVE REPORTS
# --------------------------------------------------

feature_df.to_csv(
    REPORT_DIR / "feature_scores.csv",
    index=False
)

association_df.to_csv(
    REPORT_DIR / "feature_target_associations.csv",
    index=False
)

missing_report.to_csv(
    REPORT_DIR / "missing_value_report.csv"
)


# --------------------------------------------------
# 13. FINAL FEATURE SCORE TABLE
# --------------------------------------------------

print("\n\n" + "=" * 70)
print("FINAL FEATURE ANALYSIS SCORE")
print("=" * 70)

print(
    feature_df[
        [
            "feature",
            "selection_status",
            "relevance_nmi",
            "relevance_points",
            "availability_points",
            "leakage_safety_points",
            "feature_score",
        ]
    ].to_string(index=False)
)


# --------------------------------------------------
# 14. FINAL RECOMMENDATIONS
# --------------------------------------------------

print("\n\nRECOMMENDED FEATURE GROUPS")

print("\nPotentially usable:")
print(APPROVED_FEATURES)

print("\nNeeds investigation:")
print(INVESTIGATE_FEATURES)

print("\nExclude initially:")
print(EXCLUDED_FEATURES)


print("\n\nREPORTS SAVED SUCCESSFULLY")
print("1. reports/feature_scores.csv")
print("2. reports/feature_target_associations.csv")
print("3. reports/missing_value_report.csv")

print("\nFEATURE ANALYSIS COMPLETED!")




import pandas as pd

df = pd.read_csv("data/raw/support_tickets_sample.csv")

print("\nISSUE TYPE ID VS CATEGORY")

print(pd.crosstab(
    df["issue_type_id"],
    df["category"]
).head(20))

print("\nSLA TARGET HOURS VS PRIORITY")

print(pd.crosstab(
    df["sla_target_hours"],
    df["priority"],
    normalize="index"
).round(3))

print("\nISSUE COMPLEXITY SCORE VS PRIORITY")

print(pd.crosstab(
    df["issue_complexity_score"],
    df["priority"],
    normalize="index"
).round(3))

print("\nCREATED AT DATE RANGE")

dates = pd.to_datetime(df["created_at"], errors="coerce")

print("Invalid dates:", dates.isna().sum())
print("Date range:", dates.min(), "to", dates.max())

print("\nCREATED AT MONTH VS CATEGORY")

print(pd.crosstab(
    dates.dt.month,
    df["category"],
    normalize="index"
).round(2))




import pandas as pd

df = pd.read_csv("data/raw/support_tickets_sample.csv")

dates = pd.to_datetime(df["created_at"])

df["created_hour"] = dates.dt.hour
df["created_day"] = dates.dt.dayofweek
df["created_year"] = dates.dt.year

print("\nCOMPLEXITY VS ASSIGNED TEAM")

print(pd.crosstab(
    df["issue_complexity_score"],
    df["assigned_team"],
    normalize="index"
).round(3))

print("\nCREATED YEAR VS CATEGORY")

print(pd.crosstab(
    df["created_year"],
    df["category"],
    normalize="index"
).round(3))

print("\nCREATED DAY VS PRIORITY")

print(pd.crosstab(
    df["created_day"],
    df["priority"],
    normalize="index"
).round(3))

print("\nCREATED HOUR VS PRIORITY")

print(pd.crosstab(
    df["created_hour"],
    df["priority"],
    normalize="index"
).round(3))
