import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "support_tickets_text_processed.csv"
)

df = pd.read_csv(DATA_PATH)

# Metadata available for investigation
metadata_columns = [
    "product",
    "channel",
    "region",
    "language",
    "subscription_type",
    "customer_segment",
    "previous_tickets",
    "customer_age",
    "customer_tenure_months"
]

# Identify descriptions with conflicting priorities
priority_counts = (
    df.groupby("cleaned_description")["priority"]
    .nunique()
)

conflict_descriptions = priority_counts[
    priority_counts > 1
].index

conflict_df = df[
    df["cleaned_description"].isin(conflict_descriptions)
].copy()

print("TOTAL ROWS:", len(df))
print("CONFLICTING DESCRIPTION GROUPS:", len(conflict_descriptions))
print("ROWS IN CONFLICTING GROUPS:", len(conflict_df))

print("\nPRIORITY DISTRIBUTION IN CONFLICTING GROUPS")
print(conflict_df["priority"].value_counts())

print("\nMETADATA DIFFERENCES")

for column in metadata_columns:
    differences = (
        conflict_df.groupby("cleaned_description")[column]
        .nunique()
    )

    different_groups = (differences > 1).sum()

    print(
        f"{column}: "
        f"{different_groups} conflicting groups have different values"
    )

print("\nEXAMPLE CONFLICT GROUPS")

for description in list(conflict_descriptions[:5]):
    rows = conflict_df[
        conflict_df["cleaned_description"] == description
    ]

    print("\nDESCRIPTION:", description)

    print(
        rows[
            ["priority"] + metadata_columns
        ].to_string(index=False)
    )