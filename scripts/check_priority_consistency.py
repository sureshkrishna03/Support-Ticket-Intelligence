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

# Check identical cleaned descriptions
priority_counts = (
    df.groupby("cleaned_description")["priority"]
    .nunique()
)

conflicts = priority_counts[priority_counts > 1]

print("TOTAL UNIQUE DESCRIPTIONS:", len(priority_counts))
print("DESCRIPTIONS WITH CONFLICTING PRIORITIES:", len(conflicts))

print("\nEXAMPLES OF CONFLICTING PRIORITY LABELS")

for description in conflicts.index[:10]:
    rows = df[df["cleaned_description"] == description]

    print("\nDescription:", description)
    print(rows["priority"].value_counts().to_dict())