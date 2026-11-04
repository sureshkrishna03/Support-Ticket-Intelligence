import pandas as pd

df=pd.read_csv("D:\\mlproject\\Support-Ticket-Intelligence\\data\\raw\\support_tickets_sample.csv")

#top five rows
print("Top five rows of the dataset")
print(df.head())

#Columns in the dataset
print("Columns in the dataset")
print("Columns",df.columns.tolist())

#info of the dataset
print("Info of the dataset")
print(df.info())

#statistics of the dataset
print("Statistics of the dataset")
print(df.describe())

#count the missing values in the dataset
print("Count the missing values in the dataset")
print(df.isnull().sum())

#check the duplicated rows in the dataset
print("Check the duplicated rows in the dataset")
print(df.duplicated().sum())

# Duplicate ticket IDs
print("\n--- DUPLICATE TICKET IDs ---")
print(df["ticket_id"].duplicated().sum())

# Duplicate descriptions
print("\n--- DUPLICATE DESCRIPTIONS ---")
print(df["issue_description"].duplicated().sum())

# Target column distributions
target_columns = ["category", "priority", "assigned_team"]

for column in target_columns:
    print(f"\n--- {column.upper()} DISTRIBUTION ---")

    print(df[column].value_counts())

    print("\nPercentage:")
    print(
        df[column]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    # Check completely duplicated rows
print("\n--- DUPLICATE COMPLETE ROWS ---")
print(df.duplicated().sum())

# Display repeated descriptions
print("\n--- REPEATED DESCRIPTIONS ---")

duplicate_mask = df["issue_description"].duplicated(keep=False)

print(
    df.loc[
        duplicate_mask,
        ["issue_description", "category", "priority", "assigned_team"]
    ].head(15).to_string(index=False)
)

# Check whether identical descriptions have different categories
print("\n--- DESCRIPTION VS CATEGORY ---")

print(
    df.groupby("issue_description")["category"]
    .nunique()
    .value_counts()
)

# Check whether repeated descriptions have consistent labels

targets = ["category", "priority", "assigned_team"]

for target in targets:
    consistency = (
        df.groupby("issue_description")[target]
        .nunique()
    )

    print(f"\n--- DESCRIPTION VS {target.upper()} ---")
    print("Descriptions with one unique label:",
          (consistency == 1).sum())

    print("Descriptions with multiple labels:",
          (consistency > 1).sum())

# Feature and target relationship investigation

print("\n--- UNIQUE VALUES PER COLUMN ---")

columns_to_check = [
    "issue_description",
    "issue_type_id",
    "category",
    "priority",
    "assigned_team",
    "issue_complexity_score"
]

for column in columns_to_check:
    print(column, ":", df[column].nunique())


# Check issue type to category mapping
print("\n--- ISSUE TYPE VS CATEGORY ---")

mapping = (
    df.groupby("issue_type_id")["category"]
    .nunique()
)

print("Issue types mapped to one category:",
      (mapping == 1).sum())

print("Issue types mapped to multiple categories:",
      (mapping > 1).sum())


# Check the relationship between issue type and category
print("\n--- ISSUE TYPE AND CATEGORY TABLE ---")

print(pd.crosstab(df["issue_type_id"], df["category"]))