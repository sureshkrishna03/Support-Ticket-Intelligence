from datasets import load_dataset
from pathlib import Path
import pandas as pd

dataset = load_dataset("s2pidape/support-ticket-dataset",
    "humans_only", split="train",streaming=True)

sample=list(dataset.take(10000))

df=pd.DataFrame(sample)

output_path=Path("data/raw/support_tickets_sample.csv")

df.to_csv(output_path, index=False)

print("Sample download successfully")
print(f"Sample dataset saved to {output_path}")
print("Rows",len(df))
print("Columns",len(df.columns))
