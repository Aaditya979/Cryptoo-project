
import pandas as pd
from pathlib import Path

# Load the dataset from the current project folder
file_path = Path(__file__).parent / "malware_sample.csv"
df = pd.read_csv(file_path, low_memory=False)

print("=" * 50)
print("DATASET OVERVIEW")
print("=" * 50)

print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\nAll column names:")
for column in df.columns:
    print("-", column)

print("\n" + "=" * 50)
print("MISSING VALUES")
print("=" * 50)

print(df.isnull().sum().sort_values(ascending=False).to_string())

# Inspect the TLS fields that may help define our policy
tls_columns = ["tls.cver", "tls.sver", "tls.scs"]

for column in tls_columns:
    print("\n" + "=" * 50)
    print(f"VALUES IN {column}")
    print("=" * 50)

    if column in df.columns:
        print(df[column].value_counts(dropna=False).to_string())
    else:
        print("Column not found!")

print("\n" + "=" * 50)
print("EXISTING MALWARE LABELS")
print("=" * 50)

print(df["meta.malware.family"].value_counts().head(15).to_string())

print("\nDataset inspection completed.")
