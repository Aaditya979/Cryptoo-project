
import pandas as pd
from pathlib import Path

df = pd.read_csv(
    Path(__file__).parent / "malware_sample.csv",
    low_memory=False
)

# Examine rows with TLS 1.3 cipher-suite identifiers
mask = df["tls.scs"].astype("string").str.upper().isin(
    ["1301", "1302"]
)

cols = [
    "tls.cver",
    "tls.sver",
    "tls.ssv",
    "tls.scs",
    "tls.ja4s",
]

print("Raw values for TLS 1.3 cipher-suite records:")
print(df.loc[mask, cols].head(15).to_string(index=False))

print("\nRaw tls.ssv values:")
print(df.loc[mask, "tls.ssv"].value_counts(dropna=False).to_string())

print("\nUnique tls.ssv values:")
print(df.loc[mask, "tls.ssv"].drop_duplicates().to_list())

print("\nUnique tls.ja4s values:")
print(df.loc[mask, "tls.ja4s"].drop_duplicates().to_list())
