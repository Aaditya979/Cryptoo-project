
import pandas as pd

df = pd.read_csv("tls_policy_labeled.csv")

target = "encryption_policy_label"

# Show how many records each class has
print("CLASS COUNTS")
print(df[target].value_counts())

# Show the TLS version and selected cipher suite
print("\nTLS VERSION + SELECTED CIPHER SUITE")
print(
    df.groupby(
        ["tls.sver", "tls.scs"],
        dropna=False
    )[target]
    .value_counts()
    .to_string()
)

# Check duplicate rows
print("\nDuplicate rows:", df.duplicated().sum())

# Inspect columns that may reveal the label
print("\nALL COLUMNS")
for col in df.columns:
    print(col)
