
import pandas as pd

# Load the dataset with policy labels
df = pd.read_csv("tls_policy_labeled.csv")

print("Dataset shape:", df.shape)

# Check the target labels
print("\nLabel distribution:")
print(df["encryption_policy_label"].value_counts(dropna=False))

# Check the percentage of each label
print("\nLabel percentages:")
print(
    df["encryption_policy_label"]
    .value_counts(normalize=True, dropna=False)
    .mul(100)
    .round(2)
)

# Inspect the TLS-related columns
tls_columns = [
    "tls.cver",
    "tls.sver",
    "tls.ssv",
    "tls.scs",
    "tls.alpn",
    "tls.sni",
    "tls.ja3",
    "tls.ja4",
    "tls.ja3s",
    "tls.ja4s",
    "encryption_policy_label",
]

available_columns = [
    col for col in tls_columns if col in df.columns
]

print("\nTLS columns available:")
print(available_columns)

print("\nSample records:")
print(df[available_columns].head(10).to_string(index=False))

print("\nMissing values in selected columns:")
print(df[available_columns].isnull().sum())
