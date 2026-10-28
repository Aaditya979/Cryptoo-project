
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent
df = pd.read_csv(BASE_DIR / "malware_sample.csv", low_memory=False)

print("=" * 60)
print("TLS VERSION FIELD COMPARISON")
print("=" * 60)

print("\ntls.cver values:")
print(df["tls.cver"].value_counts(dropna=False).to_string())

print("\ntls.sver values:")
print(df["tls.sver"].value_counts(dropna=False).to_string())

print("\nVersion combinations for each selected cipher suite:")
print(
    df.groupby(
        ["tls.cver", "tls.sver", "tls.scs"],
        dropna=False
    ).size().to_string()
)

print("\n" + "=" * 60)
print("EXAMPLES WITH TLS 1.3 CIPHER SUITES")
print("=" * 60)

tls13 = df[df["tls.scs"].astype("string").isin(["1301", "1302"])]

columns = [
    "tls.cver",
    "tls.sver",
    "tls.scs",
    "tls.sext",
    "tls.ja3",
    "tls.ja4",
    "tls.ja3s",
    "tls.ja4s",
]

columns = [column for column in columns if column in df.columns]

print("Number of TLS 1.3 cipher-suite records:", len(tls13))
print(tls13[columns].head(10).to_string(index=False))

print("\nDiagnostic completed.")
