
import pandas as pd
from pathlib import Path

df = pd.read_csv(
    Path(__file__).parent / "malware_sample.csv",
    low_memory=False
)

# Examine TLS 1.3 cipher-suite records
tls13 = df[df["tls.scs"].astype("string").isin(["1301", "1302"])]

print("TLS 1.3 suite counts:")
print(tls13["tls.scs"].value_counts())

print("\nTLS extension values:")
print(tls13["tls.sext"].value_counts(dropna=False).head(20))

print("\nServer extension values:")
print(tls13["tls.ssv"].value_counts(dropna=False).head(20))

print("\nServer JA4S values:")
print(tls13["tls.ja4s"].value_counts(dropna=False).head(20))

print("\nClient offered cipher suites (examples):")
print(tls13["tls.csg"].head(5).to_string(index=False))

print("\nSelected cipher suite and metadata inspection complete.")
