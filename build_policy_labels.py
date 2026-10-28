
import pandas as pd
import ast
from pathlib import Path

BASE_DIR = Path(__file__).parent

INPUT_FILE = BASE_DIR / "malware_sample.csv"
OUTPUT_FILE = BASE_DIR / "tls_policy_labeled.csv"
MAPPING_FILE = BASE_DIR / "tls_policy_mapping.csv"


# 1. Load dataset
df = pd.read_csv(INPUT_FILE, low_memory=False)

print("Dataset loaded:", df.shape)


# 2. Cipher-suite names found in the dataset
cipher_suites = {
    "1301": "TLS_AES_128_GCM_SHA256",
    "1302": "TLS_AES_256_GCM_SHA384",
    "C02B": "TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256",
    "C02C": "TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384",
    "C02F": "TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256",
    "C030": "TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384",
    "C014": "TLS_ECDHE_RSA_WITH_AES_256_CBC_SHA",
}


# 3. Define preliminary project policy
mapping_rows = []

for suite in ["C02B", "C02C", "C02F", "C030"]:
    mapping_rows.append({
        "tls_version": "TLS 1.2",
        "cipher_suite": suite,
        "cipher_suite_name": cipher_suites[suite],
        "policy_status": "Approved",
        "reason": "AES-GCM suite permitted by the preliminary project policy."
    })

for suite in ["1301", "1302"]:
    mapping_rows.append({
        "tls_version": "TLS 1.3",
        "cipher_suite": suite,
        "cipher_suite_name": cipher_suites[suite],
        "policy_status": "Approved",
        "reason": "TLS 1.3 AES-GCM suite permitted by the preliminary project policy."
    })

mapping_rows.append({
    "tls_version": "TLS 1.0",
    "cipher_suite": "C014",
    "cipher_suite_name": cipher_suites["C014"],
    "policy_status": "Not Approved",
    "reason": "TLS 1.0 is below the project's minimum TLS 1.2 requirement."
})

mapping = pd.DataFrame(mapping_rows)
mapping.to_csv(MAPPING_FILE, index=False)


# 4. Normalize TLS metadata
def normalize_code(value):
    if pd.isna(value):
        return None
    return str(value).strip().upper()


def normalize_version(value):
    if pd.isna(value):
        return None

    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return None


def parse_server_version(value):
    """Parse values such as "['0304']" and return the version code."""
    if pd.isna(value):
        return None

    text = str(value).strip()

    try:
        parsed = ast.literal_eval(text)

        if isinstance(parsed, list) and parsed:
            return str(parsed[0]).strip().upper()

        if isinstance(parsed, str):
            return parsed.strip().upper()

    except (ValueError, SyntaxError):
        pass

    return text.upper()


df["server_version_code"] = df["tls.ssv"].apply(
    parse_server_version
)

df["suite_key"] = df["tls.scs"].apply(normalize_code)


# 5. Identify TLS version
server_version_map = {
    "0304": "TLS 1.3",
    "0303": "TLS 1.2",
    "0301": "TLS 1.0",
}

df["tls_version"] = df["server_version_code"].map(
    server_version_map
)

# Fall back only when the server version is unavailable.
fallback_version_map = {
    "304": "TLS 1.3",
    "303": "TLS 1.2",
    "301": "TLS 1.0",
}

fallback = df["tls.sver"].apply(
    normalize_version
).map(fallback_version_map)

df["tls_version"] = df["tls_version"].fillna(fallback)


# 6. Assign policy labels
policy_lookup = mapping.set_index(
    ["tls_version", "cipher_suite"]
)["policy_status"].to_dict()


def assign_policy(row):
    version = row["tls_version"]
    suite = row["suite_key"]

    if pd.isna(version) or pd.isna(suite):
        return "Unknown"

    return policy_lookup.get(
        (version, suite),
        "Unknown"
    )


df["encryption_policy_label"] = df.apply(
    assign_policy,
    axis=1
)


# 7. Save results
df.drop(
    columns=["server_version_code", "suite_key"],
    errors="ignore"
).to_csv(OUTPUT_FILE, index=False)


# 8. Print validation information
print("\nLABEL COUNTS")
print(df["encryption_policy_label"].value_counts(dropna=False))

print("\nTLS VERSION COUNTS")
print(df["tls_version"].value_counts(dropna=False))

print("\nVERSION + CIPHER SUITE + LABEL")
print(
    df.groupby(
        ["tls_version", "tls.scs", "encryption_policy_label"],
        dropna=False
    ).size().to_string()
)

print("\nFiles created:")
print(MAPPING_FILE.name)
print(OUTPUT_FILE.name)

print("\nPreliminary labeling completed.")
print("Validate the policy against NIST before training the model.")
