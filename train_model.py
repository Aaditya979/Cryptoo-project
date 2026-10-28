
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
    accuracy_score
)

# 1. Load and clean
df = pd.read_csv("tls_policy_labeled.csv")
df = df.drop_duplicates().copy()

df = df[
    df["encryption_policy_label"].isin(
        ["Approved", "Not Approved"]
    )
].copy()

print("Dataset shape:", df.shape)
print("\nClass counts:")
print(df["encryption_policy_label"].value_counts())

# 2. Select model inputs
drop_columns = [
    "encryption_policy_label",
    "tls_version",
    "meta.sample.id",
    "meta.malware.family",
    "meta.application.name",
    "tls.rec",
    "source_file",
    "bs", "ps", "br", "pr", "dp", "sp",
    "da", "sa", "ts", "td"
]

X = df.drop(columns=drop_columns, errors="ignore")
y = df["encryption_policy_label"]

X = X.dropna(axis=1, how="all")

# Normalize categorical metadata while preserving missing values
X = X.astype(object)

for col in X.columns:
    X[col] = X[col].map(
        lambda v: str(v) if pd.notna(v) else np.nan
    )

# 3. Split before training
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# 4. Preprocessing
preprocessor = ColumnTransformer([
    (
        "categorical",
        Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]),
        list(X.columns)
    )
])

# 5. Model
model = Pipeline([
    ("preprocessor", preprocessor),
    (
        "classifier",
        RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
    )
])

print("\nTraining Random Forest...")
model.fit(X_train, y_train)

# 6. Evaluate on the held-out test set
predictions = model.predict(X_test)

print("\nCLASSIFICATION REPORT")
print(classification_report(
    y_test,
    predictions,
    labels=["Approved", "Not Approved"],
    zero_division=0
))

print("CONFUSION MATRIX")
print(confusion_matrix(
    y_test,
    predictions,
    labels=["Approved", "Not Approved"]
))

print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
print(
    "Balanced accuracy:",
    round(balanced_accuracy_score(y_test, predictions), 4)
)

# 7. Save model and feature columns
joblib.dump(model, "tls_random_forest.joblib")
joblib.dump(list(X.columns), "tls_feature_columns.joblib")

# 8. Save held-out test results for dashboard
test_results = pd.DataFrame({
    "actual": y_test.to_numpy(),
    "predicted": predictions
})

test_results.to_csv("tls_test_results.csv", index=False)

print("\nSaved:")
print("- tls_random_forest.joblib")
print("- tls_feature_columns.joblib")
print("- tls_test_results.csv")
