
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import plotly.express as px

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix
)

st.set_page_config(
    page_title="TLS Encryption Policy Analyzer",
    page_icon="🔐",
    layout="wide"
)

st.title("🔐 TLS Encryption Policy Analyzer")
st.caption(
    "Machine Learning-Based Classification of TLS Metadata"
)

# Load project files
@st.cache_resource
def load_model():
    return joblib.load("tls_random_forest.joblib")

@st.cache_data
def load_dataset():
    return pd.read_csv("tls_policy_labeled.csv")

@st.cache_data
def load_test_results():
    return pd.read_csv("tls_test_results.csv")

try:
    model = load_model()
    df = load_dataset()
    test_results = load_test_results()
    feature_columns = joblib.load("tls_feature_columns.joblib")
except FileNotFoundError as error:
    st.error(
        f"Missing file: {error}. "
        "Run python train_model.py first."
    )
    st.stop()

# Sidebar
st.sidebar.title("Navigation")
section = st.sidebar.radio(
    "Choose section",
    [
        "Overview",
        "TLS Analytics",
        "Model Evaluation",
        "Classification Demo"
    ]
)

# Overview
if section == "Overview":
    st.header("Project Overview")

    total = len(df)
    approved = (df["encryption_policy_label"] == "Approved").sum()
    rejected = (
        df["encryption_policy_label"] == "Not Approved"
    ).sum()
    unknown = (
        df["encryption_policy_label"] == "Unknown"
    ).sum()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total records", f"{total:,}")
    c2.metric("Approved", f"{approved:,}")
    c3.metric("Not Approved", f"{rejected:,}")
    c4.metric("Unknown", f"{unknown:,}")

    st.subheader("Policy label distribution")

    counts = (
        df["encryption_policy_label"]
        .value_counts()
        .rename_axis("Policy label")
        .reset_index(name="Records")
    )

    fig = px.bar(
        counts,
        x="Policy label",
        y="Records",
        color="Policy label",
        title="Distribution of policy labels"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Policy labels are derived from the project's TLS rules. "
        "They do not independently prove that a connection is secure."
    )

# TLS analytics
elif section == "TLS Analytics":
    st.header("TLS Metadata Analytics")

    if "tls.scs" in df.columns:
        suite_counts = (
            df["tls.scs"]
            .fillna("Missing")
            .value_counts()
            .rename_axis("Cipher suite")
            .reset_index(name="Records")
        )

        st.subheader("Selected cipher suites")
        fig = px.bar(
            suite_counts,
            x="Cipher suite",
            y="Records",
            title="Cipher suite frequency"
        )
        st.plotly_chart(fig, use_container_width=True)

    if "tls_version" in df.columns:
        version_counts = (
            df["tls_version"]
            .fillna("Unknown")
            .value_counts()
            .rename_axis("TLS version")
            .reset_index(name="Records")
        )

        st.subheader("TLS version distribution")
        fig = px.bar(
            version_counts,
            x="TLS version",
            y="Records",
            title="TLS versions in the dataset"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Dataset preview")
    st.dataframe(df.head(100), use_container_width=True)

# Model evaluation
elif section == "Model Evaluation":
    st.header("Held-out Test Evaluation")

    actual = test_results["actual"]
    predicted = test_results["predicted"]

    accuracy = accuracy_score(actual, predicted)
    balanced = balanced_accuracy_score(actual, predicted)

    c1, c2, c3 = st.columns(3)
    c1.metric("Test records", len(test_results))
    c2.metric("Accuracy", f"{accuracy:.2%}")
    c3.metric("Balanced accuracy", f"{balanced:.2%}")

    st.subheader("Classification report")
    report = classification_report(
        actual,
        predicted,
        labels=["Approved", "Not Approved"],
        output_dict=True,
        zero_division=0
    )
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.round(3), use_container_width=True)

    st.subheader("Confusion matrix")

    labels = ["Approved", "Not Approved"]
    matrix = confusion_matrix(
        actual,
        predicted,
        labels=labels
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=labels,
        columns=labels
    )
    matrix_df.index.name = "Actual"
    matrix_df.columns.name = "Predicted"

    fig = px.imshow(
        matrix_df,
        text_auto=True,
        aspect="auto",
        title="Actual versus predicted labels"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        test_results.head(100),
        use_container_width=True
    )

    st.warning(
        "The test split contains very few Not Approved examples. "
        "Scores may not generalize to new connections. "
        "The label is policy-derived, and TLS version/cipher-suite "
        "metadata may directly reveal the label."
    )

# Classification demo
elif section == "Classification Demo":
    st.header("Classify TLS Metadata")

    st.write(
        "Upload a CSV containing the model's input columns. "
        "The model will predict Approved or Not Approved. "
        "This demo does not independently verify full TLS security."
    )

    uploaded = st.file_uploader(
        "Upload TLS metadata CSV",
        type=["csv"]
    )

    if uploaded is not None:
        incoming = pd.read_csv(uploaded)

        # Add missing model columns and preserve expected order
        for col in feature_columns:
            if col not in incoming.columns:
                incoming[col] = np.nan

        model_input = incoming[feature_columns].copy()

        for col in model_input.columns:
            model_input[col] = model_input[col].map(
                lambda v: str(v) if pd.notna(v) else np.nan
            )

        if st.button("Run classification", type="primary"):
            predictions = model.predict(model_input)

            output = incoming.copy()
            output["model_prediction"] = predictions

            # Mark records with missing core metadata as Unknown.
            # This is a conservative demo rule, not a full TLS parser.
            if "tls.sver" in incoming.columns and "tls.scs" in incoming.columns:
                missing_core = (
                    incoming["tls.sver"].isna()
                    | incoming["tls.scs"].isna()
                )
                output.loc[missing_core, "model_prediction"] = "Unknown"

            st.subheader("Prediction summary")
            st.dataframe(
                output["model_prediction"]
                .value_counts()
                .rename_axis("Prediction")
                .reset_index(name="Records"),
                use_container_width=True
            )

            st.subheader("Prediction results")
            st.dataframe(output, use_container_width=True)

            csv_data = output.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download predictions CSV",
                data=csv_data,
                file_name="tls_predictions.csv",
                mime="text/csv"
            )

st.divider()
st.caption(
    "Research prototype | NIST-informed project policy | "
    "Not a certification of connection security"
)

