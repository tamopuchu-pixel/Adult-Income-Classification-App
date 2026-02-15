# ==========================================
# Streamlit App - Adult Income Classification
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================
# Page Config
# ==========================================

st.set_page_config(page_title="Adult Income Classifier", layout="wide")

st.title("Adult Income Classification App")
st.write("Download test dataset or upload your own CSV file for evaluation.")


# ==========================================
# Load Saved Models & Artifacts
# ==========================================

MODEL_PATH = "model"

feature_columns = joblib.load(os.path.join(MODEL_PATH, "feature_columns.pkl"))
scaler = joblib.load(os.path.join(MODEL_PATH, "scaler.pkl"))

models = {
    "Logistic Regression": joblib.load(os.path.join(MODEL_PATH, "logistic_regression.pkl")),
    "Decision Tree": joblib.load(os.path.join(MODEL_PATH, "decision_tree.pkl")),
    "KNN": joblib.load(os.path.join(MODEL_PATH, "knn.pkl")),
    "Naive Bayes": joblib.load(os.path.join(MODEL_PATH, "naive_bayes.pkl")),
    "Random Forest": joblib.load(os.path.join(MODEL_PATH, "random_forest.pkl")),
    "XGBoost": joblib.load(os.path.join(MODEL_PATH, "xgboost.pkl")),
}


# ==========================================
# Sidebar - Model Selection
# ==========================================

st.sidebar.header("Model Selection")
selected_model_name = st.sidebar.selectbox(
    "Choose a Model",
    list(models.keys())
)

selected_model = models[selected_model_name]


# ==========================================
# 📥 Download Test Dataset Feature
# ==========================================

st.header("Download Sample Test Dataset")

if os.path.exists("Data/adult.test"):
    with open("Data/adult.test", "rb") as file:
        st.download_button(
            label="Download adult.test",
            data=file,
            file_name="adult.test",
            mime="text/csv"
        )
else:
    st.warning("adult.test file not found in Data folder.")


# ==========================================
# 📤 Upload Test Dataset
# ==========================================

st.header("Upload Test Dataset")
uploaded_file = st.file_uploader(
    "Upload Test Dataset",
    type=["csv", "test", "txt"]
)


if uploaded_file is not None:

    df = pd.read_csv(
        uploaded_file,
        header=None,
        names=column_names,
        skiprows=1,              # skip first metadata row
        sep=",",
        engine="python",         # safer parser
        skipinitialspace=True    # removes spaces after commas
    )

    column_names = [
        "age", "workclass", "fnlwgt", "education", "education-num",
        "marital-status", "occupation", "relationship", "race",
        "sex", "capital-gain", "capital-loss", "hours-per-week",
        "native-country", "income"
    ]

    df.columns = column_names

    # Clean
    df.replace("?", np.nan, inplace=True)
    df.dropna(inplace=True)

    df["income"] = df["income"].str.replace(".", "", regex=False)

    df.drop("fnlwgt", axis=1, inplace=True)

    df["income"] = df["income"].apply(
        lambda x: 1 if x.strip() == ">50K" else 0
    )

    df = pd.get_dummies(df, drop_first=True)

    X_test = df.drop("income", axis=1)
    y_test = df["income"]

    # Align columns
    X_test = X_test.reindex(columns=feature_columns, fill_value=0)

    X_test_scaled = scaler.transform(X_test)

    # ==========================================
    # Predictions
    # ==========================================

    y_pred = selected_model.predict(X_test_scaled)
    y_prob = selected_model.predict_proba(X_test_scaled)[:, 1]

    # ==========================================
    # Metrics
    # ==========================================

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    st.header("Evaluation Metrics")

    col1, col2, col3 = st.columns(3)

    col1.metric("Accuracy", f"{accuracy:.4f}")
    col1.metric("AUC", f"{auc:.4f}")

    col2.metric("Precision", f"{precision:.4f}")
    col2.metric("Recall", f"{recall:.4f}")

    col3.metric("F1 Score", f"{f1:.4f}")
    col3.metric("MCC", f"{mcc:.4f}")

    # ==========================================
    # Confusion Matrix
    # ==========================================

    st.header("Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    st.pyplot(fig)

    # ==========================================
    # Classification Report
    # ==========================================

    st.header("Classification Report")

    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    st.dataframe(report_df)

else:
    st.info("Download adult.test or upload a CSV file to evaluate the model.")
