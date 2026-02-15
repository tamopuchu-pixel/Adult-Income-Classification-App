# ==========================================
# Streamlit App - Adult Income Classification (Professional Version)
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import io
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
    roc_curve
)

# ==========================================
# Page Config
# ==========================================

st.set_page_config(page_title="Adult Income Classifier", layout="wide")
st.title("Adult Income Classification Dashboard")
st.markdown("Model evaluation on the UCI Adult Income dataset.")

# ==========================================
# Load Models
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
# Sidebar
# ==========================================

st.sidebar.title("⚙️ Settings")

selected_model_name = st.sidebar.selectbox(
    "Select Model",
    list(models.keys())
)

selected_model = models[selected_model_name]

# Download sample
st.sidebar.subheader("Download Sample Dataset")

if os.path.exists("Data/adult.test"):
    with open("Data/adult.test", "rb") as file:
        st.sidebar.download_button(
            "Download adult.test",
            data=file,
            file_name="adult.test"
        )

# Upload file
st.sidebar.subheader("Upload Test Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload adult.test file",
    type=["csv", "test", "txt"]
)

# ==========================================
# Main Logic
# ==========================================

if uploaded_file is not None:

    with st.spinner("Processing dataset..."):

        column_names = [
            "age", "workclass", "fnlwgt", "education", "education-num",
            "marital-status", "occupation", "relationship", "race",
            "sex", "capital-gain", "capital-loss", "hours-per-week",
            "native-country", "income"
        ]

        file_content = uploaded_file.read().decode("utf-8", errors="ignore")

        df = pd.read_csv(
            io.StringIO(file_content),
            header=None,
            names=column_names,
            skiprows=1,
            sep=r",\s*",
            engine="python",
            on_bad_lines="skip"
        )

        df.replace("?", np.nan, inplace=True)
        df.dropna(inplace=True)

        df["income"] = df["income"].str.replace(".", "", regex=False)
        df.drop("fnlwgt", axis=1, inplace=True)

        df["income"] = df["income"].apply(
            lambda x: 1 if x.strip() == ">50K" else 0
        )

        # Strip whitespace from ALL string columns
        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].str.strip()


        df = pd.get_dummies(df, drop_first=True)

        X_test = df.drop("income", axis=1)
        y_test = df["income"]

        X_test = X_test.reindex(columns=feature_columns, fill_value=0)
        X_test_scaled = scaler.transform(X_test)

        # Predictions
        y_prob = selected_model.predict_proba(X_test_scaled)[:, 1]

        threshold = st.sidebar.slider(
            "Classification Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.01
        )

        y_pred = (y_prob >= threshold).astype(int)



        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)

    # ==========================================
    # Tabs Layout
    # ==========================================

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Metrics", "📈 ROC & Confusion Matrix", "📋 Reports", "📂 Data & Comparison"]
    )

    # ================= Metrics Tab =================
    with tab1:

        col1, col2, col3 = st.columns(3)

        col1.metric("Accuracy", f"{accuracy:.4f}")
        col1.metric("AUC", f"{auc:.4f}")

        col2.metric("Precision", f"{precision:.4f}")
        col2.metric("Recall", f"{recall:.4f}")

        col3.metric("F1 Score", f"{f1:.4f}")
        col3.metric("MCC", f"{mcc:.4f}")

    # ================= ROC + CM =================
    with tab2:

        col1, col2 = st.columns(2)

        # ROC
        with col1:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
            ax.plot([0, 1], [0, 1], linestyle="--")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("ROC Curve")
            ax.legend()
            st.pyplot(fig)

        # Confusion Matrix
        with col2:
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["<=50K", ">50K"],
                yticklabels=["<=50K", ">50K"],
                ax=ax
            )
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

    # ================= Classification Report =================
    with tab3:

        report = classification_report(
            y_test,
            y_pred,
            target_names=["<=50K", ">50K"],
            output_dict=True
        )

        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df)

    # ================= Data & Comparison =================
    with tab4:

        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        st.subheader("Class Distribution")
        fig, ax = plt.subplots()
        y_test.value_counts().plot(kind="bar", ax=ax)
        ax.set_xticklabels(["<=50K", ">50K"], rotation=0)
        st.pyplot(fig)

        st.subheader("Model Comparison")

        comparison_results = {}

        for name, model in models.items():
            y_pred_temp = model.predict(X_test_scaled)
            y_prob_temp = model.predict_proba(X_test_scaled)[:, 1]

            comparison_results[name] = [
                accuracy_score(y_test, y_pred_temp),
                roc_auc_score(y_test, y_prob_temp),
                f1_score(y_test, y_pred_temp)
            ]

        comparison_df = pd.DataFrame(
            comparison_results,
            index=["Accuracy", "AUC", "F1 Score"]
        ).T

        st.dataframe(comparison_df.round(4))

        # Feature Importance
        if selected_model_name in ["Random Forest", "Decision Tree", "XGBoost"]:

            st.subheader("Feature Importance")

            importances = selected_model.feature_importances_
            fi_df = pd.DataFrame({
                "Feature": feature_columns,
                "Importance": importances
            }).sort_values(by="Importance", ascending=False).head(15)

            fig, ax = plt.subplots()
            ax.barh(fi_df["Feature"], fi_df["Importance"])
            ax.invert_yaxis()
            st.pyplot(fig)

        # Download predictions
        output_df = df.copy()
        output_df["Predicted Income"] = y_pred

        csv = output_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Predictions",
            data=csv,
            file_name="predictions.csv",
            mime="text/csv"
        )

else:
    st.info("Upload adult.test file from the UCI Adult dataset to begin evaluation.")
