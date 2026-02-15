**Adult-Income-Classification-App**

1.  **Problem Statement**

The objective of this project is to build and evaluate multiple machine learning models to predict whether an individual's annual income exceeds **$50,000** based on demographic and employment-related attributes.

This is formulated as a **binary classification problem**, where:

- 0 → Income ≤ 50K
- 1 → Income > 50K

The key goals of this project are:

- Perform structured data preprocessing and feature engineering
- Train and evaluate multiple machine learning algorithms
- Compare model performance using multiple evaluation metrics
- Deploy the trained models using a Streamlit web application

Instead of relying only on accuracy, this project evaluates models using a comprehensive set of metrics including:

- Accuracy
- AUC (Area Under ROC Curve)
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

This ensures a robust and balanced evaluation of model performance.

1.  **Dataset Description**

This project uses the **Adult Income Dataset**, available from the **UCI Machine Learning Repository**.

**Dataset Overview**

- Total Instances: 48,842
- Training Samples: 32,561
- Test Samples: 16,281
- Number of Input Features: 14
- Target Variable: income
- Problem Type: Binary Classification

The dataset contains demographic and employment-related information collected from U.S. Census records.

**Feature Categories**

**Numerical Features**

- age
- education-num
- capital-gain
- capital-loss
- hours-per-week

**Categorical Features**

- workclass
- education
- marital-status
- occupation
- relationship
- race
- sex
- native-country

**Target Variable**

The target variable income has two categories:

- <=50K
- \>50K

For modeling purposes, it was converted into binary format:

- <=50K → 0
- \>50K → 1

1.  **MODELS USED**

**The following six supervised machine learning algorithms were implemented and evaluated:**

1.  **Logistic Regression**
    A linear model that estimates class probabilities using the logistic (sigmoid) function. It serves as a strong baseline for binary classification problems.
2.  **Decision Tree Classifier** 
    A non-linear model that splits data recursively based on feature values to form decision rules.
3.  **K-Nearest Neighbors (KNN)**
    A distance-based algorithm that classifies a sample based on the majority class among its _k_ nearest neighbors.
4.  **Gaussian Naive Bayes** 
    A probabilistic classifier based on Bayes’ theorem with the assumption of conditional independence between features.
5.  **Random Forest Classifier** 
    An ensemble learning method that combines multiple decision trees using bagging to improve generalization and reduce overfitting
6.  **XGBoost Classifier**
    An optimized gradient boosting framework that builds trees sequentially to minimize prediction error. Known for high performance and robustness.

**Model Evaluation Metrics**

Each model was evaluated using the following performance metrics:

- **Accuracy** – Overall correctness of predictions
- **AUC (Area Under ROC Curve)** – Ability to distinguish between classes
- **Precision** – Correctness of positive predictions
- **Recall** – Ability to detect actual positives
- **F1 Score** – Harmonic mean of Precision and Recall
- **MCC (Matthews Correlation Coefficient)** – Balanced metric suitable for imbalanced datasets

**Model Comparison Table (Populated based on metrics achieved in prediction of test data in the app)**

| **Model Name** | **Accuracy** | **AUC** | **Precision** | **Recall** | **F1 Score** | **MCC** |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.7742 | 0.8206 | 0.9934 | 0.0814 | 0.1504 | 0.2489 |
| Decision Tree | 0.8017 | 0.7472 | 0.8774 | 0.2243 | 0.3573 | 0.3799 |
| KNN | 0.7659 | 0.7084 | 0.6565 | 0.0986 | 0.1715 | 0.1868 |
| Naive Bayes | 0.2457 | 0.50 | 0.2457 | 1.0 | 0.3945 | 0.00 |
| Random Forest | 0.8039 | 0.8274 | 0.9278 | 0.2189 | 0.3543 | 0.3931 |
| XGBoost | 0.8051 | 0.8580 | 0.9823 | 0.2105 | 0.3468 | 0.4034 |

**MODEL OBSERVATIONS**

**Dataset Considerations**

On analyzing the test dataset (adult.test) we see that it is imbalanced, The class distribution and percentages are given below:-

**Class Distribution**

- **<=50K:** 12,435 samples
- **\>50K:** 3,846 samples

**Class Percentages**

- **<=50K:** 76.38%
- **\>50K:** 23.62%

**Imbalance Ratio**

The majority class (<=50K) is about **3.2 times larger** than the minority class (>50K).

This explains:

- High accuracy (~0.80) across models
- Very low recall for the minority class
- High precision but poor F1 in many models

**Logistic Regression**

- Very **high precision (0.9934)** → when the model predicts the positive class, it is almost always correct.
- Extremely **low recall (0.0814)** → it misses most of the actual positive cases.
- This suggests the model is **very conservative in predicting the positive class**.
- Decent AUC (0.8206) indicates that the model ranks probabilities reasonably well, but the classification threshold may be poorly calibrated.
- Low F1 and moderate MCC confirm imbalance in prediction behavior.

**Decision Tree**

- Better balance between precision and recall compared to Logistic Regression.
- Recall improved significantly (0.2243).
- F1 score more than doubled compared to Logistic Regression.
- AUC lower than Logistic Regression, meaning probability ranking may be weaker.
- Higher MCC (0.3799) indicates better overall class correlation.

**KNN**

- Lower accuracy and AUC compared to tree-based models.
- Recall remains very low.
- Precision moderate but not outstanding.
- MCC is quite low (0.1868), showing weak predictive power.

**Naive Bayes**

- Recall is 1.0 → the model predicts **all instances as positive**.
- Accuracy is extremely low (0.2457).
- AUC = 0.50 → performance equivalent to random guessing.
- MCC = 0 → no correlation between predictions and true labels.

**Random Forest**

- High accuracy and strong AUC.
- Precision very high.
- Recall still low but slightly better than Logistic Regression.
- Good MCC (0.3931), indicating reliable performance across both classes.

**XGBoost**

- **Best AUC (0.8580)** → strongest probability ranking ability.
- Very high precision (0.9823).
- Recall remains modest.
- Highest MCC (0.4034) among all models.
- Slightly better overall than Random Forest.

**CONCLUSION**

XGBoost is the **best performing model overall** on this dataset followed by Random Forest.
