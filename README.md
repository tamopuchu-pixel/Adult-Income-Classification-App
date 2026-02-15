**Adult-Income-Classification-App**

1.  **Problem Statement**

The objective of this project is to build and evaluate multiple machine learning models to predict whether an individual's annual income exceeds **$50,000** based on demographic and employment-related attributes.

This is formulated as a **binary classification problem**, where:

*   0 → Income ≤ 50K
*   1 → Income > 50K

The key goals of this project are:

*   Perform structured data preprocessing and feature engineering
*   Train and evaluate multiple machine learning algorithms
*   Compare model performance using multiple evaluation metrics
*   Deploy the trained models using a Streamlit web application

Instead of relying only on accuracy, this project evaluates models using a comprehensive set of metrics including:

*   Accuracy
*   AUC (Area Under ROC Curve)
*   Precision
*   Recall
*   F1 Score
*   Matthews Correlation Coefficient (MCC)

This ensures a robust and balanced evaluation of model performance.

1.  **Dataset Description**

This project uses the **Adult Income Dataset**, available from the **UCI Machine Learning Repository**.

**Dataset Overview**

*   Total Instances: 48,842
*   Training Samples: 32,561
*   Test Samples: 16,281
*   Number of Input Features: 14
*   Target Variable: income
*   Problem Type: Binary Classification

The dataset contains demographic and employment-related information collected from U.S. Census records.

**Feature Categories**

**Numerical Features**

*   age
*   education-num
*   capital-gain
*   capital-loss
*   hours-per-week

**Categorical Features**

*   workclass
*   education
*   marital-status
*   occupation
*   relationship
*   race
*   sex
*   native-country

**Target Variable**

The target variable income has two categories:

*   <=50K
*   \>50K

For modeling purposes, it was converted into binary format:

*   <=50K → 0
*   \>50K → 1

1.  **MODELS USED**

**The following six supervised machine learning algorithms were implemented and evaluated:**

1.  **Logistic Regression  
    **A linear model that estimates class probabilities using the logistic (sigmoid) function. It serves as a strong baseline for binary classification problems.
2.  **Decision Tree Classifier  
    **A non-linear model that splits data recursively based on feature values to form decision rules**.**
3.  **K-Nearest Neighbors (KNN)  
    **A distance-based algorithm that classifies a sample based on the majority class among its _k_ nearest neighbors.
4.  **Gaussian Naive Bayes  
    **A probabilistic classifier based on Bayes’ theorem with the assumption of conditional independence between features.
5.  **Random Forest Classifier  
    **An ensemble learning method that combines multiple decision trees using bagging to improve generalization and reduce overfitting**.**
6.  **XGBoost Classifier  
    **An optimized gradient boosting framework that builds trees sequentially to minimize prediction error. Known for high performance and robustness.

**Model Evaluation Metrics**

Each model was evaluated using the following performance metrics:

*   **Accuracy** – Overall correctness of predictions
*   **AUC (Area Under ROC Curve)** – Ability to distinguish between classes
*   **Precision** – Correctness of positive predictions
*   **Recall** – Ability to detect actual positives
*   **F1 Score** – Harmonic mean of Precision and Recall
*   **MCC (Matthews Correlation Coefficient)** – Balanced metric suitable for imbalanced datasets

**Model Comparison Table (Populated based on metrics achieved in prediction of test data in the app)**

| Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.7766 | 0.8206 | 0.9883 | 0.0916 | 0.1677 | 0.2634 |
| Decision Tree | 0.8110 | 0.7472 | 0.6871 | 0.4232 | 0.5238 | 0.4330 |
| KNN | 0.7669 | 0.7084 | 0.6331 | 0.1222 | 0.2048 | 0.2007 |
| Naive Bayes | 0.2457 | 0.50 | 0.2457 | 1.0 | 0.3945 | 0.00 |
| Random Forest | 0.8202 | 0.8274 | 0.7478 | 0.4046 | 0.5251 | 0.4567 |
| XGBoost | 0.8240 | 0.8533 | 0.7640 | 0.4103 | 0.5338 | 0.4694 |

**MODEL OBSERVATIONS**

**Dataset Considerations**

On analyzing the test dataset (adult.test) we see that it is imbalanced, The class distribution and percentages are given below:-

**Class Distribution**

*   **<=50K:** 12,435 samples
*   **\>50K:** 3,846 samples

**Class Percentages**

*   **<=50K:** 76.38%
*   **\>50K:** 23.62%

**Imbalance Ratio**

The majority class (<=50K) is about **3.23 times larger** than the minority class (>50K).

This explains:

*   High accuracy (~0.80) across models
*   Very low recall for the minority class
*   High precision but poor F1 in many models

In order to deal with this imbalance I have taken class\_weight="balanced" for Logistic Regression, Decision Tree and Random Forest models. For XGBoost I have taken scale\_pos\_weight=3.23, where 3.23 is the imbalance ratio.

The threshold has been by default kept at 0.39 as it gives the best results. However a slider has been included in the app to check the value of the metrices if the threshold is changed.

**Logistic Regression**

*   Very **high precision** → when the model predicts the positive class, it is almost always correct.
*   Extremely **low recall** → it misses most of the actual positive cases.
*   This suggests the model is **very conservative in predicting the positive class**.
*   Decent AUC indicates that the model ranks probabilities reasonably well, but the classification threshold may be poorly calibrated.
*   Low F1 and moderate MCC confirm imbalance in prediction behavior.

**Decision Tree**

*   Better balance between precision and recall compared to Logistic Regression.
*   Recall improved significantly (0.4232).
*   F1 score very strong compared to Logistic Regression (0.5238)
*   Higher MCC (0.4330) indicates better overall class correlation.
*   AUC (0.7472) is lower than ensemble models
*   Prone to Variance

**KNN**

*   Lower accuracy and AUC compared to tree-based models.
*   Recall remains low.
*   Precision moderate but not outstanding.
*   MCC is quite low (0.2007), showing weak predictive power.
*   KNN struggles because the data set is high-dimensional

**Naive Bayes**

*   Recall is 1.0 → the model predicts **all instances as positive**.
*   Accuracy is extremely low (0.2457).
*   AUC = 0.50 → performance equivalent to random guessing.
*   MCC = 0 → no correlation between predictions and true labels.

**Random Forest**

*   High accuracy and strong AUC.
*   Precision very high.
*   Strong balance between precision and recall
*   Good MCC (0.4567), indicating reliable performance across both classes.
*   Good F1 (0.5251)
*   It handles non-linearities, reduces variance and is robust to noise

**XGBoost**

*   **Best AUC (0.8533)** → strongest probability ranking ability.
*   Good Precision Balance
*   Strong Recall (0.4103)
*   Highest MCC (0.4694) among all models.
*   Captures complex feature interactions
*   Handles imbalance via scale\_pos\_weight
*   Benefits from threshold tuning

**CONCLUSION**

XGBoost is the **best performing model overall** on this dataset followed by Random Forest and Decision Tree.

XGBoost provides the best trade-off between minority detection, overall accuracy and balanced performance.

The results indicate that ensemble tree-based models outperform linear and distance-based models on the Adult dataset due to its nonlinear feature interactions. XGBoost achieved the best overall performance with the highest F1 score (0.5338) and MCC (0.4694), demonstrating strong minority class detection while maintaining high overall accuracy. Logistic Regression showed high precision but poor recall, indicating conservative predictions. Naive Bayes failed due to violation of independence assumptions. Overall, XGBoost is selected as the final model for deployment.
