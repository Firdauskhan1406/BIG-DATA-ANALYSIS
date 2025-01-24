# BIG-DATA-ANALYSIS
Objective of the Notebook
The uploaded notebook focuses on customer churn prediction using a machine learning pipeline. It involves preprocessing, model training, evaluation, and insights extraction. The goal is to predict whether a customer will churn based on features like website interaction and customer behavior.

Key Steps and Outputs
1. Data Preprocessing
Numerical columns (e.g., BounceRates, ExitRates, PageValues) are scaled using StandardScaler.
Categorical columns (e.g., VisitorType, TrafficType) are encoded using OneHotEncoder.
This is done using a ColumnTransformer to create a combined preprocessing pipeline.
2. Model Pipeline
A Pipeline is used to chain preprocessing and modeling steps.
LogisticRegression is the chosen model for prediction.
3. Model Training
The pipeline is fitted on training data to predict churn outcomes.
4. Evaluation
Confusion Matrix:
lua
Copy
Edit
[[ 204  371]
 [  62 3062]]
Classification Report:
yaml
Copy
Edit
             precision    recall  f1-score   support

          0       0.77      0.35      0.49       575
          1       0.89      0.98      0.93      3124

   accuracy                           0.88      3699
  macro avg       0.83      0.67      0.71      3699
yaml
Copy
Edit
weighted avg       0.87      0.88      0.86      3699
go
Copy
Edit
 ```
ROC AUC Score: 0.8857.
5. Feature Importance
The coefficients from LogisticRegression show the impact of features on churn prediction. For example:
ExitRates: 0.771
TrafficType_9: 0.892
PageValues: -1.470.
Libraries and Tools Used
Scikit-learn:
Pipeline for combining preprocessing and model training.
ColumnTransformer for handling numerical and categorical data.
LogisticRegression for classification.
Pandas: For analyzing feature importance and manipulating datasets.
Metrics: confusion_matrix, classification_report, and roc_auc_score for evaluating the model.
Deliverables
A comprehensive machine learning pipeline.
Insights on feature importance and prediction accuracy.
Clear evaluation metrics demonstrating the model's performance.
