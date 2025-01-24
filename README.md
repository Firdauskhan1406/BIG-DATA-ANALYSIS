# BIG-DATA-ANALYSIS

"COMAPNY" : CODTECH IT SOLUTIONS

"NAME" : FIRDAUS KHAN

"INTERN ID" : CT08NJP

"DOMAIN": DATA ANALYTICS

"DURATION: 4 WEEKS

"MENTOR" : NEELA SANTOSH


##OUTPUT OF THE CODES
![Image](https://github.com/user-attachments/assets/841f0d04-129f-4d89-be5c-a8c625ce8929)

![Image](https://github.com/user-attachments/assets/957c445e-450b-4142-ba97-3f946e2f81fb)

![Image](https://github.com/user-attachments/assets/11bf43eb-8a5c-4d86-a0e5-5fcc5b535f4b)

Objective of the Notebook
The Customer Churn Prediction project focuses on predicting whether a customer will stop using a service based on behavioral and transactional data. This prediction helps businesses retain customers by identifying potential churners early and implementing proactive measures to engage them.

Objective of the Project
The main objective is to develop a machine learning model capable of accurately predicting customer churn. By analyzing behavioral features like website usage and customer interaction patterns, the project provides actionable insights for improving customer retention strategies.

Steps Involved
Data Preprocessing: The dataset contains both numerical and categorical features, requiring different preprocessing techniques:



Numerical Features (e.g., BounceRates, ExitRates, Administrative_Duration): These features were scaled using the StandardScaler, which standardizes values by removing the mean and scaling them to unit variance.
Categorical Features (e.g., VisitorType, TrafficType): These were encoded using OneHotEncoder, converting them into binary vectors, ensuring the machine learning model understands the categorical data.
The ColumnTransformer was used to combine these transformations into a single preprocessing pipeline.

Pipeline Implementation: A Pipeline was created to streamline the preprocessing and model training steps. This ensures reproducibility and simplicity, as the pipeline allows the same preprocessing steps to be applied to both the training and test datasets without manual intervention.

Model Training: The machine learning model used for this project was LogisticRegression. This algorithm is ideal for binary classification tasks like churn prediction because it provides probabilistic outputs and interpretable coefficients.



Model Evaluation: The model’s performance was evaluated using the following metrics:

Confusion Matrix: Showed the number of correct and incorrect predictions for churn and non-churn classes.
Classification Report: Included precision, recall, and F1-score to evaluate prediction quality.
ROC AUC Score: At 0.8857, this score indicates strong discrimination between churn and non-churn customers.
Feature Importance Analysis: The coefficients from the logistic regression model revealed the impact of individual features. For example:



High ExitRates significantly increase churn likelihood.
High PageValues reduce churn probability, suggesting that engaged customers are less likely to churn.
Specific TrafficType categories, such as TrafficType_9, strongly correlate with churn.
Insights and Impact


The project highlights the importance of customer behavior metrics in predicting churn. Features like ExitRates and PageValues provide direct indicators of customer satisfaction and engagement. By understanding these factors, businesses can implement targeted interventions, such as personalized offers or improved website experiences, to reduce churn rates.
The uploaded notebook focuses on customer churn prediction using a machine learning pipeline. It involves preprocessing, model training, evaluation, and insights extraction. The goal is to predict whether a customer will churn based on features like website interaction and customer behavior.
Tools and Libraries
Key libraries include:



Scikit-learn for model building, preprocessing, and evaluation.
Pandas for data manipulation and feature analysis.
Metrics like roc_auc_score for assessing model quality

Loading Data: The dataset is loaded into the notebook using standard libraries like pandas. The columns include features such as:

Behavioral metrics: BounceRates, ExitRates, PageValues.
Temporal metrics: Administrative_Duration, Informational_Duration, ProductRelated_Duration.
Categorical metrics: VisitorType, TrafficType.
Handling Missing Values:

Any missing values in the dataset are filled using median or mode, ensuring the absence of NaN values during modeling.
Scaling Numerical Features:

Columns like BounceRates and PageValues are scaled using the StandardScaler. This ensures that all numerical features are on the same scale, preventing features with larger magnitudes from dominating the model.
Encoding Categorical Features:

Categorical columns like VisitorType and TrafficType are transformed into numerical representations using OneHotEncoder. This creates binary columns for each category, allowing the model to understand categorical data effectively.
ColumnTransformer:



Both numerical and categorical transformations are combined into a single ColumnTransformer, streamlining the preprocessing steps.
3. Pipeline Creation
The Pipeline object is a powerful tool provided by scikit-learn. It integrates preprocessing and modeling into a single workflow, making the process repeatable and scalable.

Pipeline Steps:



Preprocessing:
The ColumnTransformer applies the scaling and encoding transformations.
Model:
The LogisticRegression classifier is added as the final step in the pipeline. This model is well-suited for binary classification tasks like churn prediction.
4. Data Splitting
Before training, the dataset is split into training and testing sets using train_test_split. This ensures that the model's performance is evaluated on unseen data, providing a realistic estimate of its accuracy.



5. Model Training
The pipeline is fitted to the training data using the fit method. During this step:



The preprocessing transformations are applied to the training set.
The LogisticRegression model learns patterns in the data to predict churn.
6. Predictions
After training, the model is used to make predictions on the test set. Two types of predictions are generated:



Binary Predictions:
The predict method provides binary outputs indicating whether a customer will churn.
Probability Predictions:
The predict_proba method outputs the probability of each class (churn or not churn). The probabilities help in threshold-based decision-making.


7. Evaluation
Evaluation metrics are critical for assessing model performance. The notebook uses several metrics:

Confusion Matrix:
Displays the true positives, true negatives, false positives, and false negatives. It is a key tool for understanding prediction errors.


Classification Report:
Provides precision, recall, and F1-score for each class. These metrics give a detailed view of the model's performance.
Precision: The proportion of true positive predictions among all positive predictions.
Recall: The proportion of true positives detected among all actual positives.
F1-Score: The harmonic mean of precision and recall


ROC AUC Score:
Measures the model's ability to distinguish between classes. A score closer to 1 indicates excellent performance.
8. Feature Importance
Logistic regression coefficients provide insights into the importance of each feature. The coefficients indicate the direction and magnitude of each feature's effect on the prediction:



Positive coefficients: Features that increase the likelihood of churn.
Negative coefficients: Features that decrease the likelihood of churn.



Key Steps and Outputs
1. Data Preprocessing
Numerical columns (e.g., BounceRates, ExitRates, PageValues) are scaled using StandardScaler.
Categorical columns (e.g., VisitorType, TrafficType) are encoded using OneHotEncoder.
This is done using a ColumnTransformer to create a combined preprocessing pipeline.


3. Model Pipeline
A Pipeline is used to chain preprocessing and modeling steps.
LogisticRegression is the chosen model for prediction.




5. Model Training
The pipeline is fitted on training data to predict churn outcomes.


7. Evaluation




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



Loading and Exploring the Dataset
The dataset is loaded into a Pandas DataFrame for preliminary analysis. Key tasks include:

Previewing the data: Using methods like .head() and .info() to understand the structure and identify data types.
Checking for missing values: Missing values are handled later during preprocessing.
The dataset contains features like BounceRates, ExitRates, and PageValues, which represent customer behavior on the website. The target variable is likely labeled as Churn, indicating whether a customer has left the service.

3. Feature Engineering
Features are categorized into numerical and categorical groups:

Numerical features include columns like BounceRates, ExitRates, and PageValues.
Categorical features include VisitorType and TrafficType.
This distinction is critical for applying appropriate preprocessing techniques to each type of data.

4. Data Preprocessing
A ColumnTransformer is used to preprocess the data:

Numerical features: Scaled using StandardScaler to standardize the data, ensuring that all features contribute equally to the model.
Categorical features: Encoded using OneHotEncoder to convert categorical variables into binary vectors.
The ColumnTransformer combines these transformations into a unified preprocessing step, simplifying the pipeline creation.

5. Creating a Machine Learning Pipeline
A Pipeline is created to chain together preprocessing and model training steps. The pipeline includes:

Preprocessor: The ColumnTransformer defined earlier.
Classifier: A LogisticRegression model, which is well-suited for binary classification tasks like churn prediction.
Using a pipeline ensures that preprocessing steps are applied consistently to both training and testing data, reducing the risk of data leakage.

6. Splitting the Data
The dataset is split into training and testing sets using train_test_split. The training set is used to fit the model, while the testing set evaluates its performance. The split ratio (commonly 80:20 or 70:30) ensures the model has sufficient data for learning and testing.

7. Training the Model
The fit method trains the pipeline on the training data. During this process:

The preprocessor standardizes numerical features and encodes categorical features.
The LogisticRegression model learns the relationship between features and the target variable (Churn).
8. Making Predictions
After training, the model makes predictions on the test set:

Binary Predictions: Using predict(), the model outputs whether each customer is likely to churn (0 or 1).
Probabilistic Predictions: Using predict_proba(), the model provides the probability of each customer belonging to the churn class.
9. Evaluating the Model
The model’s performance is assessed using the following metrics:

Confusion Matrix:

Provides the number of true positives, true negatives, false positives, and false negatives.
Helps visualize classification performance.
Classification Report:

Includes precision, recall, F1-score, and support for each class (churn and non-churn).
Precision measures the proportion of positive predictions that are correct.
Recall measures the proportion of actual positives that are correctly predicted.
ROC AUC Score:

Measures the model’s ability to distinguish between churn and non-churn customers.
A score closer to 1 indicates better performance.
10. Analyzing Feature Importance
Feature importance is derived from the coefficients of the LogisticRegression model:

Positive coefficients indicate that an increase in the feature value increases the likelihood of churn.
Negative coefficients indicate that an increase in the feature value decreases the likelihood of churn.
For example:

ExitRates with a high positive coefficient suggests that customers who frequently exit pages are more likely to churn.
PageValues with a high negative coefficient suggests that customers who find value in pages are less likely to churn.
Results and Insights
The evaluation metrics show that the model performs well, with a high ROC AUC score of 0.8857. Key insights include:

Behavioral metrics like ExitRates and PageValues are strong indicators of churn.
Certain customer types (VisitorType and TrafficType) influence churn probabilities significantly.
Tools and Techniques Used
Scikit-learn:

Pipeline: To streamline preprocessing and modeling.
LogisticRegression: For binary classification.
ColumnTransformer, StandardScaler, and OneHotEncoder: For data preprocessing.
Pandas: For exploratory data analysis and feature importance visualization.


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
