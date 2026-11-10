# 📡 Telecom Customer Churn Prediction Using Machine Learning

## 📌 Project Overview

Telecom Customer Churn Prediction is a Machine Learning classification project that predicts whether a customer is likely to leave a telecom company or continue using its services.

Customer churn is a major challenge for telecom companies because losing existing customers directly impacts business revenue. This project uses customer demographics, account information, subscribed services, and billing details to identify customers who are more likely to churn.

The primary goal is to help telecom companies understand customer behavior and take proactive steps to improve customer retention.

## 🎯 Project Objectives

- Predict whether a customer is likely to churn or stay.
- Identify important factors influencing customer churn.
- Compare multiple Machine Learning classification algorithms.
- Improve model performance through hyperparameter tuning.
- Analyze model performance using ROC-AUC and classification metrics.
- Optimize the classification threshold to improve churn detection.

## 📊 Dataset Information

The project uses a Telecom Customer Churn dataset containing customer demographics, subscribed services, contract details, and billing information.

The dataset includes features such as:

- **Customer Information:** Gender, Senior Citizen, Partner, Dependents.
- **Account Information:** Tenure, Contract.
- **Service Information:** Phone Service, Multiple Lines, Internet Service, Online Security, Online Backup, Device Protection, Tech Support, Streaming TV, Streaming Movies.
- **Billing Information:** Paperless Billing, Payment Method, Monthly Charges, Total Charges.

**Target Variable:** Churn

- Yes – Customer has left the company.
- No – Customer continues using the service.

## ⚙️ Technologies & Libraries Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Jupyter Notebook

## 🔍 Project Workflow

### 1. Data Preprocessing
- Handled missing values.
- Removed unnecessary columns.
- Cleaned categorical and numerical data.
- Applied encoding techniques to categorical variables.
- Prepared data for Machine Learning models.

### 2. Exploratory Data Analysis (EDA)
- Analyzed customer demographics.
- Explored relationships between customer attributes and churn.
- Visualized churn distribution.
- Studied the impact of tenure, monthly charges, contracts, and payment methods on customer churn.

### 3. Model Building

Implemented and compared multiple Machine Learning classification algorithms:

- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier

### 4. Hyperparameter Tuning

Applied RandomizedSearchCV to optimize model hyperparameters and improve predictive performance.

### 5. Model Evaluation

Evaluated model performance using:

- Accuracy Score
- Precision
- Recall
- F1-Score
- Confusion Matrix
- ROC Curve
- ROC-AUC Score

### 6. ROC Analysis & Threshold Optimization

Performed ROC curve analysis to understand the model's ability to distinguish between churn and non-churn customers.

Applied classification threshold tuning to optimize the prediction results rather than relying only on the default threshold of 0.5.

### 7. Final Model Selection

Selected Gradient Boosting as the final model after model comparison, hyperparameter tuning, ROC analysis, and threshold optimization.

## 📈 Key Insights

Feature importance analysis identified several influential factors associated with customer churn:

- Tenure
- Internet Service (Fiber Optic)
- Payment Method (Electronic Check)
- Total Charges
- Contract Type
- Monthly Charges

These features provide useful insights into customer behavior and potential churn patterns.

## 💡 Business Applications

This project can help telecom companies:

- Identify customers at high risk of leaving.
- Develop targeted customer retention strategies.
- Offer personalized discounts and service plans.
- Improve customer satisfaction.
- Reduce customer acquisition costs by retaining existing customers.

## 🚀 Future Improvements

- Develop an interactive web application using Streamlit or Flask.
- Deploy the trained Machine Learning model.
- Build a customer churn prediction dashboard.
- Provide personalized customer retention recommendations.
- Experiment with advanced ensemble learning techniques.

## 👨‍💻 Author

**Abhay**

Machine Learning Enthusiast | Aspiring AI Engineer

This project was developed as part of my journey in learning Machine Learning and building practical AI/ML projects.

---

⭐ If you find this project interesting, feel free to explore the repository and share your feedback!