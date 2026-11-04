# Credit Card Fraud Detection

A Machine Learning-based application that detects potentially fraudulent credit card transactions using a **K-Nearest Neighbors (KNN) classifier**. The project focuses on identifying fraudulent transactions in a highly imbalanced dataset while maintaining a balance between precision and recall.

## Project Overview

Credit card fraud detection is a binary classification problem where the objective is to distinguish between:

- **Genuine Transactions (Class 0)**
- **Fraudulent Transactions (Class 1)**

This project includes data preprocessing, exploratory data analysis, model training, evaluation, threshold tuning, and an interactive Streamlit frontend for transaction prediction.

## Features

- Exploratory Data Analysis (EDA)
- Data preprocessing and feature scaling
- Handling imbalanced classification data
- Training and comparison of multiple ML algorithms
- KNN model selection
- Classification threshold optimization
- Interactive Streamlit frontend
- Fraud probability prediction

## Machine Learning Models

The following algorithms were evaluated:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Naive Bayes
- Decision Tree
- Support Vector Machine (SVM)

**Final Selected Model:** K-Nearest Neighbors (KNN)

KNN was selected based on its fraud-class precision, recall, and F1-score performance.

## Model Performance

Final test-set performance using a classification threshold of 0.40:

| Metric | Fraud Class |
|---|---:|
| Precision | 97% |
| Recall | 77% |
| F1-Score | 86% |

**Confusion Matrix:**

| | Predicted Genuine | Predicted Fraud |
|---|---:|---:|
| Actual Genuine | 56,654 | 2 |
| Actual Fraud | 21 | 69 |

The model correctly identified 69 out of 90 fraudulent transactions in the test set.

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Streamlit
- Jupyter Notebook

## Dataset Information

The dataset contains anonymized transaction features:

- `Time` – Time elapsed between transactions
- `V1` to `V28` – PCA-transformed features
- `Amount` – Transaction amount
- `Class` – Target variable (0 = Genuine, 1 = Fraud)

The dataset is highly imbalanced, making precision, recall, and F1-score more informative than accuracy alone.

## Project Structure

```text
CREDIT CARD FRAUD DETECTION/
│
├── app.py
├── credit_card_fraud_detection.ipynb
├── requirements.txt
└── README.md
```

## Installation and Setup

**1. Clone the repository**

```bash
git clone https://github.com/bitbxabhay/Machine-Learning-Project.git
```

**2. Navigate to the project directory**

```bash
cd Machine-Learning-Project
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Run the Streamlit application**

```bash
streamlit run app.py
```

## How It Works

1. Enter transaction details in the application.
2. Provide Time, Amount, and PCA features (V1–V28).
3. Click the Predict Transaction button.
4. The trained KNN model evaluates the transaction.
5. The application displays the prediction and fraud probability.

The classification threshold is set to **0.40**.

## Important Note

The V1–V28 features are PCA-transformed variables. They are anonymized and cannot be directly interpreted as ordinary transaction details. Meaningful predictions require inputs prepared using the same preprocessing as the training dataset.

The dataset and trained model files are not included in this repository.

## Future Improvements

- Improve fraud detection recall.
- Experiment with advanced ML algorithms such as XGBoost.
- Optimize model inference performance.
- Improve frontend usability and visualization.
- Explore real-time transaction monitoring.

## Author

**Abhay**

Engineering Student | Machine Learning Enthusiast

[GitHub Profile](https://github.com/bitbxabhay)

---

*This project was developed for learning and demonstrating practical Machine Learning concepts in fraud detection.*
