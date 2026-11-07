# Chronic Kidney Disease (CKD) Detection

A Machine Learning project that predicts the possibility of Chronic Kidney Disease (CKD) using patient health parameters and Logistic Regression.

## About the Project

Chronic Kidney Disease is a medical condition that affects kidney function over time. Early identification of potential CKD cases can help support timely medical evaluation.

This project uses Machine Learning to classify patient data into two categories:

- **CKD:** Chronic Kidney Disease detected by the model.
- **Not CKD:** Chronic Kidney Disease not detected by the model.

The project also includes a web interface where users can enter patient health parameters and obtain a model prediction.

## Features

- Data cleaning and preprocessing
- Handling missing values
- Exploratory Data Analysis (EDA)
- Categorical feature encoding
- Feature scaling (where applicable)
- Feature importance analysis
- Logistic Regression classification
- Flask-based web application
- Real-time prediction through a user-friendly interface

## Dataset

The project uses the Chronic Kidney Disease dataset obtained from Kaggle.

The dataset contains 24 input features:

| Feature | Description |
|---|---|
| age | Age |
| bp | Blood Pressure |
| sg | Specific Gravity |
| al | Albumin |
| su | Sugar |
| rbc | Red Blood Cells |
| pc | Pus Cell |
| pcc | Pus Cell Clumps |
| ba | Bacteria |
| bgr | Blood Glucose Random |
| bu | Blood Urea |
| sc | Serum Creatinine |
| sod | Sodium |
| pot | Potassium |
| hemo | Hemoglobin |
| pcv | Packed Cell Volume |
| wc | White Blood Cell Count |
| rc | Red Blood Cell Count |
| htn | Hypertension |
| dm | Diabetes Mellitus |
| cad | Coronary Artery Disease |
| appet | Appetite |
| pe | Pedal Edema |
| ane | Anemia |

**Target Variable:** `classification`

## Machine Learning Model

**Algorithm:** Logistic Regression

Logistic Regression was selected as the classification algorithm for predicting CKD.

The model is trained using the processed dataset and integrated into the Flask application for predictions.

## Tech Stack

- **Programming Language:** Python
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn
- **Model:** Logistic Regression
- **Backend:** Flask
- **Frontend:** HTML, CSS, JavaScript
- **Development Environment:** Jupyter Notebook, VS Code

## Project Structure

```text
CHRONIC KIDNEY DISEASE DETECTION/
│
├── app.py
├── model/
├── templates/
│   └── index.html
├── static/
│   ├── css/
│   ├── js/
│   └── images/
├── notebooks/
│   └── CKD_Detection.ipynb
├── requirements.txt
└── README.md
```

## Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/bitbxabhay/Machine-Learning-Projects.git
```

### 2. Navigate to the project directory

```bash
cd Machine-Learning-Projects
cd "CHRONIC KIDNEY DISEASE DETECTION"
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
python app.py
```

### 5. Open in browser

```text
http://127.0.0.1:5000/
```

## Model Evaluation

Model performance is evaluated using appropriate classification metrics:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

*Add your actual evaluation results here after final model testing.*

## Future Improvements

- Experiment with additional classification algorithms.
- Improve model performance through hyperparameter tuning.
- Enhance the web interface and user experience.
- Deploy the application online.

## Disclaimer

This project is developed for educational and machine learning purposes only. Its predictions are not medical diagnoses and should not be used as a substitute for professional medical advice, diagnosis, or treatment.

## Author

**Abhay**

GitHub: [@bitbxabhay](https://github.com/bitbxabhay)

---

If you find this project useful, consider giving the repository a ⭐.