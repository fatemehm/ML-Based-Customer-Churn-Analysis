# Customer Churn Prediction

## Overview
This project predicts customer churn using the Telco-Customer-Churn dataset, featuring EDA, feature engineering, modeling (Logistic Regression, Random Forest, XGBoost), SHAP explainability for Logistic Regression (ROC-AUC: 0.730), and a Streamlit app for deployment.

## Live Demo
**Try the app:** [Customer Churn Prediction App](https://ml-based-customer-churn-analysis-drgzetudpaiinsd9hpsa7k.streamlit.app/)

## Setup

### 1. Clone the repo:

git clone https://github.com/fatemehm/ML-Based-Customer-Churn-Analysis
cd ML-Based-Churn-Analysis

### 2. Create virtual environment and install dependencies:

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

### 3. Run locally:
bashstreamlit run app/app.py
Access at http://localhost:8501

📁 Project Structure
ML-Based-Churn-Analysis/
├── app/
│   └── app.py                      # Streamlit web application
├── notebooks/
│   └── churn_prediction.ipynb      # Analysis and modeling notebook
├── diagrams/                        # Visualizations (EDA, SHAP plots)
├── models/                          # Saved models and scalers
│   ├── lr_model.pkl
│   ├── scaler.pkl
│   ├── X_train.pkl
│   └── X_test.pkl
├── data/
│   └── Telco-Customer-Churn.csv    # Dataset
├── requirements.txt                 # Python dependencies
├── runtime.txt                      # Python version specification
└── README.md                        # Project documentation

Results

Churn rate: ~26%
Best model: Logistic Regression (ROC-AUC: 0.730)
Top features influencing churn:

TotalCharges
Tenure
MonthlyCharges
Contract_Month-to-month
TechSupport_No
InternetService_Fiber optic
PaymentMethod_Electronic check



Features

Interactive Web App: Input customer details and get real-time churn predictions
SHAP Explanations: Understand which features drive each prediction
Model Comparison: Logistic Regression, Random Forest, and XGBoost evaluated
Feature Importance: Global and local explanations for model decisions

🛠️ Technologies Used

Python: pandas, numpy, scikit-learn, xgboost
Visualization: matplotlib, seaborn, SHAP
Web Framework: Streamlit
Deployment: Streamlit Community Cloud
Version Control: Git, GitHub

Model Performance
--------------------------------
|Model               | ROC-AUC |
|--------------------|---------|
|Logistic Regression |   0.74  |
|--------------------|---------|
|Random Forest       |   0.69  |
|--------------------|---------|
|XGBoost             |   0.70  |
--------------------------------

Deployment

Local: http://localhost:8501
Cloud: https://ml-based-customer-churn-analysis-drgzetudpaiinsd9hpsa7k.streamlit.app/

Demo Video: https://github.com/fatemehm/ML-Based-Customer-Churn-Analysis/blob/main/Demo_video.mkv

License
This project is open source and available under the MIT License.
