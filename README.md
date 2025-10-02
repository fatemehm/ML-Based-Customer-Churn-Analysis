# 📊 Customer Churn Prediction

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.37+-red)
![License](https://img.shields.io/badge/License-MIT-green)

## 🎯 Overview
This project predicts customer churn using the Telco-Customer-Churn dataset. It features comprehensive exploratory data analysis (EDA), feature engineering, multiple machine learning models (Logistic Regression, Random Forest, XGBoost), SHAP explainability, and an interactive Streamlit web application for real-time predictions.

## 🚀 Live Demo
**Try the app:** [Customer Churn Prediction App](https://ml-based-customer-churn-analysis-drgzetudpaiinsd9hpsa7k.streamlit.app/)

## ✨ Features

- **Interactive Web App**: Input customer details and get real-time churn predictions
- **SHAP Explanations**: Understand which features drive each prediction with visual explanations
- **Model Comparison**: Evaluated Logistic Regression, Random Forest, and XGBoost
- **Feature Importance**: Both global and local explanations for model decisions
- **User-Friendly Interface**: Clean, intuitive design with real-time probability scores

## 📈 Results

- **Churn Rate**: ~26% of customers
- **Best Model**: Logistic Regression (ROC-AUC: 0.74)
- **Top Features Influencing Churn**:
  - TotalCharges
  - Tenure
  - MonthlyCharges
  - Contract_Month-to-month
  - TechSupport_No
  - InternetService_Fiber optic
  - PaymentMethod_Electronic check

## 📊 Model Performance

| Model               | ROC-AUC |
|---------------------|---------|
| Logistic Regression | 0.74    |
| Random Forest       | 0.69    |
| XGBoost             | 0.70    |

## 🛠️ Technologies Used

- **Languages**: Python 3.11
- **Data Processing**: pandas, numpy
- **Machine Learning**: scikit-learn, xgboost
- **Explainability**: SHAP
- **Visualization**: matplotlib, seaborn
- **Web Framework**: Streamlit
- **Deployment**: Streamlit Community Cloud
- **Version Control**: Git, GitHub

## 📁 Project Structure

```plaintext
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
```

## 🚀 Setup & Installation

### Prerequisites
- Python 3.11 or higher
- pip package manager

### 1. Clone the Repository
```bash
git clone https://github.com/fatemehm/ML-Based-Customer-Churn-Analysis.git
cd ML-Based-Customer-Churn-Analysis
```

### 2. Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Locally
```bash
streamlit run app/app.py
```

Access the app at `http://localhost:8501`

## 📓 Jupyter Notebook

To explore the analysis and model development:

```bash
jupyter notebook notebooks/churn_prediction.ipynb
```

## 🎥 Demo Video

Watch the full demonstration of the Customer Churn Prediction app in action:

[![Customer Churn Prediction Demo](https://github.com/fatemehm/ML-Based-Customer-Churn-Analysis/blob/main/Demo_video.mkv)]

*The video showcases:*
- How to input customer information
- Real-time churn prediction results
- SHAP explanations and feature importance visualization
- Interactive model insights


## 📝 License

This project is open source and available under the [MIT License](LICENSE).


