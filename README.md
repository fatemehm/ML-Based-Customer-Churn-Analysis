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
- **Best Model**: Logistic Regression (ROC-AUC: 0.730)
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

[![Customer Churn Prediction Demo](https://img.shields.io/badge/Watch-Demo%20Video-red?style=for-the-badge&logo=youtube)](YOUR_VIDEO_LINK_HERE)

*The video showcases:*
- How to input customer information
- Real-time churn prediction results
- SHAP explanations and feature importance visualization
- Interactive model insights

## 🔍 Key Insights

1. **Contract Type Matters**: Month-to-month contracts have significantly higher churn rates
2. **Support is Critical**: Customers without tech support are more likely to churn
3. **Payment Method Impact**: Electronic check payments correlate with higher churn
4. **Fiber Optic Paradox**: Despite being a premium service, fiber optic customers show higher churn
5. **Tenure Effect**: Longer customer relationships strongly predict retention

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 👤 Author

**Fatemeh M**
- GitHub: [@fatemehm](https://github.com/fatemehm)

## 🙏 Acknowledgments

- Dataset: [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) from Kaggle
- SHAP library for model interpretability
- Streamlit for the amazing web framework

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

⭐ If you found this project helpful, please consider giving it a star!