import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import warnings
import inspect
warnings.filterwarnings('ignore')

# Get the directory of the current script
BASE_DIR = Path(__file__).resolve().parent.parent

# Load model and scaler with absolute paths
model = joblib.load(BASE_DIR / 'models' / 'lr_model.pkl')
scaler = joblib.load(BASE_DIR / 'models' / 'scaler.pkl')
X_train = joblib.load(BASE_DIR / 'models' / 'X_train.pkl')
X_test = joblib.load(BASE_DIR / 'models' / 'X_test.pkl')

# Get the feature names from training data
feature_names = X_train.columns.tolist()

# Load data for feature reference
df = pd.read_csv(BASE_DIR / 'data' / 'Telco-Customer-Churn.csv')
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(subset=['TotalCharges'], inplace=True)
df.drop('customerID', axis=1, inplace=True)

# Encode binary categoricals
binary_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
le = LabelEncoder()
for col in binary_cols:
    df[col] = le.fit_transform(df[col])

# One-hot encode multi-category
categorical_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']
df = pd.get_dummies(df, columns=categorical_cols)

# Create SHAP explainer once (outside button)
explainer = shap.LinearExplainer(model, X_train)

# Streamlit app
st.title("Customer Churn Prediction")
st.write("Predict churn probability with Logistic Regression (ROC-AUC: 0.730) and view SHAP explanations.")

# Input form
st.header("Customer Input")
tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.slider("Monthly Charges ($)", 0.0, 120.0, 50.0)
total_charges = st.slider("Total Charges ($)", 0.0, 10000.0, 600.0)
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
gender = st.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.selectbox("Partner", ["No", "Yes"])
dependents = st.selectbox("Dependents", ["No", "Yes"])
phone_service = st.selectbox("Phone Service", ["No", "Yes"])
multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

# Prepare input - create a dataframe with all features initialized to 0
input_data = pd.DataFrame(0, index=[0], columns=feature_names)

# Set numerical features
input_data['tenure'] = tenure
input_data['MonthlyCharges'] = monthly_charges
input_data['TotalCharges'] = total_charges

# Set binary features
input_data['gender'] = 1 if gender == "Male" else 0
input_data['SeniorCitizen'] = 1 if senior_citizen == "Yes" else 0
input_data['Partner'] = 1 if partner == "Yes" else 0
input_data['Dependents'] = 1 if dependents == "Yes" else 0
input_data['PhoneService'] = 1 if phone_service == "Yes" else 0
input_data['PaperlessBilling'] = 1 if paperless_billing == "Yes" else 0

# Set categorical features (one-hot encoded)
categorical_mappings = {
    'MultipleLines': multiple_lines,
    'InternetService': internet_service,
    'OnlineSecurity': online_security,
    'OnlineBackup': online_backup,
    'DeviceProtection': device_protection,
    'TechSupport': tech_support,
    'StreamingTV': streaming_tv,
    'StreamingMovies': streaming_movies,
    'Contract': contract,
    'PaymentMethod': payment_method
}

for feature_prefix, value in categorical_mappings.items():
    col_name = f'{feature_prefix}_{value}'
    if col_name in input_data.columns:
        input_data[col_name] = 1

# Scale numerical features
numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
input_data[numerical_cols] = scaler.transform(input_data[numerical_cols])

# Predict
if st.button("Predict"):
    pred = model.predict_proba(input_data)[:, 1][0]
    st.write(f"### Churn Probability: {pred:.2%}")
    if pred > 0.5:
        st.error("Prediction: Churn")
    else:
        st.success("Prediction: No Churn")

    # SHAP explanation
    shap_values = explainer.shap_values(input_data)
    if shap_values.ndim == 1:
        shap_values = shap_values.reshape(1, -1)
    
    st.write("---")
    st.write("### SHAP Explanation for This Prediction")
    st.write("Top features influencing this prediction:")
    
    # Get feature contributions
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'SHAP Value': shap_values[0].astype(float),
        'Feature Value': input_data.iloc[0].values.astype(float)
    })
    feature_importance['Abs SHAP'] = np.abs(feature_importance['SHAP Value'])
    feature_importance = feature_importance.sort_values('Abs SHAP', ascending=False).head(10)
    
    # Create a waterfall-style bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['red' if x > 0 else 'blue' for x in feature_importance['SHAP Value']]
    ax.barh(feature_importance['Feature'], feature_importance['SHAP Value'], color=colors)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('SHAP Value (impact on prediction)', fontsize=12)
    ax.set_title('Top 10 Features Impact on Prediction\n(Red = Increases Churn, Blue = Decreases Churn)', fontsize=12)
    plt.tight_layout()
    st.pyplot(fig)
    
    # Show the table
    st.write("#### Feature Details:")
    display_df = feature_importance[['Feature', 'SHAP Value', 'Feature Value']].copy()
    display_df['SHAP Value'] = display_df['SHAP Value'].apply(lambda x: f"{x:.4f}")
    display_df['Feature Value'] = display_df['Feature Value'].apply(lambda x: f"{x:.4f}")
    
    # Use width parameter if available, otherwise fall back to use_container_width
    try:
        st.dataframe(display_df.reset_index(drop=True), width='stretch')
    except TypeError:
        st.dataframe(display_df.reset_index(drop=True), use_container_width=True)

# Display feature importance
st.write("---")
st.header("Global Feature Importance (Logistic Regression)")
st.write("Average impact of each feature across all predictions:")
shap_values_global = explainer.shap_values(X_test)
if shap_values_global.ndim == 1:
    shap_values_global = shap_values_global.reshape(-1, 1)

fig2, ax2 = plt.subplots(figsize=(10, 8))
shap.summary_plot(shap_values_global, X_test, plot_type='bar', show=False, max_display=15)
plt.title('SHAP Feature Importance (Global - Top 15 Features)', fontsize=14)
plt.tight_layout()
st.pyplot(fig2)