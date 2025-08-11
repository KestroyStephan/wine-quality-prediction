# Imports & Page Setup
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import os
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Set Streamlit page configuration
st.set_page_config(page_title="Wine Quality Prediction App", layout="wide")

# Load saved model and scaler
MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"
DATA_PATH = "data/WineQT.csv"

# Check if files exist before loading
if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
else:
    st.error("Model or scaler file not found. Please run wine_model_training.py first.")
    st.stop()

# Load dataset for exploration
if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
else:
    st.error("Dataset file not found.")
    st.stop()

# Create 'good' column: 1 for quality >= 7, else 0
df['good'] = df['quality'].apply(lambda q: 1 if q >= 7 else 0)

st.title(" Wine Quality Prediction App")
st.write("This app predicts whether a wine is **Good** or **Bad** based on its chemical properties.")

# Sidebar menu
menu = st.sidebar.radio(
    "Navigation",
    ["Home", "Data Exploration", "Visualizations", "Prediction", "Model Performance", "About"]
)

#  Home
if menu == "Home":
    st.subheader(" Project Overview")
    st.write("""
    This application uses the **Wine Quality Dataset** to predict whether a wine is good or bad
    based on its chemical properties.

    **Features in the dataset include:**
    - Fixed acidity
    - Volatile acidity
    - Citric acid
    - Residual sugar
    - Chlorides
    - Free sulfur dioxide
    - Total sulfur dioxide
    - Density
    - pH
    - Sulphates
    - Alcohol

    **Target variable:**
    - Quality score (converted into Good = 1, Bad = 0)
    """)

#  Data Exploration
elif menu == "Data Exploration":
    st.subheader(" Dataset Overview")
    st.write("Shape of dataset:", df.shape)
    st.dataframe(df.head())

    st.subheader(" Data Types")
    st.write(df.dtypes)

    st.subheader(" Missing Values")
    st.write(df.isnull().sum())

    st.subheader(" Interactive Filtering")
    filter_col = st.selectbox("Select column to filter", options=df.columns)
    unique_vals = df[filter_col].unique().tolist()
    selected_vals = st.multiselect("Select values", options=unique_vals, default=unique_vals)
    filtered_df = df[df[filter_col].isin(selected_vals)]
    st.write("Filtered rows:", filtered_df.shape[0])
    st.dataframe(filtered_df)

#  Visualizations Page
elif menu == "Visualizations":
    st.subheader(" Visualizations")

    # Distribution of Good vs Bad wines
    st.markdown("### 1 Distribution of Good vs Bad Wines")
    fig1 = px.histogram(df, x="good", color="good", title="Good (1) vs Bad (0) Wine Distribution")
    st.plotly_chart(fig1, use_container_width=True)

    # Alcohol content vs Wine Quality
    st.markdown("### 2 Alcohol Content vs Wine Quality")
    fig2 = px.box(df, x="good", y="alcohol", color="good", points="all",
                  title="Alcohol Content by Wine Quality")
    st.plotly_chart(fig2, use_container_width=True)

    # Correlation heatmap
    st.markdown("### 3 Correlation Heatmap")
    corr_matrix = df.corr()
    fig3 = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                     title="Feature Correlation Heatmap",
                     color_continuous_scale="RdBu_r")
    st.plotly_chart(fig3, use_container_width=True)

    # Alcohol vs Volatile Acidity scatter plot
    st.markdown("### 4 Alcohol vs Volatile Acidity (Colored by Quality)")
    fig4 = px.scatter(df, x="alcohol", y="volatile acidity", color=df["good"].astype(str),
                      title="Alcohol vs Volatile Acidity")
    st.plotly_chart(fig4, use_container_width=True)

#  Prediction Page
elif menu == "Prediction":
    st.subheader(" Predict Wine Quality (Good or Bad)")

    st.write("Adjust the values below to see if the model predicts the wine as **Good** or **Bad**.")

    # Create input sliders for each feature
    fixed_acidity = st.number_input("Fixed Acidity", min_value=0.0, max_value=20.0, value=8.0, step=0.1)
    volatile_acidity = st.number_input("Volatile Acidity", min_value=0.0, max_value=2.0, value=0.5, step=0.01)
    citric_acid = st.number_input("Citric Acid", min_value=0.0, max_value=2.0, value=0.3, step=0.01)
    residual_sugar = st.number_input("Residual Sugar", min_value=0.0, max_value=20.0, value=2.5, step=0.1)
    chlorides = st.number_input("Chlorides", min_value=0.0, max_value=1.0, value=0.08, step=0.001)
    free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", min_value=0.0, max_value=100.0, value=15.0, step=1.0)
    total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", min_value=0.0, max_value=300.0, value=46.0, step=1.0)
    density = st.number_input("Density", min_value=0.98, max_value=1.05, value=0.996, step=0.0001)
    pH = st.number_input("pH", min_value=2.0, max_value=4.0, value=3.3, step=0.01)
    sulphates = st.number_input("Sulphates", min_value=0.0, max_value=2.0, value=0.65, step=0.01)
    alcohol = st.number_input("Alcohol", min_value=8.0, max_value=15.0, value=10.0, step=0.1)

    if st.button("Predict Quality"):
        # Prepare input data with correct columns (without 'quality' and 'good')
        X = df.drop(['quality', 'good'], axis=1)
        input_data = pd.DataFrame([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                                    chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
                                    density, pH, sulphates, alcohol]], columns=X.columns)

        # Scale the input
        input_scaled = scaler.transform(input_data)

        # Predict
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]

        # Show result
        if prediction == 1:
            st.success(f"This wine is predicted to be **GOOD** ({probability*100:.1f}% probability).")
        else:
            st.error(f"This wine is predicted to be **BAD** ({(1-probability)*100:.1f}% probability).")

#  Model Performance Page
elif menu == "Model Performance":
    st.subheader(" Model Performance")

    # Prepare data for evaluation
    X_perf = df.drop(['quality', 'good'], axis=1)
    y_perf = df['good']

    # Scale features
    X_scaled_perf = scaler.transform(X_perf)

    # Predictions
    y_pred_perf = model.predict(X_scaled_perf)
    y_proba_perf = model.predict_proba(X_scaled_perf)[:, 1]

    # Accuracy
    acc = accuracy_score(y_perf, y_pred_perf)
    st.write(f"**Overall Accuracy:** {acc:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_perf, y_pred_perf)
    st.markdown("**Confusion Matrix:**")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Bad', 'Good'], yticklabels=['Bad', 'Good'], ax=ax)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    st.pyplot(fig)

    # Classification Report
    st.markdown("**Classification Report:**")
    report_df = pd.DataFrame(classification_report(y_perf, y_pred_perf, output_dict=True)).transpose()
    st.dataframe(report_df)

    # ROC Curve
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_perf, y_proba_perf)
    roc_auc = auc(fpr, tpr)

    st.markdown(f"**ROC AUC:** {roc_auc:.4f}")
    fig2, ax2 = plt.subplots()
    ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    ax2.plot([0, 1], [0, 1], 'k--')
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title("ROC Curve")
    ax2.legend(loc="lower right")
    st.pyplot(fig2)

#  About Page
elif menu == "About":
    st.subheader(" About This Project")
    st.write("""
    This project was developed as part of a Machine Learning Model Deployment assignment.

    **Objective:**
    - Predict whether a wine is **Good** or **Bad** based on its chemical properties.

    **Dataset:**
    - Wine Quality Dataset (provided as `WineQT.csv`).
    - Original dataset source: [Wine Quality Dataset](https://www.kaggle.com/datasets/yasserh/wine-quality-dataset/data).

    **Workflow:**
    1. Data Exploration & Preprocessing
    2. Feature Scaling
    3. Model Training (Random Forest & Logistic Regression)
    4. Model Evaluation & Selection
    5. Model Saving with `joblib`
    6. Building this Streamlit web application
    7. Deployment to Streamlit Cloud

    **Technologies Used:**
    - Python
    - Pandas, NumPy, Seaborn, Matplotlib, Plotly
    - Scikit-learn
    - Streamlit
    - Joblib

    **Author:**
    - Kestroy Stephan
    """)

st.sidebar.info("Machine Learning Model Deployment with Streamlit")
