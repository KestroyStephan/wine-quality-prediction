# 🍷 Wine Quality Prediction App

An interactive **Machine Learning Web Application** that predicts whether a wine is **Good** or **Bad** based on its chemical properties.  
Built with **Streamlit**, trained using the **Wine Quality Dataset**, and deployed on **Streamlit Cloud**.

---

## 📌 Project Overview
This project demonstrates the complete **Machine Learning Deployment Pipeline**:
1. **Data Exploration & Preprocessing**  
2. **Model Training & Evaluation** (Random Forest, Logistic Regression)  
3. **Model Selection & Saving** with `joblib`  
4. **Interactive Web App Development** using Streamlit  
5. **Deployment to the Cloud** for public access  

The model classifies wines into:
- **Good (1)** → High-quality wine
- **Bad (0)** → Low-quality wine

---

## 📂 Dataset
- **Name:** Wine Quality Dataset (`WineQT.csv`)  
- **Source:** [UCI Machine Learning Repository](https://www.kaggle.com/datasets/yasserh/wine-quality-dataset/data)  
- **Description:** Contains physicochemical properties (acidity, sugar, pH, etc.) and quality ratings for red wine samples.  

**Features include:**
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
- `quality` → converted into binary `good` (1) and `bad` (0) for classification.

---

## 📁 Project Structure
Wine-Quality-Prediction/
├── app.py # Streamlit application
├── model.pkl # Trained ML model
├── scaler.pkl # Feature scaler
├── data/
│ └── WineQT.csv # Dataset
├── notebooks/
│ └── wine_model_training.ipynb # EDA, preprocessing, model training
├── requirements.txt # Project dependencies
└── README.md # Project documentation

yaml

## ⚙️ Installation & Usage

### 1️⃣ Clone the repository
```bash
git clone https://github.com/<your-username>/wine-quality-prediction.git
cd wine-quality-prediction
2️⃣ Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
3️⃣ Run the application locally
bash
Copy
Edit
streamlit run app.py
Open your browser and go to: http://localhost:8501

🌐 Deployment
The application is deployed using Streamlit Cloud:

Push all files to a public GitHub repository

Connect the repository to Streamlit Cloud

Set app.py as the entry point

Deploy and share the public URL

📊 Application Features
Data Exploration: Dataset overview, shape, columns, missing values, filtering

Visualizations: Interactive plots (distribution, boxplots, heatmaps, scatter plots)

Prediction: User input sliders for wine properties, real-time predictions, probability display

Model Performance: Accuracy score, confusion matrix, classification report, ROC curve

About Page: Project info, dataset source, technologies used

🧠 Technologies Used
Python

Pandas, NumPy for data manipulation

Seaborn, Matplotlib, Plotly for visualizations

Scikit-learn for ML model training

Streamlit for web app development

Joblib for model persistence

