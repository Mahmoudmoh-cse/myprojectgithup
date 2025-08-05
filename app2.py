import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title=" Health Condition Predictor", layout="wide")
st.markdown("<h1 style='text-align: center;'> Unified Health Condition Predictor</h1><hr>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title(" Settings")
    disease = st.selectbox(" Select Disease", ["Breast Cancer", "Stroke", "Heart Disease", "Diabetes"])
    model_choice = st.selectbox(" Choose ML Model", ["KNN", "Random Forest", "Logistic Regression", "Naive Bayes", "Decision Tree"])

# Load datasets
def load_breast_cancer_data():
    try:
        df = pd.read_csv("breast_cancer.csv").fillna(0)
        df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})
        return df
    except Exception as e:
        st.error(f"Error loading breast cancer data: {e}")
        return pd.DataFrame()

def load_stroke_data():
    try:
        df = pd.read_csv("healthcare-dataset-stroke-data.csv")
        df['bmi'] = df['bmi'].replace("N/A", np.nan).astype(float)
        df['bmi'] = df['bmi'].fillna(df['bmi'].mean())
        df['gender'] = df['gender'].map({'Male': 1, 'Female': 0, 'Other': 0})
        df['ever_married'] = df['ever_married'].map({'Yes': 1, 'No': 0})
        df['Residence_type'] = df['Residence_type'].map({'Urban': 1, 'Rural': 0})
        df = pd.get_dummies(df, columns=['work_type', 'smoking_status'], drop_first=True)
        df = df.dropna()
        df = df.sample(frac=0.3, random_state=1)  # optional sampling
        return df
    except Exception as e:
        st.error(f"Error loading stroke data: {e}")
        return pd.DataFrame()

def load_heart_data():
    try:
        return pd.read_csv("heart.csv").fillna(0)
    except Exception as e:
        st.error(f"Error loading heart disease data: {e}")
        return pd.DataFrame()

def load_diabetes_data():
    try:
        return pd.read_csv("diabetes.csv").fillna(0)
    except Exception as e:
        st.error(f"Error loading diabetes data: {e}")
        return pd.DataFrame()

# Models
models = {
    "KNN": KNeighborsClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier()
}
clf = models[model_choice]

# Input forms
def input_form_breast():
    st.subheader(" Enter Breast Cancer Details")
    inputs = {
        "radius_mean": st.number_input("Radius Mean", 0.0, 50.0, 14.0),
        "texture_mean": st.number_input("Texture Mean", 0.0, 50.0, 20.0),
        "perimeter_mean": st.number_input("Perimeter Mean", 0.0, 200.0, 90.0),
        "area_mean": st.number_input("Area Mean", 0.0, 3000.0, 500.0),
        "smoothness_mean": st.number_input("Smoothness Mean", 0.0, 1.0, 0.1)
    }
    return pd.DataFrame([inputs])

def input_form_stroke():
    st.subheader("📝 Enter Stroke Details")
    inputs = {
        'gender': st.selectbox("Gender (1=Male, 0=Female)", [1, 0]),
        'age': st.slider("Age", 0, 100, 45),
        'hypertension': st.selectbox("Hypertension", [0, 1]),
        'heart_disease': st.selectbox("Heart Disease", [0, 1]),
        'ever_married': st.selectbox("Ever Married", [1, 0]),
        'avg_glucose_level': st.slider("Avg Glucose Level", 50, 300, 120),
        'bmi': st.slider("BMI", 10, 60, 28),
        'Residence_type': st.selectbox("Residence Type", [1, 0]),
        'work_type_Govt_job': st.selectbox("Work Type: Govt Job", [0, 1]),
        'work_type_Never_worked': st.selectbox("Work Type: Never Worked", [0, 1]),
        'work_type_Private': st.selectbox("Work Type: Private", [0, 1]),
        'work_type_Self_employed': st.selectbox("Work Type: Self Employed", [0, 1]),
        'smoking_status_formerly smoked': st.selectbox("Smoking: Formerly smoked", [0, 1]),
        'smoking_status_never smoked': st.selectbox("Smoking: Never smoked", [0, 1]),
        'smoking_status_smokes': st.selectbox("Smoking: Smokes", [0, 1])
    }
    return pd.DataFrame([inputs])

def input_form_heart():
    st.subheader(" Enter Heart Disease Details")
    inputs = {
        'age': st.slider("Age", 20, 100, 50),
        'sex': st.selectbox("Sex (1=Male, 0=Female)", [1, 0]),
        'cp': st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3]),
        'trestbps': st.slider("Resting BP", 90, 200, 120),
        'chol': st.slider("Cholesterol", 100, 600, 250),
        'fbs': st.selectbox("Fasting Blood Sugar > 120 (1=yes)", [1, 0]),
        'restecg': st.selectbox("Rest ECG (0-2)", [0, 1, 2]),
        'thalach': st.slider("Max Heart Rate", 60, 220, 150),
        'exang': st.selectbox("Exercise Induced Angina (1=yes)", [1, 0]),
        'oldpeak': st.slider("Oldpeak", 0.0, 6.0, 1.0),
        'slope': st.selectbox("Slope (0-2)", [0, 1, 2]),
        'ca': st.selectbox("# of vessels colored (0-3)", [0, 1, 2, 3]),
        'thal': st.selectbox("Thal (1,2,3)", [1, 2, 3])
    }
    return pd.DataFrame([inputs])

def input_form_diabetes():
    st.subheader(" Enter Diabetes Details")
    inputs = {
        'Pregnancies': st.slider("Pregnancies", 0, 20, 2),
        'Glucose': st.slider("Glucose", 0, 200, 120),
        'BloodPressure': st.slider("Blood Pressure", 0, 150, 70),
        'SkinThickness': st.slider("Skin Thickness", 0, 100, 20),
        'Insulin': st.slider("Insulin", 0, 1000, 80),
        'BMI': st.slider("BMI", 0.0, 70.0, 28.0),
        'DiabetesPedigreeFunction': st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5),
        'Age': st.slider("Age", 10, 100, 33)
    }
    return pd.DataFrame([inputs])

# Prepare data
df = pd.DataFrame()
X, y, user_input = None, None, None

if disease == "Breast Cancer":
    df = load_breast_cancer_data()
    if not df.empty:
        features = ["radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean"]
        X = df[features]
        y = df["diagnosis"]
        user_input = input_form_breast().reindex(columns=X.columns, fill_value=0)

elif disease == "Stroke":
    df = load_stroke_data()
    if not df.empty:
        X = df.drop(columns=["id", "stroke"])
        y = df["stroke"]
        user_input = input_form_stroke().reindex(columns=X.columns, fill_value=0)

elif disease == "Heart Disease":
    df = load_heart_data()
    if not df.empty:
        X = df.drop(columns=["target"])
        y = df["target"]
        user_input = input_form_heart().reindex(columns=X.columns, fill_value=0)

elif disease == "Diabetes":
    df = load_diabetes_data()
    if not df.empty:
        X = df.drop(columns=["Outcome"])
        y = df["Outcome"]
        user_input = input_form_diabetes().reindex(columns=X.columns, fill_value=0)

# Train and evaluate
if X is not None and y is not None and not X.empty and not y.empty:
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', clf)
    ])
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        st.markdown("###  Model Evaluation Results")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.code(classification_report(y_test, y_pred), language="text")
        with col2:
            fig, ax = plt.subplots()
            sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap="Blues", ax=ax)
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Training failed: {e}")
else:
    st.warning(" Dataset is empty or invalid. Please verify the selected disease and data files.")

# Prediction
st.markdown("###  Prediction on Your Input")
if user_input is not None and not user_input.empty:
    try:
        prediction = pipeline.predict(user_input)[0]
        st.success(f"Prediction Result: {'Positive' if prediction == 1 else 'Negative'}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")