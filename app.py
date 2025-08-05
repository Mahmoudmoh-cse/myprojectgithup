# --- Import necessary libraries ---
import streamlit as st  # For creating web app interface
import pandas as pd  # For data manipulation
import numpy as np  # For numerical operations
from sklearn.preprocessing import StandardScaler  # For feature scaling
from sklearn.model_selection import train_test_split  # For splitting data
from sklearn.linear_model import LogisticRegression  # ML model
from sklearn.svm import SVC  # ML model
from sklearn.tree import DecisionTreeClassifier  # ML model
from sklearn.naive_bayes import GaussianNB  # ML model
from sklearn.neighbors import KNeighborsClassifier  # ML model
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report  # Evaluation metrics
import seaborn as sns  # For visualizations
import matplotlib.pyplot as plt  # For plotting

# --- Set Streamlit page configuration ---
st.set_page_config(page_title="Customer Churn Predictor", layout="wide")

# --- Load and preprocess data ---
@st.cache_data  # Caches the function to avoid reloading on every app interaction
def load_data():
    df = pd.read_csv("Telco-Customer-Churn.csv")  # Load dataset
    df = df[df["TotalCharges"].str.strip() != ""]  # Remove rows with blank TotalCharges
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"])  # Convert TotalCharges to numeric
    df.drop("customerID", axis=1, inplace=True)  # Drop customerID column (not useful for modeling)
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})  # Encode churn as binary

    X_raw = df.drop("Churn", axis=1)  # Features before encoding
    y = df["Churn"]  # Target variable
    X_encoded = pd.get_dummies(X_raw, drop_first=True)  # One-hot encode categorical variables
    return X_raw, X_encoded, y

# --- Load and split data ---
X_raw, X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)  # Split dataset into train/test

# --- Scale numeric features ---
numerical_cols = ["tenure", "MonthlyCharges", "TotalCharges"]  # Columns to scale
scaler = StandardScaler()  # Create scaler instance
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])  # Fit and scale training data
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])  # Scale test data using same scaler

# --- Initialize and train multiple ML models ---
models = {
    " Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
    "Naive Bayes": GaussianNB(),
    "SVM (RBF Kernel)": SVC(kernel='rbf', C=1.0, probability=True, random_state=42),
    " Decision Tree (Entropy)": DecisionTreeClassifier(criterion='gini', max_depth=10, random_state=42),
    "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=3)
}
for model in models.values():
    model.fit(X_train, y_train)  # Train each model on training data

# --- Define valid options for categorical input fields ---
categorical_input_options = {
    # Dictionary mapping each categorical field to its possible values
    "gender": ["Female", "Male"],
    "SeniorCitizen": [0, 1],
    "Partner": ["No", "Yes"],
    "Dependents": ["No", "Yes"],
    "PhoneService": ["No", "Yes"],
    "MultipleLines": ["No phone service", "No", "Yes"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["No internet service", "No", "Yes"],
    "OnlineBackup": ["No internet service", "No", "Yes"],
    "DeviceProtection": ["No internet service", "No", "Yes"],
    "TechSupport": ["No internet service", "No", "Yes"],
    "StreamingTV": ["No internet service", "No", "Yes"],
    "StreamingMovies": ["No internet service", "No", "Yes"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["No", "Yes"],
    "PaymentMethod": [
        "Electronic check", "Mailed check", 
        "Bank transfer (automatic)", "Credit card (automatic)"
    ]
}

# --- Page Title and Sidebar Controls ---
st.title(" Telecom Customer Churn Prediction")
st.sidebar.header(" Model & Customer Inputs")

# --- Model selection from sidebar ---
model_name = st.sidebar.selectbox("Choose a model:", list(models.keys()))
model = models[model_name]  # Select corresponding model

# --- Pre-fill example user profiles ---
st.sidebar.subheader(" Quick Start")
if st.sidebar.button(" Typical Staying Customer"):
    # Example input for likely-to-stay customer
    user_input_raw = {
        "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
        "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "DSL",
        "OnlineSecurity": "Yes", "OnlineBackup": "Yes", "DeviceProtection": "Yes",
        "TechSupport": "Yes", "StreamingTV": "No", "StreamingMovies": "No",
        "Contract": "Two year", "PaperlessBilling": "No", "PaymentMethod": "Mailed check",
        "tenure": 60, "MonthlyCharges": 30.0, "TotalCharges": 1800.0
    }
elif st.sidebar.button("Likely to Churn"):
    # Example input for likely-to-churn customer
    user_input_raw = {
        "gender": "Female", "SeniorCitizen": 1, "Partner": "No", "Dependents": "No",
        "PhoneService": "Yes", "MultipleLines": "Yes", "InternetService": "Fiber optic",
        "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No",
        "TechSupport": "No", "StreamingTV": "Yes", "StreamingMovies": "Yes",
        "Contract": "Month-to-month", "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
        "tenure": 2, "MonthlyCharges": 90.0, "TotalCharges": 180.0
    }
else:
    user_input_raw = {}  # Empty dict if no button pressed

# --- Sidebar input forms grouped by category ---
with st.sidebar.expander(" Customer Info", expanded=True):
    for feature in ["gender", "SeniorCitizen", "Partner", "Dependents"]:
        options = categorical_input_options[feature]
        user_input_raw[feature] = st.selectbox(
            feature, options, index=options.index(user_input_raw.get(feature, options[0]))
        )

with st.sidebar.expander(" Services Used", expanded=False):
    for feature in ["PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
                    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]:
        options = categorical_input_options[feature]
        user_input_raw[feature] = st.selectbox(
            feature, options, index=options.index(user_input_raw.get(feature, options[0]))
        )

with st.sidebar.expander(" Billing & Contract Info", expanded=False):
    for feature in ["Contract", "PaperlessBilling", "PaymentMethod"]:
        options = categorical_input_options[feature]
        user_input_raw[feature] = st.selectbox(
            feature, options, index=options.index(user_input_raw.get(feature, options[0]))
        )

    # Numeric inputs
    user_input_raw["tenure"] = st.number_input("Tenure (months)", 0, 100, value=user_input_raw.get("tenure", 1))
    user_input_raw["MonthlyCharges"] = st.number_input("Monthly Charges", 0.0, 200.0, value=user_input_raw.get("MonthlyCharges", 20.0))
    user_input_raw["TotalCharges"] = st.number_input("Total Charges", 0.0, 10000.0, value=user_input_raw.get("TotalCharges", 100.0))

# --- Prepare input for prediction ---
input_df_raw = pd.DataFrame([user_input_raw])  # Create DataFrame from user input
input_df_encoded = pd.get_dummies(input_df_raw)  # Encode categorical features
input_df_encoded = input_df_encoded.reindex(columns=X.columns, fill_value=0)  # Align with training data columns
input_df_encoded[numerical_cols] = scaler.transform(input_df_encoded[numerical_cols])  # Scale numeric features

# --- Make prediction ---
prediction = model.predict(input_df_encoded)[0]  # Predict churn
prediction_proba = model.predict_proba(input_df_encoded)[0][prediction] if hasattr(model, "predict_proba") else None  # Predict probability if available

# --- Display prediction result ---
st.subheader(" Prediction Result")
if prediction == 1:
    st.success("**Prediction:** The customer is likely to **Churn** ")
else:
    st.success("**Prediction:** The customer is likely to **Stay** ")

if prediction_proba is not None:
    st.write(f"**Confidence Level:** {prediction_proba:.2%}")

# --- Evaluate model performance ---
st.subheader(" Model Performance on Test Set")
test_pred = model.predict(X_test)
test_acc = accuracy_score(y_test, test_pred)
st.metric(label="Test Accuracy", value=f"{test_acc * 100:.2f}%")

# --- Confusion Matrix ---
st.write("### Confusion Matrix")
cm = confusion_matrix(y_test, test_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', xticklabels=["Stay", "Churn"], yticklabels=["Stay", "Churn"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
st.pyplot(fig)

# --- Classification Report ---
st.write("### Classification Report")
report = classification_report(y_test, test_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
st.dataframe(report_df.style.background_gradient(cmap="Purples"))  # Display with color gradient

# --- Additional feature input hints ---
with st.sidebar.expander(" Feature Hints"):
    st.markdown("""
    - **Categorical fields** are pre-defined.
    - **tenure**: Number of months the customer has stayed.
    - **MonthlyCharges**: Monthly amount billed.
    - **TotalCharges**: Total amount billed.
    """)
st.sidebar.markdown("---")
st.sidebar.info(" Use dropdowns and sliders to simulate customer profiles easily.")

