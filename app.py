import streamlit as st
import numpy as np
import joblib

# Feature list (giống lúc train)
features = [
    'fixed acidity',
    'volatile acidity',
    'citric acid',
    'residual sugar',
    'chlorides',
    'free sulfur dioxide',
    'total sulfur dioxide',
    'density',
    'pH',
    'sulphates',
    'alcohol'
]

# Cho phép chọn model
model_files = {
    'Random Forest': 'models/rf.pkl',
    'Logistic Regression': 'models/lr.pkl',
    'SVC': 'models/svc.pkl',
    'Decision Tree': 'models/tree.pkl',
    'KNN': 'models/knn.pkl'
}

st.title("🍷 Dự đoán chất lượng rượu bằng nhiều mô hình")

# Chọn model
selected_model_name = st.selectbox("Chọn mô hình:", list(model_files.keys()))
model = joblib.load(model_files[selected_model_name])

# Nhập dữ liệu
inputs = []
for feat in features:
    val = st.number_input(f"{feat}", min_value=0.0, step=0.1)
    inputs.append(val)

# Dự đoán
if st.button("Dự đoán"):
    X_input = np.array([inputs])
    prediction = model.predict(X_input)[0]
    st.success(f"✅ [{selected_model_name}] dự đoán chất lượng: {prediction}")
