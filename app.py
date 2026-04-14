import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("Iris.csv")

# Features and target
X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = df['Species']

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# App UI
st.title("🌸 Iris Flower Prediction App")

st.write("Enter flower measurements:")

sepal_length = st.slider("Sepal Length", 4.0, 8.0)
sepal_width = st.slider("Sepal Width", 2.0, 4.5)
petal_length = st.slider("Petal Length", 1.0, 7.0)
petal_width = st.slider("Petal Width", 0.1, 2.5)

# Prediction
if st.button("Predict"):
    prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    st.success(f"Prediction: {prediction[0]}")
