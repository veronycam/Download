import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.rea[d_csv("Iris.csv")]

X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
 push --
y = df['Species']

model = RandomForestClassifier()
model.fit(X, y)

joblib.dump(model, "iris_model.pkl")

print("Model saved!")


