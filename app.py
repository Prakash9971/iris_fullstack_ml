from flask import Flask, render_template, request
import numpy as np
import joblib
from sklearn.datasets import load_iris

app = Flask(__name__)

# Load trained model and scaler
model = joblib.load("knn_model.pkl")
scaler = joblib.load("scaler.pkl")
iris = load_iris()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = [
        float(request.form["sepal_length"]),
        float(request.form["sepal_width"]),
        float(request.form["petal_length"]),
        float(request.form["petal_width"])
    ]

    data = np.array([data])
    data_scaled = scaler.transform(data)
    prediction = model.predict(data_scaled)

    result = iris.target_names[prediction[0]]
    return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
