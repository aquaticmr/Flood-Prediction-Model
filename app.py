# app.py

from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model, scaler, and feature names
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")

# Load metrics
with open("metrics.txt", "r") as f:
    lines = f.read().splitlines()
    rmse, r2, mae = map(lambda x: round(float(x), 4), lines)

@app.route('/')
def index():
    return render_template("index.html", features=features, rmse=rmse, r2=r2, mae=mae, plot_url="/static/plot.png")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    values = np.array(data['features'], dtype=float).reshape(1, -1)
    scaled_values = scaler.transform(values)
    prediction = model.predict(scaled_values)[0]
    return jsonify({'prediction': round(prediction, 4)})

if __name__ == '__main__':
    app.run(debug=True)
