import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

# Load the dataset
file_path = "flood.csv"
df = pd.read_csv(file_path)

# Preview and handle missing values
print("Dataset preview:\n", df.head())
print("\nMissing values before handling:\n", df.isnull().sum())
df.fillna(df.mean(numeric_only=True), inplace=True)
print("\nMissing values after handling:\n", df.isnull().sum())

# Drop unnecessary columns
columns_to_drop = ['RiverManagement']
df.drop(columns=columns_to_drop, inplace=True)

# Handle missing values again if any
df.fillna(df.mean(numeric_only=True), inplace=True)

# Define features (X) and target (y)
X = df.iloc[:, :-1]  # All columns except the last one
y = df.iloc[:, -1]   # Target variable (FloodProbability)

# Scale features using MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Convert back to DataFrame for consistent column names
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.3, random_state=42)

# Train Ridge Regression model
model = Ridge()
model.fit(X_train, y_train)

# Save model, scaler, and feature names
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(X.columns.tolist(), 'features.pkl')

# Predict and evaluate
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# Save metrics
with open("metrics.txt", "w") as f:
    f.write(f"{rmse}\n{r2}\n{mae}")

# Save Actual vs Predicted Plot
if not os.path.exists("static"):
    os.makedirs("static")

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color='blue', label="Predicted vs Actual")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', label="Ideal Fit")
plt.xlabel("Actual Flood Probability")
plt.ylabel("Predicted Flood Probability")
plt.title("Ridge Regression: Actual vs Predicted Flood Probability")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("static/plot.png")
plt.close()

print("\u2705 Training complete. Model, scaler, metrics, and plot saved.")
