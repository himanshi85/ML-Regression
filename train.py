import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import os

# Load dataset
df = pd.read_csv("Data/dataset.csv")

X = df[["Population", "Income", "MarketingSpend"]]
y = df["Profit"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
pred = model.predict(X_test)
print("R2 Score:", r2_score(y_test, pred))
print("MSE:", mean_squared_error(y_test, pred))

# Save model
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/profit_model.pkl")

print("Model saved successfully!")
