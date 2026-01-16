import joblib
import sys

if len(sys.argv) != 4:
    print("Usage: python predict.py <Population> <Income> <MarketingSpend>")
    sys.exit(1)

population = float(sys.argv[1])
income = float(sys.argv[2])
marketing = float(sys.argv[3])

model = joblib.load("model/profit_model.pkl")

result = model.predict([[population, income, marketing]])

print(f"Predicted Profit: {result[0]:.2f}")
