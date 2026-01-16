import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import yaml
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def load_config():
    config_path = BASE_DIR / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_dataset(config):
    data_cfg = config["data"]  

    data_path = BASE_DIR / data_cfg["path"]

    print("Dataset path:", data_path)
    print("Dataset exists:", data_path.exists())

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    df = pd.read_csv(data_path)

    target = data_cfg["target_column"]
    X = df.drop(columns=[target])
    y = df[target]

    return X, y


def train_and_evaluate(X, y, config):
    eval_cfg = config["evaluation"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=eval_cfg["test_size"],
        random_state=eval_cfg["random_state"],
    )

    model_cfg = config["model"]
    model = LinearRegression(
        fit_intercept=model_cfg["params"]["fit_intercept"]
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return model, mse, r2


def main():
    config = load_config()
    X, y = load_dataset(config)

    model, mse, r2 = train_and_evaluate(X, y, config)

    print("Model trained successfully")
    print(f"Coefficients: {model.coef_}")
    print(f"Intercept: {model.intercept_:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"R2 Score: {r2:.4f}")


if __name__ == "__main__":
    main()
