import os
import joblib
import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


DATA_PATH = Path("data/processed/imdb_oscar.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

TARGET_COL = "IMDB_Rating"


def load_data(path=DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def prepare_features(df: pd.DataFrame):
    # numerical features
    numeric_cols = ["No_of_Votes", "Meta_score", "Runtime", "Gross"]
    data_numeric = df[numeric_cols].copy()

    # genre dummies (already in the CSV as 0/1 columns)
    genre_cols = [c for c in df.columns if c not in numeric_cols + [TARGET_COL, "film", "oscar_wins", "oscar_nominations"] + ["Star1","Star2","Star3","Star4","Director"] and set(df[c].dropna().unique()).issubset({0,1})]
    data_genres = df[genre_cols].copy()

    # actors (Star1..4 columns)
    actors = pd.concat([df[c] for c in ["Star1","Star2","Star3","Star4"] if c in df], ignore_index=True)
    top_actors = actors.value_counts().head(50).index
    data_actors = pd.DataFrame({f"actor_{a}": df[["Star1","Star2","Star3","Star4"]].isin([a]).any(axis=1).astype(int) for a in top_actors})

    # director
    if "Director" in df:
        top_directors = df["Director"].value_counts().head(20).index
        data_directors = pd.DataFrame({f"director_{d}": (df["Director"] == d).astype(int) for d in top_directors})
    else:
        data_directors = pd.DataFrame()

    # combine everything
    X_features = pd.concat([data_numeric, data_genres, data_actors, data_directors], axis=1).fillna(0)

    y_target = pd.to_numeric(df[TARGET_COL], errors="coerce")
    return X_features, y_target


def evaluate_model(model, X_train, X_test, y_train, y_test, name: str):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"\n=== {name} ===")
    print(f"MAE: {mae:.3f}")
    print(f"R²:  {r2:.3f}")
    return model, mae, r2


def main():
    df = load_data()
    X, y = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    
    # Saving features
    joblib.dump(X.columns.tolist(), MODEL_DIR / "features.pkl")

    results = {}

    # 1. RandomForest
    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model_rf, mae_rf, r2_rf = evaluate_model(rf, X_train, X_test, y_train, y_test, "RandomForest")
    joblib.dump(model_rf, MODEL_DIR / "rf_model.pkl")

    results["RandomForest"] = (mae_rf, r2_rf)

    # 2. GradientBoosting
    gb = GradientBoostingRegressor(random_state=42)
    model_gb, mae_gb, r2_gb = evaluate_model(gb, X_train, X_test, y_train, y_test, "GradientBoosting")
    joblib.dump(model_gb, MODEL_DIR / "gb_model.pkl")

    results["GradientBoosting"] = (mae_gb, r2_gb)

    # 3. Linear Regression
    lr = LinearRegression()
    model_lr, mae_lr, r2_lr = evaluate_model(lr, X_train, X_test, y_train, y_test, "LinearRegression")
    joblib.dump(model_lr, MODEL_DIR / "lr_model.pkl")

    results["LinearRegression"] = (mae_lr, r2_lr)

    # summary
    print("\n=== Final Comparison ===")
    for k, (mae, r2) in results.items():
        print(f"{k:<20} | MAE={mae:.3f} | R²={r2:.3f}")


if __name__ == "__main__":
    main()