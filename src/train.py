import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

def main():
    # 1. Resolve paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "delivery_data.csv")
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "eta_model.joblib")

    print(f"Loading data from: {data_path}...")
    df = pd.read_csv(data_path)

    # 2. Define target and drop leakage/irrelevant columns
    target_col = "Time_taken_min"
    drop_cols = ["Order_ID", "Order_Date", "Average_Speed_kmph"]
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])

    # 3. Specify feature subsets
    numeric_features = [
        "Order_Hour",
        "Is_Weekend",
        "Is_Festival",
        "Rider_Experience_Years",
        "Rider_Rating",
        "Restaurant_Rating",
        "Order_Items",
        "Preparation_Time_Min",
        "Road_Distance_km",
        "Number_of_Signals",
    ]

    categorical_features = [
        "Day_of_Week",
        "Weather",
        "Pickup_Zone",
        "Dropoff_Zone",
        "Vehicle_Type",
        "Cuisine_Type",
        "Restaurant_Load",
        "Delivery_Distance_Category",
        "Traffic_Level",
        "Delivery_Priority",
    ]

    X = df[numeric_features + categorical_features]
    y = df[target_col]

    # 4. Train-test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"Dataset split: Train={len(X_train):,} samples, Test={len(X_test):,} samples\n")

    # 5. Build ColumnTransformer preprocessing pipeline
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # 6. Define models to compare
    models = {
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=100, max_depth=16, random_state=42, n_jobs=-1
        ),
        "XGBRegressor": XGBRegressor(
            n_estimators=150, max_depth=6, learning_rate=0.08, random_state=42, n_jobs=-1
        ),
    }

    results = {}
    fitted_pipelines = {}

    for name, regressor in models.items():
        print(f"Training {name}...")
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("regressor", regressor),
        ])

        pipeline.fit(X_train, y_train)
        fitted_pipelines[name] = pipeline

        y_pred = pipeline.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results[name] = {"MAE": mae, "R2": r2}
        print(f"  {name} -> Test MAE: {mae:.2f} mins, R2 Score: {r2:.4f}\n")

    # 7. Identify the better model (higher R2 and lower MAE)
    best_model_name = min(results, key=lambda k: results[k]["MAE"])
    best_pipeline = fitted_pipelines[best_model_name]

    print("=" * 60)
    print(f"COMPARISON SUMMARY:")
    for name, metrics in results.items():
        print(f"  - {name}: MAE = {metrics['MAE']:.2f} min | R2 = {metrics['R2']:.4f}")
    print(f"\nBest Model: {best_model_name}")
    print("=" * 60)

    # 8. Save the best pipeline
    joblib.dump(best_pipeline, model_path)
    print(f"Saved best pipeline to: {model_path}")

if __name__ == "__main__":
    main()
