from sklearn.ensemble import StackingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd
from data import DataProvider
import matplotlib.pyplot as plt
import typing as T

class StackingBenchmarker:
    """
    A class to build, train, and evaluate a stacking ensemble model,
    combining multiple diverse models for robust prediction.
    """
    DEFAULT_FEATURES = ["SoC", "BMS Temperature(℃)", "Battery Current(A)"]
    VALIDATION_SPLIT = 0.2
    RANDOM_STATE = 42
    DATA_PROVIDER = DataProvider()

    def __init__(self, features: T.List[str] = None, base_models: T.List = None, meta_model=None):
        """
        Initializes the benchmarker with stacking configuration.

        Args:
            features (list): Feature column names to use for training.
            base_models (list): A list of ('name', model) tuples for Level 0.
            meta_model: The final Level 1 model to combine predictions.
        """
        self.features = features if features is not None else self.DEFAULT_FEATURES
        
        if base_models is None:
            # A good mix of models: a tree-based, a gradient boosted, and a linear model
            self.base_models = [
                ('rf', RandomForestRegressor(n_estimators=50, random_state=self.RANDOM_STATE)),
                ('lgbm', lgb.LGBMRegressor(random_state=self.RANDOM_STATE)),
                ('ridge', Ridge(random_state=self.RANDOM_STATE))
            ]
        else:
            self.base_models = base_models
            
        self.meta_model = meta_model if meta_model is not None else Ridge()

    def run_benchmark(self, charging_dfs: T.List[pd.DataFrame], rangesStrings: T.List[str]) -> T.Dict[str, T.Any]:
        """
        Trains and evaluates the Stacking ensemble on multiple datasets.
        """
        benchmark_results = {}
        for i, df in enumerate(charging_dfs):
            range_name = rangesStrings[i]
            print(f"\n🧱 Processing Range {i+1} ({range_name}) with Stacking Ensemble...")
            
            # 1. Data Prep and Scaling
            X = df[self.features].values.astype(np.float32)
            y = df["Battery Voltage(V)"].values.astype(np.float32).reshape(-1, 1)
            X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
                X, y, test_size=self.VALIDATION_SPLIT, random_state=self.RANDOM_STATE)
            x_scaler = MinMaxScaler()
            X_train = x_scaler.fit_transform(X_train_raw)
            X_test = x_scaler.transform(X_test_raw)
            y_scaler = MinMaxScaler()
            y_train = y_scaler.fit_transform(y_train_raw).ravel() # Use .ravel() for sklearn models

            # 2. Initialize and train the Stacking Regressor
            # cv=5 means it uses 5-fold cross-validation to generate predictions for the meta-model,
            # which is a robust way to prevent data leakage.
            model = StackingRegressor(estimators=self.base_models, final_estimator=self.meta_model, cv=5)
            model.fit(X_train, y_train)

            # 3. Evaluate the model
            y_pred_scaled = model.predict(X_test)
            y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))

            mse = mean_squared_error(y_test_raw, y_pred)
            mae = mean_absolute_error(y_test_raw, y_pred)
            r2 = r2_score(y_test_raw, y_pred)

            benchmark_results[range_name] = {
                "model": model, "x_scaler": x_scaler, "y_scaler": y_scaler,
                "mse": mse, "mae": mae, "r2": r2, "y_pred": y_pred, "y_true": y_test_raw, "df_original": df,
            }
            print(f"  ✅ Stacking Done. MAE: {mae:.4f} V | R²: {r2:.4f}")
        return benchmark_results