import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd
from data import DataProvider
import matplotlib.pyplot as plt
import typing as T


class XGBoostBenchmarker:
    """
    A class to build, train, and evaluate a high-performance
    XGBoost model for voltage prediction.
    """
    # --- Configuration Constants ---
    DEFAULT_FEATURES = ["SoC", "BMS Temperature(℃)", "Battery Current(A)"]
    VALIDATION_SPLIT = 0.2
    RANDOM_STATE = 42
    EARLY_STOPPING_ROUNDS = 10 # XGBoost's version of early stopping patience
    DATA_PROVIDER = DataProvider()

    def __init__(self, features: T.List[str] = None, xgb_params: T.Dict = None):
        """
        Initializes the benchmarker with XGBoost configuration.

        Args:
            features (list): Feature column names to use for training.
            xgb_params (dict): Hyperparameters for the XGBoost regressor.
        """
        self.features = features if features is not None else self.DEFAULT_FEATURES
        
        if xgb_params is None:
            # Sensible defaults for high performance
            self.xgb_params = {
                'n_estimators': 1000,
                'learning_rate': 0.05,
                'max_depth': 5,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'objective': 'reg:squarederror',
                'random_state': self.RANDOM_STATE,
                'early_stopping_rounds': self.EARLY_STOPPING_ROUNDS
            }
        else:
            self.xgb_params = xgb_params

    def run_benchmark(self, charging_dfs: T.List[pd.DataFrame], rangesStrings: T.List[str]) -> T.Dict[str, T.Any]:
        """
        Trains and evaluates the XGBoost model on multiple datasets.
        """
        benchmark_results = {}

        for i, df in enumerate(charging_dfs):
            range_name = rangesStrings[i]
            print(f"\n🚀 Processing Range {i+1} ({range_name}) with XGBoost...")

            # 1. Prepare and split data
            X = df[self.features].values.astype(np.float32)
            y = df["Battery Voltage(V)"].values.astype(np.float32).reshape(-1, 1)

            X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
                X, y, test_size=self.VALIDATION_SPLIT, random_state=self.RANDOM_STATE
            )

            # 2. Scale data (still good practice)
            x_scaler = MinMaxScaler()
            X_train = x_scaler.fit_transform(X_train_raw)
            X_test = x_scaler.transform(X_test_raw)

            y_scaler = MinMaxScaler()
            # XGBoost prefers a 1D array for y, so we use .ravel()
            y_train = y_scaler.fit_transform(y_train_raw).ravel()
            y_test = y_scaler.transform(y_test_raw).ravel()

            # 3. Initialize and train the XGBoost model
            model = xgb.XGBRegressor(**self.xgb_params)
            
            model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=False # Set to True to see training progress
            )

            # 4. Evaluate the model
            y_pred_scaled = model.predict(X_test)
            # Reshape for inverse transformation
            y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))
            
            # Calculate metrics
            mse = mean_squared_error(y_test_raw, y_pred)
            mae = mean_absolute_error(y_test_raw, y_pred)
            r2 = r2_score(y_test_raw, y_pred)
            
            # Store results
            benchmark_results[range_name] = {
                "model": model,
                "x_scaler": x_scaler,
                "y_scaler": y_scaler,
                "mse": mse,
                "mae": mae,
                "r2": r2,
                "y_pred": y_pred,
                "y_true": y_test_raw,
                "df_original": df,
            }
            print(f"  ✅ XGBoost Done. MAE: {mae:.4f} V | R²: {r2:.4f}")
            
        return benchmark_results