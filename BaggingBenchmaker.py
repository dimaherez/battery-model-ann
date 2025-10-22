import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from data import DataProvider 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import typing as T

class EnsembleModelBenchmarker:
    """
    A class to build, train, and evaluate an ENSEMBLE of neural network 
    models for robust voltage prediction.
    """

    # --- Configuration Constants ---
    DEFAULT_FEATURES = ["SoC", "BMS Temperature(℃)", "Battery Current(A)"]
    BATCH_SIZE = 32
    MAX_EPOCHS = 200
    EARLY_STOPPING_PATIENCE = 15
    VALIDATION_SPLIT = 0.2
    RANDOM_STATE = 42
    DATA_PROVIDER = DataProvider()

    def __init__(self, n_estimators: int = 10, features: T.List[str] = None):
        """
        Initializes the benchmarker with ensemble configuration.

        Args:
            n_estimators (int): The number of neural network models in the ensemble.
            features (list): A list of feature column names to use for training.
        """
        self.n_estimators = n_estimators
        self.features = features if features is not None else self.DEFAULT_FEATURES
        self.input_dim = len(self.features)
        
        # Define the early stopping callback once
        self.early_stopping_callback = keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=self.EARLY_STOPPING_PATIENCE, 
            restore_best_weights=True
        )

    def _build_base_model(self) -> Sequential:
        """
        Builds a single, robust neural network to be used as a base
        estimator in the ensemble. A slightly deeper model with Dropout
        is used for better generalization.
        """
        model = Sequential([
            keras.Input(shape=(self.input_dim,)),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            # Output layer for regression (1 continuous value)
            Dense(1) 
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model

    def run_benchmark(self, charging_dfs: T.List[pd.DataFrame], rangesStrings: T.List[str]) -> T.Dict[str, T.Any]:
        """
        Trains and evaluates the ensemble model on multiple datasets.

        Args:
            charging_dfs: A list of DataFrames for distinct charging ranges.
            rangesStrings: A list of strings corresponding to the range names.

        Returns:
            A dictionary containing results and trained artifacts for each experiment.
        """
        benchmark_results = {}

        for i, df in enumerate(charging_dfs):
            range_name = rangesStrings[i]
            print(f"\n⚡ Processing Charging Range {i+1} ({range_name}) with a {self.n_estimators}-model ensemble...")

            # 1. Prepare and split data
            try:
                X = df[self.features].values.astype(np.float32)
                y = df["Battery Voltage(V)"].values.astype(np.float32).reshape(-1, 1)
            except KeyError as e:
                print(f"Error: Missing feature column {e} in DataFrame for range {range_name}. Skipping.")
                continue

            X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
                X, y, test_size=self.VALIDATION_SPLIT, random_state=self.RANDOM_STATE
            )

            # 2. Scale data (fit ONLY on training data)
            x_scaler = MinMaxScaler()
            X_train = x_scaler.fit_transform(X_train_raw)
            X_test = x_scaler.transform(X_test_raw)

            y_scaler = MinMaxScaler()
            y_train = y_scaler.fit_transform(y_train_raw)
            y_test = y_scaler.transform(y_test_raw)
            
            # --- Ensemble Training Loop ---
            ensemble_models = []
            for j in range(self.n_estimators):
                print(f"  -> Training estimator {j+1}/{self.n_estimators}...")
                
                # Bagging: Create a bootstrap sample (sample with replacement)
                indices = np.random.choice(len(X_train), size=len(X_train), replace=True)
                X_batch, y_batch = X_train[indices], y_train[indices]
                
                # Build and train a fresh model on the bootstrap sample
                model = self._build_base_model()
                model.fit(
                    X_batch, y_batch,
                    epochs=self.MAX_EPOCHS,
                    batch_size=self.BATCH_SIZE,
                    validation_data=(X_test, y_test),
                    callbacks=[self.early_stopping_callback],
                    verbose=0
                )
                ensemble_models.append(model)
            
            # 3. Evaluate the full ensemble
            # Gather predictions from all models in the ensemble
            all_predictions_scaled = [model.predict(X_test, verbose=0) for model in ensemble_models]
            
            # Average the predictions
            y_pred_scaled_avg = np.mean(all_predictions_scaled, axis=0)
            
            # Inverse transform the final prediction
            y_pred = y_scaler.inverse_transform(y_pred_scaled_avg)
            
            # Calculate metrics
            mse = mean_squared_error(y_test_raw, y_pred)
            mae = mean_absolute_error(y_test_raw, y_pred)
            r2 = r2_score(y_test_raw, y_pred)
            
            # Store results
            benchmark_results[range_name] = {
                "ensemble_models": ensemble_models,
                "x_scaler": x_scaler,
                "y_scaler": y_scaler,
                "mse": mse,
                "mae": mae,
                "r2": r2,
                "y_pred": y_pred,
                "y_true": y_test_raw,
                "df_original": df,
            }
            print(f"  ✅ Ensemble Done. MAE: {mae:.4f} V | R²: {r2:.4f}")
            
        return benchmark_results

    def predict_and_plot(self, result_key: str, results: T.Dict[str, T.Any]):
        """
        Uses the trained ensemble to make predictions on a full dataset and
        plots the results against a smoothed baseline.
        """
        result = results.get(result_key)
        if not result:
            print(f"Error: Result key '{result_key}' not found.")
            return

        # Extract required components
        ensemble = result['ensemble_models']
        x_scaler, y_scaler = result['x_scaler'], result['y_scaler']
        df_original = result['df_original']
        
        smoothed_df = self.DATA_PROVIDER.get_smoothed_data(df_original)
        smoothed_df_sorted = smoothed_df.sort_values(by="SoC").reset_index(drop=True)

        X_for_prediction = smoothed_df_sorted[self.features].values
        soc_for_plotting = smoothed_df_sorted["SoC"].values
        voltage_smoothed_baseline = smoothed_df_sorted["Battery Voltage(V)"].values

        X_for_prediction_scaled = x_scaler.transform(X_for_prediction)
        
        # Get predictions from all models in the ensemble
        ensemble_preds_scaled = [model.predict(X_for_prediction_scaled, verbose=0) for model in ensemble]
        
        # Average the predictions
        avg_pred_scaled = np.mean(ensemble_preds_scaled, axis=0)
        
        # Inverse transform the final averaged prediction
        voltage_model_prediction = y_scaler.inverse_transform(avg_pred_scaled).flatten()

        # Plotting
        plt.figure(figsize=(12, 7))
        plt.scatter(df_original["SoC"], df_original["Battery Voltage(V)"], 
                    label='Raw Original Data', alpha=0.2, color='gray', s=10)
        plt.plot(soc_for_plotting, voltage_smoothed_baseline,
                 label='Smoothed Baseline Curve', color='blue', linewidth=3)
        plt.plot(soc_for_plotting, voltage_model_prediction,
                 label='Ensemble NN Prediction', color='red', linewidth=3, linestyle='--')
        
        plt.title(f'Ensemble Prediction vs. Smoothed Baseline for "{result_key}"')
        plt.xlabel('State of Charge (SoC) %')
        plt.ylabel('Battery Voltage (V)')
        plt.legend()
        plt.grid(True)
        plt.show()