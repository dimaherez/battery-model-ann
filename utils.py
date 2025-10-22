from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from data import DataProvider
import numpy as np
import typing as T

def predict_and_plot(result_key: str, results: T.Dict[str, T.Any], data_provider: DataProvider, features: T.List[str] = ["SoC", "BMS Temperature(℃)", "Battery Current(A)"]):
        """
        Uses the trained model to make predictions and plots the results.
        Also calculates metrics against the smoothed baseline.
        """
        result = results.get(result_key)
        if not result:
            print(f"Error: Result key '{result_key}' not found.")
            return

        # --- 1. Get components from results ---
        
        # Handle different model storage (Ensemble vs. single model)
        if "ensemble_models" in result:
            ensemble = result['ensemble_models'] # For EnsembleModelBenchmarker
        else:
            model = result['model'] # For all other benchmarkers
            
        x_scaler, y_scaler = result['x_scaler'], result['y_scaler']
        df_original = result['df_original']
        
        # --- 2. Prepare the "True" Smoothed Data ---
        smoothed_df = data_provider.get_smoothed_data(df_original).sort_values(by="SoC").reset_index(drop=True)
        X_for_prediction = smoothed_df[features].values
        soc_for_plotting = smoothed_df["SoC"].values
        
        # This is your "true" smoothed ground truth
        voltage_smoothed_baseline = smoothed_df["Battery Voltage(V)"].values.flatten()

        # --- 3. Generate Model Predictions for the Same Data ---
        X_for_prediction_scaled = x_scaler.transform(X_for_prediction)
        
        # Get scaled predictions based on model type
        if "ensemble_models" in result:
            # Logic for EnsembleModelBenchmarker
            ensemble_preds_scaled = [m.predict(X_for_prediction_scaled, verbose=0) for m in ensemble]
            avg_pred_scaled = np.mean(ensemble_preds_scaled, axis=0)
        else:
            avg_pred_scaled = model.predict(X_for_prediction_scaled)
        
        # Inverse transform to get the final predicted line
        voltage_model_prediction = y_scaler.inverse_transform(avg_pred_scaled.reshape(-1, 1)).flatten()

        # --- 4. 💡 CALCULATE METRICS VS. SMOOTHED DATA ---
        
        # Compare the model's line to the smoothed baseline
        mae_vs_smooth = mean_absolute_error(voltage_smoothed_baseline, voltage_model_prediction)
        mse_vs_smooth = mean_squared_error(voltage_smoothed_baseline, voltage_model_prediction)
        r2_vs_smooth = r2_score(voltage_smoothed_baseline, voltage_model_prediction)

        print(f"\n--- Metrics vs. Smoothed Baseline for '{result_key}' ---")
        print(f"  Smoothed MAE: {mae_vs_smooth:.4f} V")
        print(f"  Smoothed MSE: {mse_vs_smooth:.4f}")
        print(f"  Smoothed R²:  {r2_vs_smooth:.4f}")
        print("-------------------------------------------------")
        
        plt.figure(figsize=(12, 7))
        plt.scatter(df_original["SoC"], df_original["Battery Voltage(V)"], 
                    label='Raw Original Data', alpha=0.2, color='gray', s=10)
        plt.plot(soc_for_plotting, voltage_smoothed_baseline,
                 label='Smoothed Baseline Curve', color='blue', linewidth=3)
        plt.plot(soc_for_plotting, voltage_model_prediction,
                 label=f'Model Prediction (R²: {r2_vs_smooth:.3f})', 
                 color='red', linewidth=3, linestyle='--')
        
        plt.title(f'Model Prediction vs. Smoothed Baseline for "{result_key}"')
        plt.xlabel('State of Charge (SoC) %')
        plt.ylabel('Battery Voltage (V)')
        plt.legend()
        plt.grid(True)
        plt.show()