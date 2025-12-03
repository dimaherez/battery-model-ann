import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU
from tensorflow.keras.activations import gelu
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from data import DataProvider
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import typing as T

class SingleANNBenchmaker:

    DEFAULT_FEATURES = ["SoC", "BMS Temperature(℃)", "Battery Current(A)"]
    BATCH_SIZE = 16
    MAX_EPOCHS = 200
    EARLY_STOPPING_PATIENCE = 15
    VALIDATION_SPLIT = 0.2
    RANDOM_STATE = 42
    DATA_PROVIDER = DataProvider()

    def __init__(self, features: T.List[str] = None):
        
        self.features = features if features is not None else self.DEFAULT_FEATURES
        self.input_dim = len(self.features)
        
        self.models = {
            "ReLU": lambda dim: self._build_sequential_model(dim, [32, 16], activation='relu'),
            "LeakyReLU": lambda dim: self._build_leaky_model(dim, [32, 16]),
            "Tanh": lambda dim: self._build_sequential_model(dim, [32, 16], activation='tanh'),
            "GELU": lambda dim: self._build_sequential_model(dim, [32, 16], activation=gelu),
        }
        
        self.early_stopping_callback = keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=self.EARLY_STOPPING_PATIENCE, 
            restore_best_weights=True
        )


    def _build_sequential_model(self, input_dim: int, hidden_layers: T.List[int], 
                                activation: T.Union[str, T.Callable] = 'relu', 
                                output_activation: T.Optional[str] = None) -> Sequential:
        
        layers = [keras.Input(shape=(input_dim,))]
        
        
        for units in hidden_layers:
            layers.append(Dense(units, activation=activation))
        
        layers.append(Dense(1, activation=output_activation)) 
        
        return Sequential(layers)

    def _build_leaky_model(self, input_dim: int, hidden_layers: T.List[int]) -> Sequential:
        layers = [keras.Input(shape=(input_dim,))]
        
        for units in hidden_layers:
            layers.append(Dense(units))
            layers.append(LeakyReLU(alpha=0.01))
            
        layers.append(Dense(1))

        return Sequential(layers)

    def run_benchmark(self, charging_dfs: T.List[pd.DataFrame], rangesStrings: T.List[str]) -> T.Dict[str, T.Any]:
        benchmark_results = {}

        for i, df in enumerate(charging_dfs):
            range_name = rangesStrings[i]
            print(f"\n⚡ Processing Charging Range {i+1} ({range_name}): Current ∈ [{df['Battery Current(A)'].min():.1f}, {df['Battery Current(A)'].max():.1f}] A")

            try:
                X = df[self.features].values.astype(np.float32)
                y = df["Battery Voltage(V)"].values.astype(np.float32).reshape(-1, 1)
            except KeyError as e:
                print(f"Error: Missing feature column {e} in DataFrame for range {range_name}. Skipping.")
                continue

            X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
                    X, y, test_size=self.VALIDATION_SPLIT, random_state=self.RANDOM_STATE
                )

            x_scaler = MinMaxScaler()
            X_train = x_scaler.fit_transform(X_train_raw)
            X_test = x_scaler.transform(X_test_raw)
            y_scaler = MinMaxScaler()
            y_train = y_scaler.fit_transform(y_train_raw)
            y_test = y_scaler.transform(y_test_raw)
            train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(
                buffer_size=len(X_train)
            ).batch(self.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
            
            test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(
                self.BATCH_SIZE
            ).prefetch(tf.data.AUTOTUNE)
            for name, builder in self.models.items():
                print(f"  -> Training model: {name}...")
                model = builder(X_train.shape[1])
                model.compile(optimizer='adam', loss='mse', metrics=['mae'])
                history = model.fit(
                    train_ds,
                    epochs=self.MAX_EPOCHS,
                    validation_data=test_ds,
                    callbacks=[self.early_stopping_callback],
                    verbose=0
                )
                y_pred_scaled = model.predict(X_test, verbose=0)
                y_pred = y_scaler.inverse_transform(y_pred_scaled)
                mse = mean_squared_error(y_test_raw, y_pred)
                mae = mean_absolute_error(y_test_raw, y_pred)
                r2 = r2_score(y_test_raw, y_pred)
                
                key = f"{range_name}_{name}"
                benchmark_results[key] = {
                    "model": model,
                    "x_scaler": x_scaler,
                    "y_scaler": y_scaler,
                    "history": history.history,
                    "mse": mse,
                    "mae": mae,
                    "r2": r2,
                    "x_test": X_test_raw,
                    "y_pred": y_pred,
                    "y_true": y_test_raw,
                    "df_original": df,
                    "epochs_trained": len(history.history['loss'])
                }
                print(f"  ✅ Done. MAE: {mae:.4f} V | R²: {r2:.4f} | Epochs: {len(history.history['loss'])}")
        return benchmark_results
            