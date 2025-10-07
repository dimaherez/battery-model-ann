import pandas as pd
from numpy.polynomial.polynomial import Polynomial

class DataProvider:
    def __init__(self):
        self.file_path = "dataset.xlsx"
        self.cols = ["Timestamp", "SoC(%)", "Battery Current(A)", "Battery Voltage(V)"]
        self.searching_current = 10
        self.spread = 3
        self.min_current = self.searching_current - self.spread
        self.max_current = self.searching_current + self.spread

    def read_excel(self):
        return pd.read_excel(self.file_path, usecols=self.cols)
        
    def scale_data(self, df):
        df["Timestamp"] = pd.to_datetime(df['Timestamp'])
        df['time_diff_sec'] = df['Timestamp'].shift(1) - df['Timestamp']
        df['time_diff_sec'] = df['time_diff_sec'].dt.total_seconds().abs() // 60
        df["SoC"] = df["SoC(%)"] / 100
        df["Battery Voltage(V)"] = df["Battery Voltage(V)"] / 8
        # df["Battery Current(A)"] = df["Battery Current(A)"] * -1
        df = df.dropna(subset=["time_diff_sec"])
        return df
    
    def quantile_grouping(self, df, bin_size=0.01, q=0.95):
        df["SoC_bin"] = (df["SoC"] // bin_size) * bin_size
        grouped = df.groupby("SoC_bin").agg({
            "Battery Voltage(V)": lambda x: x.quantile(q),
            "Battery Current(A)": lambda x: x.quantile(q),
            "time_diff_sec": "mean"
        }).reset_index().rename(columns={"SoC_bin": "SoC"})
        return grouped
    
    def get_grouped_df_by_soc(self, df):
        return df.groupby("SoC").agg({
            "Battery Voltage(V)": "median",
            "Battery Current(A)": "median",
            'time_diff_sec': "mean"
        }).reset_index()
    
    def smooth_voltages(self, df):
        x = df["SoC"]
        y = df["Battery Voltage(V)"]

        # Polynomial fit for smoother voltage
        poly_fit = Polynomial.fit(x, y, deg=5)
        smoothed_voltage = poly_fit(x)

        # Reattach all original fields with smoothed voltage
        smoothed_df = df.copy()
        smoothed_df["Battery Voltage(V)"] = smoothed_voltage

        # Optional: insert empty Timestamp if not present
        if "Timestamp" not in smoothed_df.columns:
            smoothed_df["Timestamp"] = pd.NaT

        return smoothed_df
    


    def get_discharging_data(self, df, current_min, current_max): # positive currents
        filtered = df[(df["Battery Current(A)"] >= current_min) & (df["Battery Current(A)"] <= current_max)]
        grouped_df = self.get_grouped_df_by_soc(filtered)
    
        return self.smooth_voltages(grouped_df)

    def get_charging_data(self, df, current_min, current_max): # negative currents
        filtered = df[(df["Battery Current(A)"] >= current_max) & (df["Battery Current(A)"] <= current_min)]
        return filtered
    
    def get_smoothed_data(self, df):
        grouped_df = self.get_grouped_df_by_soc(df)
        return self.smooth_voltages(grouped_df)
    
