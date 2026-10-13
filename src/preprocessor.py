import pandas as pd


class Preprocessor:
    def clean(self, df):
        """Fill the missing minutes by interpolation."""
        return df.interpolate(limit_direction="both")

    def to_hourly(self, df):
        """2M minute rows -> hourly rows (mean for rates, sum for energy)."""
        hourly = df.resample("h").agg({
            "Global_active_power": "mean",
            "Voltage": "mean",
            "Global_intensity": "mean",
            "Sub_metering_1": "sum",
            "Sub_metering_2": "sum",
            "Sub_metering_3": "sum",
        })
        print("Hourly shape:", hourly.shape)
        return hourly

    def add_features(self, hourly):
        h = hourly.copy()
        h["hour"] = h.index.hour
        h["dayofweek"] = h.index.dayofweek
        h["is_weekend"] = (h["dayofweek"] >= 5).astype(int)
        h["month"] = h.index.month
        p = h["Global_active_power"]
        h["rolling_mean_24"] = p.rolling(24, min_periods=1).mean()
        h["rolling_std_24"] = p.rolling(24, min_periods=1).std().fillna(0)
        h["total_submeter"] = h[["Sub_metering_1", "Sub_metering_2",
                                 "Sub_metering_3"]].sum(axis=1)
        # kW x 1000 = Wh per hour; subtract what the sub-meters measured
        h["other_use"] = p * 1000 - h["total_submeter"]
        return h