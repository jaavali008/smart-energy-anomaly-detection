import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


class Forecaster:
    def __init__(self, train_frac=0.8):
        self.train_frac = train_frac
        self.y_test = None
        self.pred = None
        self.importance = None

    def _build_xy(self, h):
        p = h["Global_active_power"]
        f = h[["Global_active_power", "hour", "dayofweek",
               "is_weekend", "month"]].copy()
        f["lag_1"] = p.shift(1)                        # 1 hour ago
        f["lag_24"] = p.shift(24)                      # same hour yesterday
        f["lag_168"] = p.shift(168)                    # same hour last week
        f["roll_24"] = p.shift(1).rolling(24).mean()   # previous 24 h, no leakage
        f = f.dropna()
        return f.drop(columns="Global_active_power"), f["Global_active_power"]

    @staticmethod
    def _report(name, y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        print(f"{name:35s} MAE={mae:.3f} kW   RMSE={rmse:.3f} kW")

    def run(self, h):
        X, y = self._build_xy(h)
        split = int(len(X) * self.train_frac)          # time-based split
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, self.y_test = y.iloc[:split], y.iloc[split:]

        model = RandomForestRegressor(n_estimators=100, max_depth=15,
                                      n_jobs=-1, random_state=42)
        model.fit(X_train, y_train)
        self.pred = pd.Series(model.predict(X_test), index=X_test.index)

        self._report("Baseline (same hour yesterday)", self.y_test, X_test["lag_24"])
        self._report("Random Forest", self.y_test, self.pred)

        self.importance = pd.Series(model.feature_importances_,
                                    index=X.columns).sort_values(ascending=False)
        print("\nFeature importance:\n", self.importance.round(3))