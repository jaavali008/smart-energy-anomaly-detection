from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class AnomalyDetector:
    FEATURES = ["Global_active_power", "Voltage", "Global_intensity", "hour",
                "is_weekend", "rolling_mean_24", "rolling_std_24", "other_use"]

    def __init__(self, contamination=0.01, z_limit=3):
        self.contamination = contamination
        self.z_limit = z_limit

    def detect(self, h):
        h = h.copy()

        # Isolation Forest
        X = StandardScaler().fit_transform(h[self.FEATURES])
        iso = IsolationForest(n_estimators=200, contamination=self.contamination,
                              random_state=42)
        h["iso_anomaly"] = iso.fit_predict(X) == -1
        h["iso_score"] = -iso.score_samples(X)      # higher = more abnormal

        # Z-score baseline, compared against the same hour of day
        g = h.groupby("hour")["Global_active_power"]
        h["z"] = (h["Global_active_power"] - g.transform("mean")) / g.transform("std")
        h["z_anomaly"] = h["z"].abs() > self.z_limit

        print("Isolation Forest anomalies:", h["iso_anomaly"].sum())
        print("Z-score anomalies:", h["z_anomaly"].sum())
        print("Flagged by both:", (h["iso_anomaly"] & h["z_anomaly"]).sum())
        return h