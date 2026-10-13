import os
import sqlite3
import pandas as pd

SOURCES = {"Sub_metering_1": "Kitchen", "Sub_metering_2": "Laundry",
           "Sub_metering_3": "Water heater/AC", "other_use": "Other appliances"}
CO2_PER_KWH = 0.82      # assumed grid factor, change if needed


class AlertManager:
    def __init__(self, db_path="output/alerts.db"):
        self.db_path = db_path
        self.alert_df = None

    def build(self, h):
        alerts = h[h["iso_anomaly"]].copy()

        # severity from the anomaly score (3 equal groups)
        alerts["severity"] = pd.qcut(alerts["iso_score"], 3,
                                     labels=["Low", "Medium", "High"]).astype(str)

        # main source = most unusual vs its own normal for that hour
        z_src = pd.DataFrame(index=h.index)
        for col in SOURCES:
            g = h.groupby("hour")[col]
            z_src[col] = (h[col] - g.transform("mean")) / g.transform("std")
        alerts["main_source"] = z_src.loc[alerts.index].idxmax(axis=1).map(SOURCES)

        # excess energy over the normal for that hour, and CO2
        normal = h.groupby("hour")["Global_active_power"].transform("mean")
        alerts["excess_kwh"] = (alerts["Global_active_power"]
                                - normal[alerts.index]).clip(lower=0)
        alerts["co2_kg"] = alerts["excess_kwh"] * CO2_PER_KWH
        alerts["timestamp"] = alerts.index.strftime("%Y-%m-%d %H:%M")

        self.alert_df = alerts[["timestamp", "Global_active_power", "iso_score",
                                "severity", "main_source", "excess_kwh",
                                "co2_kg"]].reset_index(drop=True)
        self.alert_df.columns = ["timestamp", "power_kw", "score", "severity",
                                 "main_source", "excess_kwh", "co2_kg"]
        return self.alert_df

    def save_sqlite(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        self.alert_df.to_sql("alerts", conn, if_exists="replace", index=False)

        print(pd.read_sql("""
            SELECT severity, COUNT(*) AS alerts,
                   ROUND(SUM(excess_kwh),1) AS excess_kwh,
                   ROUND(SUM(co2_kg),1) AS co2_kg
            FROM alerts GROUP BY severity ORDER BY alerts DESC""", conn))
        print(pd.read_sql("""
            SELECT main_source, COUNT(*) AS alerts,
                   ROUND(SUM(excess_kwh),1) AS excess_kwh
            FROM alerts GROUP BY main_source ORDER BY alerts DESC""", conn))
        conn.close()