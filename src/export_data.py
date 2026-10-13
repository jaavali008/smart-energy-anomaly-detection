import json
import os
import numpy as np
from sklearn.metrics import mean_absolute_error


def export_web(h, alert_df, fc, path="web/data.js"):
    """Save the pipeline results as web/data.js for the JavaScript dashboard."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    p = h["Global_active_power"]
    daily_mean = p.resample("D").mean().dropna()
    daily_kwh = p.resample("D").sum().reindex(daily_mean.index)
    g = p.groupby(h["hour"])
    heat = h.pivot_table(values="Global_active_power", index="dayofweek",
                         columns="hour", aggfunc="mean")
    last = fc.y_test.index[-168:]
    base = p.shift(24).reindex(fc.y_test.index)
    r = lambda s, n=3: [round(float(x), n) for x in s]

    data = {
        "dates": [d.strftime("%Y-%m-%d") for d in daily_mean.index],
        "daily_mean": r(daily_mean), "daily_kwh": r(daily_kwh, 2),
        "heat": [r(row) for row in heat.values],
        "hour_mean": r(g.mean()), "hour_std": r(g.std()),
        "fc_labels": [t.strftime("%d %b %H:00") for t in last],
        "fc_actual": r(fc.y_test[last]), "fc_pred": r(fc.pred[last]),
        "mae_rf": round(float(mean_absolute_error(fc.y_test, fc.pred)), 3),
        "mae_base": round(float(mean_absolute_error(fc.y_test, base)), 3),
        "n_iso": int(h["iso_anomaly"].sum()), "n_z": int(h["z_anomaly"].sum()),
        "n_both": int((h["iso_anomaly"] & h["z_anomaly"]).sum()),
        "anomalies": [
            {"t": a.timestamp, "kw": round(a.power_kw, 3), "sev": a.severity,
             "src": a.main_source, "ex": round(a.excess_kwh, 3), "sc": round(a.score, 3)}
            for a in alert_df.itertuples()],
    }
    with open(path, "w", encoding="utf-8") as f:
        f.write("window.DATA = " + json.dumps(data) + ";")
    print("Saved", path, f"({os.path.getsize(path)/1024:.0f} KB)")
