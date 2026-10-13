import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

RATE = 7    # assumed Rs per kWh


class Dashboard:
    def __init__(self, h, alert_df, y_test, pred):
        self.h, self.alert_df = h, alert_df
        self.y_test, self.pred = y_test, pred

    def kpis(self):
        total = self.h["Global_active_power"].sum()
        excess = self.alert_df["excess_kwh"].sum()
        co2 = self.alert_df["co2_kg"].sum()
        print("=== SMART ENERGY DASHBOARD ===")
        print(f"Total consumption : {total:,.0f} kWh")
        print(f"Anomalies detected: {len(self.alert_df)}")
        print(f"Wasted energy     : {excess:,.0f} kWh  (~Rs {excess*RATE:,.0f})")
        print(f"CO2 avoidable     : {co2:,.0f} kg")

        top_hour = self.h.groupby("hour")["Global_active_power"].mean().idxmax()
        top_src = self.alert_df["main_source"].value_counts().idxmax()
        print("\nEnergy saving tips:")
        print(f"- Peak usage is around {top_hour}:00. Shift heavy appliances to off-peak hours.")
        print(f"- Top anomaly source: {top_src}. Check those appliances first.")
        print(f"- Avoiding the flagged spikes would save about {excess:,.0f} kWh "
              f"and {co2:,.0f} kg of CO2.")

    def build(self, html_path="output/dashboard.html"):
        h = self.h
        a = h[h["iso_anomaly"]]
        heat = h.pivot_table(values="Global_active_power", index="dayofweek",
                             columns="hour", aggfunc="mean")
        src = self.alert_df.groupby("main_source")["excess_kwh"].sum().sort_values()
        last = self.y_test.index[-168:]

        fig = make_subplots(
            rows=3, cols=2, row_heights=[0.35, 0.3, 0.35],
            specs=[[{"colspan": 2}, None], [{}, {}], [{"colspan": 2}, None]],
            subplot_titles=("Power with anomalies",
                            "Average power: day vs hour (kW)",
                            "Wasted energy by source (kWh)",
                            "Forecast vs actual (last 7 days)"))

        fig.add_trace(go.Scattergl(x=h.index, y=h["Global_active_power"],
                                   mode="lines", name="Power (kW)",
                                   line=dict(width=1)), row=1, col=1)
        fig.add_trace(go.Scattergl(x=a.index, y=a["Global_active_power"],
                                   mode="markers", name="Anomaly",
                                   marker=dict(color="red", size=5)), row=1, col=1)
        fig.add_trace(go.Heatmap(z=heat.values, x=heat.columns,
                                 y=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
                                 colorscale="YlOrRd", showscale=False,
                                 name="Avg power"), row=2, col=1)
        fig.add_trace(go.Bar(x=src.values, y=src.index, orientation="h",
                             marker_color="crimson", name="Excess kWh"), row=2, col=2)
        fig.add_trace(go.Scatter(x=last, y=self.y_test[last], name="Actual"), row=3, col=1)
        fig.add_trace(go.Scatter(x=last, y=self.pred[last], name="Forecast"), row=3, col=1)

        fig.update_layout(height=1000,
                          title="Smart Energy Consumption Analytics & Anomaly Detection")
        fig.write_html(html_path, include_plotlyjs="cdn")
        print("Saved", html_path)
        return fig