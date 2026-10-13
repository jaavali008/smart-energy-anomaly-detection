# Smart Energy Consumption Analytics and Anomaly Detection System

Data science project on the UCI Individual Household Electric Power Consumption dataset
(about 2 million minute-level readings, Dec 2006 to Nov 2010).

## What it does
- Cleans the data and resamples it to hourly (pandas)
- Detects anomalies with Isolation Forest and a Z-score baseline (scikit-learn)
- Forecasts hourly power with Random Forest, compared with a "same hour yesterday" baseline
- Stores alerts in SQLite and models anomaly-to-source relationships in Neo4j (Cypher)
- Interactive Plotly dashboard with energy, cost and CO2 estimates

## Project structure
```
src/        DataLoader, Preprocessor, AnomalyDetector, Forecaster,
            AlertManager, GraphStore, Dashboard
main.py     runs the full pipeline cell by cell
output/     dashboard.html
```

## Dataset
Download from https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption
and place `household_power_consumption.txt` in a `data/` folder (not included in the repo because of its size).

## Run
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```
Neo4j is optional. Set `RUN_NEO4J = True` in `main.py` and enter your own credentials when asked.

## Notes
- Anomalies mean unusual usage, not confirmed faults (no labelled data).
- The CO2 factor (0.82 kg/kWh) and tariff (Rs 7/kWh) are assumptions and can be changed.