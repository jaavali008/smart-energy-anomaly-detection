# %% 1. Load data (Unit I)
from getpass import getpass
from src.data_loader import DataLoader
from src.preprocessor import Preprocessor
from src.anomaly_detector import AnomalyDetector
from src.forecaster import Forecaster
from src.alert_manager import AlertManager
from src.graph_store import GraphStore
from src.dashboard import Dashboard

loader = DataLoader()
loader.download()
df = loader.load()

# %% 2. Clean, resample to hourly, add features (Unit I)
prep = Preprocessor()
df = prep.clean(df)
hourly = prep.to_hourly(df)
h = prep.add_features(hourly)
print(h.describe().T[["mean", "min", "max"]])

# %% 3. Anomaly detection (Unit II)
h = AnomalyDetector(contamination=0.01).detect(h)

# %% 4. Forecasting (Unit II)
fc = Forecaster()
fc.run(h)

# %% 5. Alerts + SQLite (Unit IV)
am = AlertManager()
alert_df = am.build(h)
am.save_sqlite()

# %% 6. Neo4j (Unit IV), optional
RUN_NEO4J = False      # set True only when you want Neo4j
if RUN_NEO4J:
    uri = input("NEO4J_URI: ").strip()
    user = input("NEO4J_USERNAME: ").strip()
    pwd = getpass("NEO4J_PASSWORD: ").strip()
    gs = GraphStore(uri, user, pwd)
    gs.load(alert_df)
    gs.report()
    gs.close()

# %% 7. Dashboard (Unit V)
dash = Dashboard(h, alert_df, fc.y_test, fc.pred)
dash.kpis()
fig = dash.build()
fig.show()
# %%
