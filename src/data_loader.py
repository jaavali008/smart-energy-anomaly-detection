import os
import zipfile
import urllib.request
import pandas as pd

URL = ("https://archive.ics.uci.edu/static/public/235/"
       "individual+household+electric+power+consumption.zip")


class DataLoader:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.txt_path = os.path.join(data_dir, "household_power_consumption.txt")

    def download(self):
        if os.path.exists(self.txt_path):
            print("Dataset already present, skipping download.")
            return
        os.makedirs(self.data_dir, exist_ok=True)
        zip_path = os.path.join(self.data_dir, "power.zip")
        req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as r, open(zip_path, "wb") as f:
            f.write(r.read())
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(self.data_dir)
        print("Downloaded and extracted.")

    def load(self):
        df = pd.read_csv(self.txt_path, sep=";", na_values="?", low_memory=False)
        df["datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"],
                                        format="%d/%m/%Y %H:%M:%S")
        df = df.drop(columns=["Date", "Time"]).set_index("datetime").astype(float)
        print("Raw shape:", df.shape)
        print("Missing values per column:\n", df.isna().sum())
        return df