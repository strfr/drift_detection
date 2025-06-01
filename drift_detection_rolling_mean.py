import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

FILE = Path("LD2011_2014.txt")
df = pd.read_csv(FILE, sep=";", index_col=0, parse_dates=True, decimal=",").astype(float)
df["TotalLoad"] = df.sum(axis=1)
df = df[~df["TotalLoad"].isna()]

window = 96 * 7 
df["rolling_mean"] = df["TotalLoad"].rolling(window=window).mean()
df["rolling_std"]  = df["TotalLoad"].rolling(window=window).std()

df["rolling_shift"] = df["rolling_mean"].diff().abs()

threshold = df["rolling_shift"].mean() + 7 * df["rolling_shift"].std()
drift_points = df[df["rolling_shift"] > threshold]

plt.figure(figsize=(18, 6))
plt.plot(df.index, df["TotalLoad"], label="Total Load", color="lightgray", lw=0.6)
plt.plot(df.index, df["rolling_mean"], label="Rolling Mean (7d)", color="blue", lw=2)
plt.plot(df.index, df["rolling_std"], label="Rolling Std (7d)", color="orange", lw=2)
plt.scatter(drift_points.index, drift_points["rolling_mean"],
            color="red", s=50, label="Detected Drift", zorder=5)
plt.title("Drift Detection via Rolling Mean – LD2011-2014")
plt.xlabel("Date"); plt.ylabel("kW")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.show()
