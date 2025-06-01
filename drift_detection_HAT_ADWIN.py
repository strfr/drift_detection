import pandas as pd
import numpy as np
from river import tree, drift, metrics
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import medfilt

FILE = Path("LD2011_2014.txt")
FIG  = "concept_drift_ld2011_2014_lines_clean.png"

df = (
    pd.read_csv(FILE, sep=";", index_col=0, parse_dates=True, decimal=",")
      .astype(float)
)
df["TotalLoad"] = df.sum(axis=1)
df = df.loc[~df["TotalLoad"].isna()]

df["Hour"], df["Weekday"] = df.index.hour, df.index.dayofweek
df["PrevLoad"] = df["TotalLoad"].shift(1).bfill()
median = df["TotalLoad"].median()
df["HighLoad"] = (df["TotalLoad"] > median).astype(int)

raw_diff = df["TotalLoad"].diff().abs().fillna(0)
z_diff   = raw_diff / raw_diff.std()
z_diff   = pd.Series(medfilt(z_diff.values, 5), index=z_diff.index)


DELTA  = 0.0005
WARMUP = 2000   

u_adwin, u_pos = drift.ADWIN(delta=DELTA), []
for i, v in enumerate(z_diff, 0):
    u_adwin.update(v)
    if u_adwin.drift_detected:
        u_pos.append(i)
        u_adwin = drift.ADWIN(delta=DELTA)

model = tree.HoeffdingAdaptiveTreeClassifier(
    grace_period=2000, delta=1e-7, leaf_prediction="nb"
)
s_adwin, s_pos = drift.ADWIN(delta=DELTA), []

for i, (idx, row) in enumerate(df.iterrows(), 0):
    x = {
        "Hour":      row["Hour"],
        "Weekday":   row["Weekday"],
        "PrevLoad":  row["PrevLoad"],
        "TotalLoad": row["TotalLoad"],
    }
    y = row["HighLoad"]

    y_hat = model.predict_one(x)
    model.learn_one(x, y)

    if y_hat is None or i < WARMUP:
        continue

    s_adwin.update(int(y_hat != y))
    if s_adwin.drift_detected:
        s_pos.append(i)
        s_adwin = drift.ADWIN(delta=DELTA)

u_ts = df.index[u_pos]
s_ts = df.index[s_pos]

plt.figure(figsize=(18, 9))
plt.plot(df.index, df["TotalLoad"],
         lw=0.6, color="steelblue", label="Total Load", zorder=1)

for t in u_ts:
    plt.axvline(x=t, color="red", alpha=0.5, linewidth=1, label="_nolegend_")

for t in s_ts:
    plt.axvline(x=t, color="lime", alpha=0.6,
                linewidth=1.2, linestyle="--", label="_nolegend_")

plt.axvline(df.index[0], color="red", alpha=0.5, linewidth=1,
            label="ADWIN z(|ΔY|)")
plt.axvline(df.index[0], color="lime", alpha=0.6, linewidth=1.2,
            linestyle="--", label="HAT + ADWIN")

plt.title("Concept-Drift Points – LD2011-2014 (Vertical Line View)")
plt.xlabel("Date")
plt.ylabel("kW")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIG, dpi=300)
plt.show()

print(f"Unsupervised drifts: {len(u_pos)}")
print(f"Supervised  drifts: {len(s_pos)}")
