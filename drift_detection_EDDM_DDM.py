import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import medfilt    
from river import tree                     

try:
    from river.drift.ddm import DDM, EDDM
except ImportError:
    try:
        from river.drift.binary import DDM, EDDM


FILE = Path("LD2011_2014.txt")          
FIG  = "drift_lines_ddm_eddm_only.png"

df = (
    pd.read_csv(FILE, sep=";", index_col=0, parse_dates=True, decimal=",")
      .astype(float)
)
df["TotalLoad"] = df.sum(axis=1)
df = df.dropna(subset=["TotalLoad"])

df["Hour"]     = df.index.hour
df["Weekday"]  = df.index.dayofweek
df["PrevLoad"] = df["TotalLoad"].shift(1).bfill()
median         = df["TotalLoad"].median()
df["HighLoad"] = (df["TotalLoad"] > median).astype(int)

model = tree.HoeffdingAdaptiveTreeClassifier(
    grace_period=30,
    delta=1e-7,
    leaf_prediction="nb"    
)

ddm  = DDM()               
eddm = EDDM()

ddm_idx, eddm_idx = [], []  

MIN_INST            = 5_000      
MIN_SAMPLES_DDM     = 300       
MIN_SAMPLES_EDDM    = 300        
ERR_WIN             = 3          
THRESH              = 0.20      

err_window = []                

for i, (ts, row) in enumerate(df.iterrows(), 0):
    features = {
        "Hour": ts.hour,
        "Weekday": ts.dayofweek,
        "PrevLoad": df["TotalLoad"].iat[i - 1] if i else row["TotalLoad"],
        "TotalLoad": row["TotalLoad"],
    }
    y       = row["HighLoad"]
    y_pred  = model.predict_one(features)
    model.learn_one(features, y)

    if y_pred is None:
        continue

    raw_err = 0 if y_pred == y else 1
    err_window.append(raw_err)
    if len(err_window) > ERR_WIN:
        err_window.pop(0)
    err_bit = 1 if sum(err_window) / len(err_window) > THRESH else 0

    if i >= MIN_INST + MIN_SAMPLES_DDM:
        ddm.update(err_bit)
        if ddm.drift_detected:
            ddm_idx.append(i)
            ddm = DDM()                

    if i >= MIN_INST + MIN_SAMPLES_EDDM:
        eddm.update(err_bit)
        if eddm.drift_detected:
            eddm_idx.append(i)
            eddm = EDDM()             

plt.figure(figsize=(18, 9))
plt.plot(df.index, df["TotalLoad"],
         lw=0.6, color="steelblue", label="Total Load")

for t in df.index[ddm_idx]:
    plt.axvline(t, color="orange", alpha=0.8, lw=1.2)
for t in df.index[eddm_idx]:
    plt.axvline(t, color="purple", alpha=0.8, lw=1.2, ls="--")
  
plt.axvline(df.index[0], color="orange", lw=1,           label="DDM (error)")
plt.axvline(df.index[0], color="purple", lw=1, ls="--",  label="EDDM (error)")

plt.title("Drift Lines – DDM vs EDDM (2011-2014)")
plt.xlabel("Date"); plt.ylabel("kW")
plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
plt.savefig(FIG, dpi=300)
plt.show()
