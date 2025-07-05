import pandas as pd
import numpy as np

df_obs = pd.read_csv("./data/recorded_weather.csv",
                     header=0, names=["date", "humid_obs", "degC_obs", "mmHg_obs"],
                     parse_dates=["date"])
print(df_obs.head())
print("total rows:", len(df_obs))

df_fcst = pd.read_csv("./data/forecasted_weather.csv",
                      header=0, names=["date", "sunlight_fcst", "humid_fcst", "abs_humid_fcst", "degC_fcst", "hPa_fcst"],
                      parse_dates=["date"])
print(df_fcst.head())
print("total rows:", len(df_fcst))

df = pd.merge(df_obs, df_fcst, on="date", how="outer")
df = df.sort_values("date").reset_index(drop=True)

solar_cutoffs = [
    (1,  5), (1, 20),
    (2,  4), (2, 19),
    (3,  5), (3, 20),
    (4,  5), (4, 20),
    (5,  5), (5, 21),
    (6,  6), (6, 21),
    (7,  7), (7, 22),
    (8,  7), (8, 23),
    (9,  7), (9, 23),
    (10,  8), (10, 23),
    (11,  7), (11, 22),
    (12,  7), (12, 22),
]

def solar_term_label(ts: pd.Timestamp) -> int:
    for i in range(23, -1, -1):
        m, d = solar_cutoffs[i]
        cutoff = ts.replace(month=m, day=d)
        if ts >= cutoff:
            return i + 1
    return 24

df["solar_term"] = df["date"].apply(solar_term_label).astype(str).astype("category")
df["hod"] = df["date"].dt.hour
df["dow"] = df["date"].dt.dayofweek
df["moy"] = df["date"].dt.month
df["time_idx"] = df.reset_index().index

df = df.iloc[:-1].reset_index(drop=True)

fcst_cols = ["sunlight_fcst","humid_fcst","abs_humid_fcst","degC_fcst","hPa_fcst"]
df = df.dropna(subset=fcst_cols, how="all").reset_index(drop=True)

df["segment_id"] = (
    df["date"].diff()
      .gt(pd.Timedelta(hours=1))
      .cumsum()
)

print(df.segment_id.value_counts().sort_index())

test_segments = [6, 7, 8]

test_df  = df[df["segment_id"].isin(test_segments)].reset_index(drop=True)
train_df = df[~df["segment_id"].isin(test_segments)].reset_index(drop=True)

train_df.to_csv("./data/weather_train.csv", index=False)
test_df.to_csv( "./data/weather_test.csv",  index=False)

print("train segments:", train_df["segment_id"].unique())
print("test  segments:", test_df["segment_id"].unique())