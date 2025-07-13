import pandas as pd
import numpy as np

# 1) load
df_obs = pd.read_csv(
    "input/obs.csv", encoding="utf-8", parse_dates=["기상관측일시"]
)
df_fcst = pd.read_csv(
    "input/fcst.csv", encoding="utf-8", parse_dates=["기상관측일시"]
)

# 2) rename
obs_rename = {
    "기상관측일시":   "date",
    "습도(%)":      "humid_obs",
    "기온(degC)":   "degC_obs",
    "대기압(mmHg)": "mmHg_obs",
}
fcst_rename = {
    "기상관측일시":      "date",
    "일사량(w/m^2)":     "sunlight_fcst",
    "습도(%)":         "humid_fcst",
    "절대습도":         "abs_humid_fcst",
    "기온(degC)":      "degC_fcst",
    "대기압(hPa)":      "hPa_fcst",
}
df_obs.rename(columns=obs_rename, inplace=True)
df_fcst.rename(columns=fcst_rename, inplace=True)

# 3) cast floats
for col in ["humid_obs","degC_obs","mmHg_obs"]:
    df_obs[col] = df_obs[col].astype(float)
for col in ["sunlight_fcst","humid_fcst","abs_humid_fcst","degC_fcst","hPa_fcst"]:
    df_fcst[col] = df_fcst[col].astype(float)

# 4) merge and reindex to hourly
df = pd.merge(df_obs, df_fcst, on="date", how="outer")
df = df.set_index("date").sort_index()
full_idx = pd.date_range(df.index.min(), df.index.max(), freq="h")
df = df.reindex(full_idx).rename_axis("date").reset_index()

# 5) remember where the forecasts begin: last 24h of fcst.csv beyond obs
horizon = 24
last_obs_time = df_obs["date"].max()
forecast_start = last_obs_time + pd.Timedelta(hours=1)

# 6) feature engineering
forecast_cols = ["sunlight_fcst","humid_fcst","abs_humid_fcst","degC_fcst","hPa_fcst"]
target_cols   = ["humid_obs","degC_obs","mmHg_obs"]
diff_cols     = [f"{c}_diff" for c in forecast_cols + target_cols]

# 6a) missing‐flag for forecasts
for c in forecast_cols:
    df[f"{c}_was_missing"] = df[c].isna().astype(str).astype("category")

# 6b) solar term & cyclical
solar_cutoffs = [
    (1,5),(1,20),(2,4),(2,19),(3,5),(3,20),(4,5),(4,20),
    (5,5),(5,21),(6,6),(6,21),(7,7),(7,22),(8,7),(8,23),
    (9,7),(9,23),(10,8),(10,23),(11,7),(11,22),(12,7),(12,22),
]
def solar_term_label(ts):
    for i,(m,d) in reversed(list(enumerate(solar_cutoffs))):
        if ts >= ts.replace(month=m, day=d):
            return str(i+1)
    return "24"
df["solar_term"] = df["date"].apply(solar_term_label).astype("category")
df["hod"] = df["date"].dt.hour.astype(str).astype("category")
df["dow"] = df["date"].dt.dayofweek.astype(str).astype("category")
df["moy"] = df["date"].dt.month.astype(str).astype("category")

# 6c) fill forecast NaNs by hourly mean
seasonal_means = df.groupby("hod", observed=False)[forecast_cols].transform("mean")
df[forecast_cols] = df[forecast_cols].fillna(seasonal_means)

# 6d) diffs
for c in target_cols + forecast_cols:
    df[f"{c}_diff"] = df[c].diff()

# 7) drop rows with missing obs/diffs **only before** forecast_start
mask_bad = (
    (df["date"] <= last_obs_time)  # historical period
    & (
        df[target_cols + diff_cols]
        .isna()
        .any(axis=1)
    )
)
df = df.loc[~mask_bad].reset_index(drop=True)

# 8) add ids
df["group_id"] = 0
df["time_idx"] = np.arange(len(df), dtype=int)

# 9) save
df.to_csv("input/base_data.csv", index=False)
print("Preprocessing Complete!")
