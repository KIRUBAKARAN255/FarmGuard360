import random
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

RAW_CSV = "farmguard360_raw_unprocessed_dataset.csv"
REFINED_CSV = "farmguard360_1year_seasonal_dataset.csv"

# =========================================================================
# PART A: GENERATE RAW UNPROCESSED DATASET (1,103 ROWS WITH SENSOR GLITCHES)
# =========================================================================
def build_raw_unprocessed_dataset():
    random.seed(42)
    np.random.seed(42)
    start_date = datetime(2025, 11, 1)
    shifts = [
        ("08:00:00", "Morning", -2.5),
        ("14:00:00", "Afternoon", +4.0),
        ("18:00:00", "Evening", -1.0),
    ]

    rows = []
    for day_offset in range(365):
        dt = start_date + timedelta(days=day_offset)
        month = dt.month
        date_str = dt.strftime("%Y-%m-%d")

        if month in (3, 4, 5, 6):
            season, base_temp, base_hum, base_rest, tol = "SUMMER_WARM", 35.5, 48.0, 10.0, 1.35
        elif month in (11, 12, 1, 2):
            season, base_temp, base_hum, base_rest, tol = "WINTER_COOL", 23.5, 72.0, 7.0, 1.20
        else:
            season, base_temp, base_hum, base_rest, tol = "TEMPERATE_STD", 29.0, 62.0, 8.0, 1.25

        adaptive_limit = round(base_rest * tol, 2)

        for time_str, shift_name, temp_delta in shifts:
            ambient_temp = round(base_temp + temp_delta + random.gauss(0, 1.1), 1)
            humidity = round(max(32.0, min(92.0, base_hum + random.gauss(0, 4.5))), 1)

            if random.random() < 0.06:
                obs_rest = round(adaptive_limit * random.uniform(1.05, 1.45), 2)
            else:
                obs_rest = round(max(1.2, random.gauss(base_rest * 0.72, 1.6)), 2)

            lethargy_idx = round(min(1.0, obs_rest / adaptive_limit), 2)

            if random.random() < 0.08:
                suit_cov = round(random.uniform(6.0, 27.5), 1)
                suit_ok = "NO"
            else:
                suit_cov = round(random.uniform(35.0, 78.0), 1)
                suit_ok = "YES"

            if lethargy_idx >= 0.75 and suit_ok == "NO":
                status = "DUAL_ALERT_LETHARGY_AND_SUIT"
            elif lethargy_idx >= 0.75:
                status = "SICK_ANIMAL_ALERT"
            elif suit_ok == "NO":
                status = "SAFETY_SUIT_BREACH"
            else:
                status = "NORMAL"

            rows.append([
                date_str, time_str, shift_name, dt.strftime("%b"), season,
                ambient_temp, humidity, "SHIELD-3 (BIO-SHED)",
                obs_rest, base_rest, adaptive_limit,
                lethargy_idx, suit_cov, suit_ok, status
            ])

    cols = [
        "Date", "Time", "Shift", "Month", "Season_Profile",
        "Ambient_Temp_C", "Humidity_Pct", "Zone",
        "Observed_Rest_Sec", "Learned_Baseline_Sec", "Adaptive_Limit_Sec",
        "Lethargy_Index", "Safety_Suit_Coverage_Pct", "Safety_Suit_OK", "Compliance_Status"
    ]
    df_dirty = pd.DataFrame(rows, columns=cols)

    # Inject realistic unprocessed sensor & logging anomalies
    df_dirty.loc[[12, 88, 215, 410, 670], "Ambient_Temp_C"] = np.nan
    df_dirty.loc[[45, 312, 590, 815], "Humidity_Pct"] = np.nan
    df_dirty.loc[[104, 502, 920], "Observed_Rest_Sec"] = np.nan
    df_dirty.loc[[250, 740], "Safety_Suit_Coverage_Pct"] = np.nan
    df_dirty.loc[[33, 444], "Observed_Rest_Sec"] = [-1.5, 0.0]
    df_dirty.loc[[199, 612], "Ambient_Temp_C"] = [-99.9, 88.5]
    df_dirty.loc[[777], "Safety_Suit_Coverage_Pct"] = 125.0
    df_dirty.loc[[18, 305], "Season_Profile"] = [" summer_warm ", "Winter_Cool"]
    df_dirty.loc[[72, 515], "Shift"] = [" morning", "EVENING "]
    df_dirty = pd.concat([df_dirty, df_dirty.iloc[[10, 55, 120, 240, 380, 510, 640, 890]]], ignore_index=True)

    df_dirty.to_csv(RAW_CSV, index=False)


# =========================================================================
# PART B: PYTHON DATA REFINERY PIPELINE (AUDIT -> CLEAN -> IMPUTE -> SAVE)
# =========================================================================
def run_refinery():
    build_raw_unprocessed_dataset()
    df_raw = pd.read_csv(RAW_CSV)

    print("==========================================================")
    print("   STAGE 1: RAW UNPROCESSED DATASET QUALITY AUDIT")
    print("==========================================================")
    print(f"Raw File                 : {RAW_CSV}")
    print(f"Total Raw Records        : {df_raw.shape[0]} rows x {df_raw.shape[1]} cols")
    print(f"Missing (NaN) Cells      : {int(df_raw.isna().sum().sum())}")
    print(f"Duplicate Shift Logs     : {int(df_raw.duplicated().sum())}")
    print(f"Negative/Zero Rest (<=0s): {int((df_raw['Observed_Rest_Sec'] <= 0).sum())}")
    print(f"Temp Spikes (<15 or >45C): {int(((df_raw['Ambient_Temp_C'] < 15) | (df_raw['Ambient_Temp_C'] > 45)).sum())}")
    print(f"Invalid PPE (>100%)      : {int((df_raw['Safety_Suit_Coverage_Pct'] > 100).sum())}")

    df = df_raw.copy()

    # 1. Trim whitespace & standardize text casing
    for col in ["Shift", "Season_Profile", "Zone", "Safety_Suit_OK", "Compliance_Status"]:
        df[col] = df[col].astype(str).str.strip()
    df["Shift"] = df["Shift"].str.title()
    df["Season_Profile"] = df["Season_Profile"].str.upper()

    # 2. Remove duplicate shift rows
    dups_dropped = int(df.duplicated(subset=["Date", "Time", "Shift"]).sum())
    df = df.drop_duplicates(subset=["Date", "Time", "Shift"]).reset_index(drop=True)

    # 3. Nullify physically impossible sensor glitches
    df.loc[(df["Ambient_Temp_C"] < 15.0) | (df["Ambient_Temp_C"] > 45.0), "Ambient_Temp_C"] = np.nan
    df.loc[(df["Humidity_Pct"] < 30.0) | (df["Humidity_Pct"] > 95.0), "Humidity_Pct"] = np.nan
    df.loc[df["Observed_Rest_Sec"] <= 0.0, "Observed_Rest_Sec"] = np.nan
    df.loc[(df["Safety_Suit_Coverage_Pct"] < 0.0) | (df["Safety_Suit_Coverage_Pct"] > 100.0), "Safety_Suit_Coverage_Pct"] = np.nan

    # 4. Climate-aware grouped median imputation (zero data loss)
    df["Ambient_Temp_C"] = df["Ambient_Temp_C"].fillna(
        df.groupby(["Season_Profile", "Shift"])["Ambient_Temp_C"].transform("median")
    ).round(1)
    df["Humidity_Pct"] = df["Humidity_Pct"].fillna(
        df.groupby("Season_Profile")["Humidity_Pct"].transform("median")
    ).round(1)
    df["Observed_Rest_Sec"] = df["Observed_Rest_Sec"].fillna(
        df.groupby("Season_Profile")["Observed_Rest_Sec"].transform("median")
    ).round(2)
    df["Safety_Suit_Coverage_Pct"] = df["Safety_Suit_Coverage_Pct"].fillna(
        df["Safety_Suit_Coverage_Pct"].median()
    ).round(1)

    # 5. Recompute derived normalized Lethargy Index & Compliance Status
    df["Lethargy_Index"] = (df["Observed_Rest_Sec"] / df["Adaptive_Limit_Sec"]).clip(0.0, 1.0).round(2)
    df["Safety_Suit_OK"] = np.where(df["Safety_Suit_Coverage_Pct"] >= 32.0, "YES", "NO")
    df["Compliance_Status"] = np.select(
        [
            (df["Lethargy_Index"] >= 0.75) & (df["Safety_Suit_OK"] == "NO"),
            (df["Lethargy_Index"] >= 0.75),
            (df["Safety_Suit_OK"] == "NO"),
        ],
        ["DUAL_ALERT_LETHARGY_AND_SUIT", "SICK_ANIMAL_ALERT", "SAFETY_SUIT_BREACH"],
        default="NORMAL"
    )

    df.to_csv(REFINED_CSV, index=False)

    print("\n==========================================================")
    print("   STAGE 3: POST-REFINERY CLEAN DATASET VERIFICATION")
    print("==========================================================")
    print(f"Refined File             : {REFINED_CSV}")
    print(f"Total Clean Records      : {df.shape[0]} rows x {df.shape[1]} cols (Removed {dups_dropped} duplicates)")
    print(f"Missing (NaN) Cells      : {int(df.isna().sum().sum())}")
    print(f"Duplicate Shift Logs     : {int(df.duplicated().sum())}")
    print(f"Negative/Zero Rest (<=0s): {int((df['Observed_Rest_Sec'] <= 0).sum())}")
    print(f"Temp Spikes (<15 or >45C): {int(((df['Ambient_Temp_C'] < 15) | (df['Ambient_Temp_C'] > 45)).sum())}")
    print(f"Invalid PPE (>100%)      : {int((df['Safety_Suit_Coverage_Pct'] > 100).sum())}")
    print("\nCompliance Breakdown:")
    print(df["Compliance_Status"].value_counts().to_string())


if __name__ == "__main__":
    run_refinery()