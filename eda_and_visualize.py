import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

CSV_PATH = "farmguard360_1year_seasonal_dataset.csv"
df = pd.read_csv(CSV_PATH)

# =========================================================================
# PART 1: FINAL EXPLORATORY DATA ANALYSIS (EDA) STATISTICAL SUMMARY
# =========================================================================
num_cols = [
    "Ambient_Temp_C",
    "Humidity_Pct",
    "Observed_Rest_Sec",
    "Adaptive_Limit_Sec",
    "Lethargy_Index",
    "Safety_Suit_Coverage_Pct",
]

print("==================================================================")
print("   1. DESCRIPTIVE STATISTICS (n = 1,095 CLEAN SHIFT RECORDS)")
print("==================================================================")
print(df[num_cols].describe().T[["mean", "50%", "std", "min", "max"]].round(2))

print("\n==================================================================")
print("   2. SEASONAL PROFILE & ADAPTIVE ALERT COMPARISON")
print("==================================================================")
seasonal = df.groupby("Season_Profile").agg(
    Records=("Date", "count"),
    Mean_Temp_C=("Ambient_Temp_C", "mean"),
    Mean_Humidity=("Humidity_Pct", "mean"),
    Mean_Rest_Sec=("Observed_Rest_Sec", "mean"),
    Adaptive_Limit=("Adaptive_Limit_Sec", "mean"),
    Mean_Lethargy=("Lethargy_Index", "mean"),
    Alert_Rate_Pct=("Lethargy_Index", lambda s: (s >= 0.75).mean() * 100),
).round(2)
print(seasonal)

summer = df[df["Season_Profile"] == "SUMMER_WARM"]
static_summer_alerts = int((summer["Observed_Rest_Sec"] >= 7.5).sum())
adaptive_summer_alerts = int((summer["Lethargy_Index"] >= 0.75).sum())
print(
    f"\nSummer Heat False-Alarm Check: Static 10s cutoff = {static_summer_alerts} alerts "
    f"vs. Adaptive 13.5s cutoff = {adaptive_summer_alerts} alerts "
    f"({static_summer_alerts - adaptive_summer_alerts} false alarms prevented)"
)

print("\n==================================================================")
print("   3. PEARSON CORRELATION MATRIX")
print("==================================================================")
print(df[num_cols].corr().round(3))

# =========================================================================
# PART 2: GENERATE ALL 6 REPORT VISUALIZATIONS (FIGURES 7.3.1 - 7.3.6)
# =========================================================================
plt.rcParams["font.sans-serif"] = "DejaVu Sans"

# Figure 7.3.1: Compliance & Health Status Distribution
fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=220)
labels = ["Normal\nCompliant", "Sick Animal\nAlert", "Safety Suit\nBreach", "Dual Alert\n(Lethargy+PPE)"]
keys = ["NORMAL", "SICK_ANIMAL_ALERT", "SAFETY_SUIT_BREACH", "DUAL_ALERT_LETHARGY_AND_SUIT"]
counts = [int((df["Compliance_Status"] == k).sum()) for k in keys]
bars = ax.bar(labels, counts, color=["#2E7D32", "#C62828", "#F9A825", "#6A1B9A"], width=0.52)
for b, c in zip(bars, counts):
    ax.text(b.get_x() + b.get_width() / 2, c + 16, f"{c} ({c/len(df)*100:.1f}%)",
            ha="center", va="bottom", fontsize=8.5, fontweight="bold")
ax.set_ylim(0, max(counts) + 120)
ax.set_ylabel("Number of shift observations")
ax.set_title("Fig 7.3.1: Biosecurity & health status distribution (n = 1,095)", fontweight="bold")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("fig_7_3_1_status_distribution.png", dpi=220)
plt.close(fig)

# Figure 7.3.2: Climate-Calibrated Lethargy Index Histogram
fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=220)
ax.hist(df["Lethargy_Index"], bins=22, color="#1565C0", edgecolor="white", linewidth=0.8)
ax.axvline(0.75, color="#C62828", linestyle="--", linewidth=1.8, label="0.75 sickness alert threshold")
ax.set_xlabel("Normalized lethargy index (0.00 – 1.00)")
ax.set_ylabel("Observations")
ax.set_title("Fig 7.3.2: Distribution of climate-calibrated lethargy index", fontweight="bold")
ax.legend(frameon=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("fig_7_3_2_lethargy_histogram.png", dpi=220)
plt.close(fig)

# Figure 7.3.3: Observed Rest Duration vs Ambient Temperature
fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=220)
for sname, scol, slbl in [
    ("WINTER_COOL", "#1565C0", "Winter Cool (8.4s limit)"),
    ("TEMPERATE_STD", "#2E7D32", "Temperate Std (10.0s limit)"),
    ("SUMMER_WARM", "#EF6C00", "Summer Warm (13.5s limit)"),
]:
    sub = df[df["Season_Profile"] == sname]
    ax.scatter(sub["Ambient_Temp_C"], sub["Observed_Rest_Sec"], c=scol, label=slbl, alpha=0.65, s=18)
ax.set_xlabel("Ambient shed temperature (°C)")
ax.set_ylabel("Observed rest duration (s)")
ax.set_title("Fig 7.3.3: Observed rest duration vs ambient temperature", fontweight="bold")
ax.legend(title="Season", frameon=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("fig_7_3_3_rest_vs_temp.png", dpi=220)
plt.close(fig)

# Figure 7.3.4: Pearson Correlation Heatmap
fig, ax = plt.subplots(figsize=(5.5, 4.1), dpi=220)
short = ["Temp", "Humidity", "Rest", "Limit", "Lethargy", "Suit Cov."]
cmat = df[num_cols].corr().values
im = ax.imshow(cmat, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
ax.set_xticks(range(len(short)))
ax.set_yticks(range(len(short)))
ax.set_xticklabels(short, rotation=30, ha="right")
ax.set_yticklabels(short)
for i in range(len(short)):
    for j in range(len(short)):
        val = cmat[i, j]
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8.0,
                color="white" if abs(val) > 0.55 else "black")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
ax.set_title("Fig 7.3.4: Pearson correlation heatmap", fontweight="bold")
plt.tight_layout()
plt.savefig("fig_7_3_4_correlation_heatmap.png", dpi=220)
plt.close(fig)

# Figure 7.3.5: Effect of Seasonal Adaptive Thresholding in Summer
fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=220)
sub_sum = summer.head(160)
ax.scatter(np.arange(len(sub_sum)), sub_sum["Observed_Rest_Sec"], c="#6A1B9A", alpha=0.75, s=20, label="Summer rest observations")
ax.axhline(7.5, color="#EF6C00", linestyle=":", linewidth=1.8, label="Static 0.75×10.0s alert line (7.5s)")
ax.axhline(10.125, color="#C62828", linestyle="--", linewidth=1.8, label="Adaptive 0.75×13.5s alert line (10.1s)")
ax.set_xlabel("Summer observation sequence index")
ax.set_ylabel("Observed rest duration (s)")
ax.set_title("Fig 7.3.5: Effect of seasonal adaptive thresholding in summer", fontweight="bold")
ax.legend(frameon=False, loc="upper right")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("fig_7_3_5_summer_adaptive.png", dpi=220)
plt.close(fig)

# Figure 7.3.6: Monthly Mean Lethargy Index Across 365 Days
fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=220)
m_order = ["Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct"]
m_leth = df.groupby("Month")["Lethargy_Index"].mean().reindex(m_order)
ax.plot(m_order, m_leth.values, marker="o", color="#1565C0", linewidth=2.0)
ax.set_ylim(0.45, 0.72)
ax.set_xlabel("Month of 1-year observation window")
ax.set_ylabel("Mean lethargy index")
ax.set_title("Fig 7.3.6: Monthly mean lethargy index (365 days)", fontweight="bold")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("fig_7_3_6_monthly_trend.png", dpi=220)
plt.close(fig)

print("\nAll 6 visualization PNGs saved successfully in FarmGuard360_Final!")