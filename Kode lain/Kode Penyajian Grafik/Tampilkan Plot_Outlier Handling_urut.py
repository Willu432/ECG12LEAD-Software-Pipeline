import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import numpy as np
import re
from typing import Dict, List
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

"""
ECG feature plots averaged per subject & condition (ADAPTIVE VERSION)
- Outlier handling: feature-wise extreme values are replaced with the mean of non-outliers
- RR interval: two-sided normal range (600–1200 ms) highlighted
- PR & QRS (& QTc): upper-limit-only (≤ 200 ms, ≤ 110 ms, ≤ 460 ms) — area below is green
- Outputs:
  (1) Main figure (all features stacked in one canvas)
  (2) One figure per feature
  All saved to folder: <SAVE_DIR>/analisis semua data/
"""

# ========= 0) CONFIG =========
# Menggunakan file yang sudah diurutkan dari diskusi sebelumnya
DEFAULT_PATH = r"D:\EKG\Skripsi Willy\Data Akhir\Finale\avg_features_data akhir.csv"
BASE_SAVE    = os.path.dirname(DEFAULT_PATH) if os.path.isabs(DEFAULT_PATH) else os.getcwd()
SAVE_DIR     = os.path.join(BASE_SAVE, "analisis_semua_data")  # >>> output folder

# Outlier config
ENABLE_OUTLIER_CLEAN  = True
ROBUST_Z              = 3.5   # |z_robust| > 3.5 -> outlier (MAD-based)
IQR_K                 = 1.5   # fallback IQR fence: [Q1 - 1.5*IQR, Q3 + 1.5*IQR]

# Two-sided normal ranges
NORMAL_RANGES: Dict[str, tuple] = {
    "rr":         (600, 1200),  # back to 600–1200 ms (two-sided)
    "qt":         (350, 440),
    "heartrate":  (60, 120),
}

# Upper-limit only; area below is green
UPPER_THRESH: Dict[str, float] = {
    "qrs": 110.0,   # ms
    "pr":  200.0,   # ms
    "qtc": 460.0,   # ms
}

# Y-axis unit labels
UNITS: Dict[str, str] = {
    "heartrate": "bpm",
    "st_deviation": "mV",
    # default: ms
}

# ========= 1) READ CSV =========
def read_csv_flex(path: str) -> pd.DataFrame:
    for sep in [None, ",", ";", "\t", "|"]:
        try:
            df = pd.read_csv(path, sep=sep, engine="python")
            df = df.loc[:, ~df.columns.astype(str).str.contains("^Unnamed", case=False, regex=True)]
            return df
        except Exception:
            continue
    raise FileNotFoundError(f"Failed to read CSV at: {path} (check delimiter/format).")

# ========= 2) NORMALIZE COLUMNS =========
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rename_map = {c: str(c).strip().lower() for c in df.columns}
    df = df.rename(columns=rename_map)

    alias = {
        "kondisi": "condition",
        "qs": "qrs",
        "st_deviation_mv" : "st_deviation",
        "st_amplitude_mv" : "st_amplitude",
        "heart_rate": "heartrate",
        "hr": "heartrate",
    }
    for old, new in alias.items():
        if old in df.columns and old != new:
            df = df.rename(columns={old: new})

    missing: List[str] = []
    if "subject" not in df.columns:
        missing.append("subject")
    if "condition" not in df.columns:
        missing.append("condition")
    if missing:
        raise KeyError(f"Missing required columns: {missing}. Available: {list(df.columns)}")
    return df

# ========= 3) SORTING & X POSITIONS (MODIFIED) =========
def add_positions(df: pd.DataFrame) -> pd.DataFrame:
    """Sorts dataframe by subject (numerically) and condition, then adds x_pos."""
    df = df.copy()
    # Ekstrak bagian numerik dari 'subject' untuk pengurutan yang benar (S1, S2, ..., S10)
    if 'subject' in df.columns:
        df['subject_numeric'] = df['subject'].str.extract(r'(\d+)', expand=False).astype(int)
        # Urutkan berdasarkan kolom subjek numerik, lalu berdasarkan kondisi
        df = df.sort_values(by=["subject_numeric", "condition"]).reset_index(drop=True)
    else:
        df = df.sort_values(by=["condition"]).reset_index(drop=True)

    # Tetapkan posisi sumbu-x berdasarkan urutan yang benar
    df["x_pos"] = range(len(df))
    return df

# ========= 3.5) OUTLIER CLEANING =========
def _robust_keep_mask(x: pd.Series, z_thresh: float, iqr_k: float) -> pd.Series:
    """True for values that are NOT outliers (NaN is always True so it's kept)."""
    x = pd.to_numeric(x, errors="coerce")
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if not np.isnan(mad) and mad > 0:
        z = 0.6745 * (x - med) / mad
        keep = (np.abs(z) <= z_thresh) | x.isna()
        return keep.astype(bool)
    # fallback IQR
    q1 = np.nanpercentile(x, 25)
    q3 = np.nanpercentile(x, 75)
    iqr = q3 - q1
    if np.isnan(iqr) or iqr == 0:
        return pd.Series(True, index=x.index)  # cannot detect: keep all
    low, high = q1 - iqr_k * iqr, q3 + iqr_k * iqr
    keep = ((x >= low) & (x <= high)) | x.isna()
    return keep.astype(bool)

def clean_outliers_per_feature(df: pd.DataFrame, non_features: set) -> pd.DataFrame:
    if not ENABLE_OUTLIER_CLEAN:
        return df

    df = df.copy()
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in non_features]

    summary = []
    for c in feature_cols:
        x = pd.to_numeric(df[c], errors="coerce")
        keep_mask = _robust_keep_mask(x, ROBUST_Z, IQR_K)
        out_mask  = (~keep_mask) & x.notna()
        n_out = int(out_mask.sum())
        if n_out > 0:
            replacement = x[keep_mask].mean(skipna=True)
            df.loc[out_mask, c] = replacement
        summary.append((c, n_out))
    if any(n_out > 0 for _, n_out in summary):
        print("🧹 Outliers cleaned (replaced by feature-wise mean):")
        for c, n in summary:
            if n > 0:
                print(f"   - {c}: {n} values replaced")
    else:
        print("✅ No outliers to replace (based on thresholds).")
    return df

# ========= 4) NORMAL RANGE SHADING UTILS =========
def _apply_normal_shading(ax, feature: str):
    if feature in UPPER_THRESH:
        hi = UPPER_THRESH[feature]
        ymin, ymax = ax.get_ylim()
        ax.axhspan(ymin, hi, alpha=0.15, color="green")
        ax.axhline(hi, linestyle="--", linewidth=1.5, color="red")
        return

    fr = NORMAL_RANGES.get(feature)
    if fr is not None and isinstance(fr, (list, tuple)) and len(fr) == 2:
        lo, hi = fr
        ax.axhspan(lo, hi, color="green", alpha=0.15)
        ax.axhline(lo, color="red", linestyle="--", linewidth=1.5)
        ax.axhline(hi, color="red", linestyle="--", linewidth=1.5)

def _unit_label(feature: str) -> str:
    return UNITS.get(feature, "ms")

def _safe_name(s: str) -> str:
    return re.sub(r"[^0-9a-zA-Z_\-]+", "_", s)

def _add_subject_xticks(ax, df):
    """Add subject indices (S1, S2, ...) at midpoints along the x-axis, sorted correctly."""
    gp = df.groupby("subject", as_index=False).agg(
        x_pos_min=("x_pos", "min"),
        x_pos_max=("x_pos", "max"),
        subject_numeric=("subject_numeric", "first")
    ).sort_values("subject_numeric")

    mids = ((gp["x_pos_min"] + gp["x_pos_max"]) / 2.0).to_list()
    labels = gp["subject"].to_list()
    ax.set_xticks(mids)
    ax.set_xticklabels(labels, rotation=0, fontsize=10)

# ========= 5) PLOTTING: main figure + per-feature =========
def make_plots(df: pd.DataFrame, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)

    # Tambahkan 'subject_numeric' ke non_features agar tidak di-plot
    non_features = {"subject", "condition", "x_pos", "subject_numeric"}
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    features = [c for c in num_cols if c not in non_features]
    if not features:
        raise ValueError("No numeric feature columns to plot.")

    # subject separators
    subject_sequence = df["subject"].tolist()
    vlines = []
    for i in range(1, len(subject_sequence)):
        if subject_sequence[i] != subject_sequence[i-1]:
            vlines.append(i - 0.5)

    print(f"Detected features: {features}")

    # ------- Figure-level legend -------
    legend_items = [
        Rectangle((0, 0), 1, 1, facecolor="green", alpha=0.15, label="Normal range / ≤ upper limit"),
        Line2D([0], [0], color="red", linestyle="--", linewidth=1.5, label="Limit line"),
    ]

    # (A) MAIN FIGURE (ALL-IN-ONE)
    n = len(features)
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(16, 4*n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, feature in zip(axes, features):
        sns.lineplot(data=df, x="x_pos", y=feature, marker="o", ax=ax, legend=False)
        _apply_normal_shading(ax, feature)

        for x in vlines:
            ax.axvline(x=x, color="grey", linestyle=":", linewidth=1.2)

        ax.set_title(f"Feature: {feature.upper()}", fontsize=13, fontweight="bold")
        ax.set_ylabel(f"Value ({_unit_label(feature)})")
        ax.grid(True, linestyle="--", alpha=0.6)

    axes[-1].set_xlabel("Condition (→ grouped by subject)")
    _add_subject_xticks(axes[-1], df)

    for idx, row in df.iterrows():
        axes[-1].text(row["x_pos"], axes[-1].get_ylim()[0], str(row["condition"]),
                      rotation=90, ha="center", va="bottom", fontsize=8,
                      transform=axes[-1].transData)

    fig.suptitle("Average ECG Features per Subject & Condition — MAIN FIGURE", fontsize=18, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.legend(handles=legend_items, loc="upper right", bbox_to_anchor=(0.985, 0.99))
    main_plot_path = os.path.join(save_dir, "ALL_features_overview.png")
    fig.savefig(main_plot_path, dpi=300, bbox_inches="tight")
    print(f"✅ Main figure saved: {main_plot_path}")
    plt.close(fig)

    # (B) PER-FEATURE FIGURES
    for feature in features:
        fig_f, ax_f = plt.subplots(figsize=(16, 5))
        sns.lineplot(data=df, x="x_pos", y=feature, marker="o", ax=ax_f, legend=False)
        _apply_normal_shading(ax_f, feature)

        for x in vlines:
            ax_f.axvline(x=x, color="grey", linestyle=":", linewidth=1.2)

        ax_f.set_title(f"{feature.upper()} by Subject & Condition", fontsize=15, fontweight="bold")
        ax_f.set_xlabel("Condition (→ grouped by subject)")
        ax_f.set_ylabel(f"Value ({_unit_label(feature)})")
        ax_f.grid(True, linestyle="--", alpha=0.6)
        _add_subject_xticks(ax_f, df)

        ymin, _ = ax_f.get_ylim()
        for idx, row in df.iterrows():
            ax_f.text(row["x_pos"], ymin, str(row["condition"]), rotation=90,
                      ha="center", va="bottom", fontsize=8, transform=ax_f.transData)

        fig_f.legend(handles=legend_items, loc="upper right", bbox_to_anchor=(0.985, 0.99))
        fname = f"FEATURE_{_safe_name(feature)}.png"
        out_path = os.path.join(save_dir, fname)
        fig_f.tight_layout(rect=[0, 0, 1, 0.93])
        fig_f.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig_f)
        print(f"   • Saved: {out_path}")

def main():
    csv_path = DEFAULT_PATH
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]

    if not os.path.exists(csv_path):
        print(f"❌ File not found: {csv_path}. Please make sure the file is in the correct directory.")
        sys.exit(1)

    print(f"📄 Reading: {csv_path}")
    df = read_csv_flex(csv_path)
    print("Detected columns:", list(df.columns))
    
    df = normalize_columns(df)
    
    if ENABLE_OUTLIER_CLEAN:
        # Tentukan kolom non-fitur sebelum cleaning
        non_features = {"subject", "condition"}
        df = clean_outliers_per_feature(df, non_features)

    # Tambahkan posisi plot SETELAH semua data cleaning selesai
    df = add_positions(df)

    print("🔎 Unique conditions (ordered):", sorted(df["condition"].unique()))
    unique_subjects = df[['subject', 'subject_numeric']].drop_duplicates().sort_values('subject_numeric')['subject'].tolist()
    print("🔎 Unique subjects (ordered):", unique_subjects)

    make_plots(df, SAVE_DIR)

if __name__ == "__main__":
    main()
    
plt.show()