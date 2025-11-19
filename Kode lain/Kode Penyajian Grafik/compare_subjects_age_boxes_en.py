# compare_subjects_age_boxes_en.py
# -*- coding: utf-8 -*-
"""
Compare subject conditions, counts, and labels.
Add faded colorboxes behind bars to mark:
- S1–S11  (Age 20–30)
- S12–S20 (Age 31–50)
- S21–S34 (Age 50+)

Legenda label (disamakan dengan skrip lain):
- PFA  -> Potential Fast Arrhythmia
- PSA  -> Potential Slow Arrhythmia
- Normal -> Normal

Outputs:
- CSV: summary_per_subject.csv, summary_per_age_group.csv
- PNG: bar_count_per_subject.png, stacked_labels_per_subject.png
"""

import os
import re
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# EDIT YOUR PATHS HERE
# =========================
INPUT_CSV  = r"D:\EKG\Skripsi Willy\Data Akhir\Finale\Data Akhir Finale.csv"
OUTPUT_DIR = r"D:\EKG\Skripsi Willy\Data Akhir\Finale\ringkasan_grafik"
FIG_DPI    = 200
# =========================

# Column name candidates (robust to naming variations)
SUBJECT_CANDS   = ["subject","Subject","SUBJECT","Nama","nama","ID","Id","id","Name","name"]
CONDITION_CANDS = ["Kondisi","kondisi","condition","Condition","state","State"]
LABEL_CANDS     = ["diagnostic_class","Diagnostic_Class","Diagnostic Class","class","Class","label","Label","pred_label","model_label"]

DELIMS_TO_TRY    = [";", ",", "\t", None]
ENCODINGS_TO_TRY = ["utf-8-sig", "utf-8", "latin1", "cp1252"]

# Pemetaan label -> legenda yang ditampilkan
LABEL_MAP = {
    "PFA": "Potential Fast Arrhythmia",
    "PSA": "Potential Slow Arrhythmia",
    "Normal": "Normal"
}
# Urutan label yang diutamakan pada legend/stack
LABEL_ORDER = ["PFA", "PSA", "Normal"]
# Warna konsisten untuk label (opsional; boleh diubah)
LABEL_COLOR_MAP = {
    "PFA":    None,  # gunakan default color cycle jika None
    "PSA":    None,
    "Normal": None
}

def load_csv_robust(path: str) -> pd.DataFrame:
    """Read CSV trying several delimiters & encodings."""
    last_err = None
    for enc in ENCODINGS_TO_TRY:
        for sep in DELIMS_TO_TRY:
            try:
                df = pd.read_csv(path, sep=sep, engine="python", encoding=enc)
                # if read into single column but header hints delimiter, try next sep
                if df.shape[1] == 1 and any(ch in str(df.columns[0]) for ch in [",",";","\t"]):
                    continue
                return df
            except Exception as e:
                last_err = e
    raise RuntimeError(f"Failed to read CSV '{path}'. Last error: {last_err}")

def pick_col(df: pd.DataFrame, candidates) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def first_col_index(df: pd.DataFrame, exact_name: str) -> int | None:
    matches = [i for i, c in enumerate(df.columns) if c == exact_name]
    return matches[0] if matches else None

def extract_subject_number(s) -> int | None:
    m = re.search(r"(\d+)", str(s))
    return int(m.group(1)) if m else None

def age_group_from_subject(s: str) -> str:
    """Bucket usia disamakan dengan skrip lain:
       - 20–30 : S1–S11
       - 31–50 : S12–S20
       - 50+   : S21–S34
    """
    n = extract_subject_number(s)
    if n is None:
        return "Unknown"
    if 1 <= n <= 11:
        return "S1–S11 (Age 20–30)"
    if 12 <= n <= 20:
        return "S12–S20 (Age 31–50)"
    if 21 <= n <= 34:
        return "S21–S34 (Age 50+)"
    return "Unknown"

def find_range(unique_subjects: list[str], start_n: int, end_n: int, subj_to_x: dict[str, int]):
    xs = []
    for s in unique_subjects:
        n = extract_subject_number(s)
        if n is not None and start_n <= n <= end_n:
            xs.append(subj_to_x[s])
    if not xs:
        return None
    return min(xs) - 0.5, max(xs) + 0.5

def sample_max_10_per_subject_condition(df: pd.DataFrame, subject_col: str, condition_col: str) -> pd.DataFrame:
    """(Opsional) Batasi maksimal 10 baris per (Subject × Condition)."""
    return (df.groupby([subject_col, condition_col], group_keys=False)
              .apply(lambda g: g.head(10))
              .reset_index(drop=True))

def main():
    # ---- Load ----
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Not found: {INPUT_CSV}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"📂 Reading: {INPUT_CSV}")
    df = load_csv_robust(INPUT_CSV)
    print(f"✅ Shape: {df.shape[0]} rows × {df.shape[1]} cols")

    # ---- Detect columns (handle duplicates) ----
    subject_col = pick_col(df, SUBJECT_CANDS)
    condition_col = pick_col(df, CONDITION_CANDS)
    label_col = pick_col(df, LABEL_CANDS)

    if subject_col:
        s_idx = first_col_index(df, subject_col)
        df["_subject_unq"] = df.iloc[:, s_idx].astype(str)
    else:
        df["_subject_unq"] = [f"Row {i}" for i in range(len(df))]

    if condition_col:
        c_idx = first_col_index(df, condition_col)
        df["_cond_unq"] = df.iloc[:, c_idx].astype(str)
    else:
        df["_cond_unq"] = np.nan

    if label_col:
        l_idx = first_col_index(df, label_col)
        df["_label_unq"] = df.iloc[:, l_idx].astype(str)
    else:
        df["_label_unq"] = "Unlabeled"

    # (Opsional) samakan pembatasan sampel seperti skrip lain:
    # df = sample_max_10_per_subject_condition(df, "_subject_unq", "_cond_unq")

    # ---- Subject order (based on S-number if present) ----
    df["_subj_num"] = df["_subject_unq"].apply(extract_subject_number)
    unique_subjects = (
        df[["_subject_unq", "_subj_num"]]
        .drop_duplicates(subset=["_subject_unq"])
        .sort_values(by=["_subj_num", "_subject_unq"], na_position="last")
        ["_subject_unq"]
        .tolist()
    )
    subj_to_x = {s: i for i, s in enumerate(unique_subjects)}

    # ---- Age group per subject ----
    df["_age_group"] = df["_subject_unq"].apply(age_group_from_subject)

    # ---- Per-subject summary ----
    counts_per_subject = df.groupby("_subject_unq").size().rename("n_rows")
    mode_condition = df.groupby("_subject_unq")["_cond_unq"].agg(
        lambda s: Counter(s.dropna()).most_common(1)[0][0] if len(s.dropna()) > 0 else np.nan
    )
    label_counts_tbl = df.pivot_table(
        index="_subject_unq", columns="_label_unq",
        values=df.columns[0], aggfunc="count", fill_value=0
    )
    age_group_map = (
        df.drop_duplicates(subset=["_subject_unq"])
          .set_index("_subject_unq")["_age_group"]
    )

    summary_subject = pd.concat(
        [age_group_map.rename("age_group"), counts_per_subject, mode_condition.rename("mode_condition"), label_counts_tbl],
        axis=1
    ).reset_index().rename(columns={"_subject_unq": "subject"})

    # ---- Per-age-group summary ----
    summary_group_counts = df.groupby("_age_group").size().rename("n_rows").reset_index()
    summary_group_labels = df.pivot_table(
        index="_age_group", columns="_label_unq",
        values=df.columns[0], aggfunc="count", fill_value=0
    ).reset_index()
    summary_group = pd.merge(summary_group_counts, summary_group_labels, on="_age_group", how="outer") \
                       .rename(columns={"_age_group": "age_group"})

    # ---- Save summaries ----
    subj_csv = os.path.join(OUTPUT_DIR, "summary_per_subject.csv")
    group_csv = os.path.join(OUTPUT_DIR, "summary_per_age_group.csv")
    summary_subject.to_csv(subj_csv, index=False)
    summary_group.to_csv(group_csv, index=False)
    print(f"💾 Saved: {subj_csv}")
    print(f"💾 Saved: {group_csv}")

    # ====== FIGURE 1: counts per subject + faded age-group colorboxes ======
    x = np.arange(len(unique_subjects))
    vals = counts_per_subject.reindex(unique_subjects).values

    fig_w = max(12, len(unique_subjects) * 0.6)
    fig_h = 5
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Ranges for colorboxes (S1–S11, S12–S20, S21–S34)
    rng1 = find_range(unique_subjects, 1, 11, subj_to_x)
    rng2 = find_range(unique_subjects, 12, 20, subj_to_x)
    rng3 = find_range(unique_subjects, 21, 34, subj_to_x)

    if rng1 is not None:
        ax.axvspan(rng1[0], rng1[1], alpha=0.10, hatch="//", label="S1–S11 • Age 20–30", color="tab:blue")
    if rng2 is not None:
        ax.axvspan(rng2[0], rng2[1], alpha=0.10, hatch="\\\\", label="S12–S20 • Age 31–50", color="tab:orange")
    if rng3 is not None:
        ax.axvspan(rng3[0], rng3[1], alpha=0.10, hatch="..", label="S21–S34 • Age 50+", color="tab:green")

    ax.bar(x, vals)
    ax.set_xticks(x)
    ax.set_xticklabels(unique_subjects, rotation=90)
    ax.set_ylabel("Number of rows")
    ax.set_xlabel("Subject")
    ax.set_title("Records per Subject\nFaded colorboxes: Age groups")
    ax.legend(loc="upper right", ncol=1, frameon=False)

    plt.tight_layout()
    out1 = os.path.join(OUTPUT_DIR, "bar_count_per_subject.png")
    plt.savefig(out1, dpi=FIG_DPI)
    print(f"💾 Saved: {out1}")
    plt.close(fig)

    # ====== FIGURE 2: stacked label distribution per subject + same colorboxes ======
    # Pastikan urutan kolom label sesuai urutan prioritas
    label_cols_all = list(label_counts_tbl.columns)
    # Susun: LABEL_ORDER dulu (yang ada), sisanya menyusul
    label_cols = [c for c in LABEL_ORDER if c in label_cols_all] + [c for c in label_cols_all if c not in LABEL_ORDER]
    label_counts_ordered = label_counts_tbl.reindex(unique_subjects).fillna(0)

    # Warna: siapkan color cycle default, kemudian override dengan LABEL_COLOR_MAP jika diset
    default_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    color_map = {}
    for i, lab in enumerate(label_cols):
        color_map[lab] = LABEL_COLOR_MAP.get(lab, default_colors[i % len(default_colors)] if default_colors else None)

    fig2, ax2 = plt.subplots(figsize=(fig_w, fig_h))

    if rng1 is not None:
        ax2.axvspan(rng1[0], rng1[1], alpha=0.10, hatch="//", label="S1–S11 • Age 20–30", color="tab:blue")
    if rng2 is not None:
        ax2.axvspan(rng2[0], rng2[1], alpha=0.10, hatch="\\\\", label="S12–S20 • Age 31–50", color="tab:orange")
    if rng3 is not None:
        ax2.axvspan(rng3[0], rng3[1], alpha=0.10, hatch="..", label="S21–S34 • Age 50+", color="tab:green")

    bottoms = np.zeros(len(unique_subjects))
    for lab in label_cols:
        vals_lab = label_counts_ordered[lab].reindex(unique_subjects).values if lab in label_counts_ordered.columns else np.zeros(len(unique_subjects))
        display_name = LABEL_MAP.get(lab, str(lab))
        ax2.bar(x, vals_lab, bottom=bottoms, label=display_name, color=color_map.get(lab, None))
        bottoms = bottoms + vals_lab

    ax2.set_xticks(x)
    ax2.set_xticklabels(unique_subjects, rotation=90)
    ax2.set_ylabel("Count per label")
    ax2.set_xlabel("Subject")
    ax2.set_title("Label Distribution per Subject\nFaded colorboxes: Age groups")
    # Untuk menghindari duplikasi label colorbox + stack legend, gabungkan unik
    handles, labels = ax2.get_legend_handles_labels()
    # Prioritaskan legend label kelas (PFA/PSA/Normal) muncul dulu
    order = []
    for nm in [LABEL_MAP.get(k, k) for k in LABEL_ORDER]:
        if nm in labels:
            order.append(labels.index(nm))
    # Tambahkan sisanya (colorboxes & label lain)
    order += [i for i, nm in enumerate(labels) if i not in order]
    handles = [handles[i] for i in order]
    labels = [labels[i] for i in order]
    ax2.legend(handles, labels, loc="upper right", ncol=1, frameon=False)

    plt.tight_layout()
    out2 = os.path.join(OUTPUT_DIR, "stacked_labels_per_subject.png")
    plt.savefig(out2, dpi=FIG_DPI)
    print(f"💾 Saved: {out2}")
    plt.close(fig2)

    print("✅ Done.")

if __name__ == "__main__":
    main()
