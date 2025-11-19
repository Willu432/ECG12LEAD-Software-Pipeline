# perbandingan_kondisi_dan_usia.py
# -*- coding: utf-8 -*-
"""
Plot bar per label (Normal, PFA, PSA) dibagi 3 bucket usia berdasarkan Subject ID:
 - 20–30  : S1–S11
 - 31–50  : S12–S20
 - 50+    : S21–S34

Keterangan (legend):
 - Class (text legend):
     PFA = Potential Fast Arrhythmia
     PSA = Potential Slow Arrhythmia
     Normal = Normal
 - Condition (color legend untuk batang):
     W  = Walking
     S  = Sitting
     LD = Laying Down
     AS = Active Sitting

Sampling:
 - Maksimal 10 baris per (Subject × Condition) sebelum agregasi.

Outputs:
 - CSV : counts_by_age3_label_condition.csv
 - PNG : class_by_condition_age_20_30.png
 - PNG : class_by_condition_age_31_50.png
 - PNG : class_by_condition_age_50_plus.png
 - PNG : class_by_condition_all_ages_combined.png
"""

import os
import re
import itertools
import csv
from typing import Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# =========================
# KONFIGURASI
# =========================
INPUT_CSV  = r"D:\EKG\Skripsi Willy\Data Akhir\Finale\Data Akhir Finale.csv"
OUTPUT_DIR = r"D:\EKG\Skripsi Willy\Data Akhir\Finale\ringkasan_grafik"
FIG_DPI    = 220
# =========================

# Kandidat nama kolom (robust)
SUBJECT_CANDS   = ["subject","Subject","SUBJECT","Name","name","ID","Id","id","Nama","nama"]
CONDITION_CANDS = ["Condition","condition","State","state","Kondisi","kondisi"]
LABEL_CANDS     = ["diagnostic_class","Diagnostic_Class","Diagnostic Class","class","Class","label","Label","pred_label","model_label"]

# Loader robust
DELIMS_TO_TRY    = [None, ",", ";", "\t", "|"]
ENCODINGS_TO_TRY = ["utf-8-sig", "utf-8", "cp1252", "latin1"]

# Penjelasan kelas (legend teks)
CLASS_MAP = {
    "PFA": "Potential Fast Arrhythmia",
    "PSA": "Potential Slow Arrhythmia",
    "Normal": "Normal",
}
# Penjelasan kondisi (legend warna)
CONDITION_MAP = {
    "W":  "Walking",
    "S":  "Sitting",
    "LD": "Laying Down",
    "AS": "Active Sitting",
}

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def load_table_robust(path: str) -> tuple[pd.DataFrame, str, str]:
    """
    Baca file tabel dengan upaya berlapis:
      1) Jika .xlsx/.xls -> read_excel
      2) read_csv sep=None (autodetect) dengan beberapa encoding
      3) Kombinasi encoding × delimiter umum
      4) Terakhir: sniff dengan csv.Sniffer
    Return: (df, used_sep, used_encoding)
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
        return df, "excel", "n/a"

    # 1) Autodetect sep (engine=python) dengan beberapa encoding
    for enc in ENCODINGS_TO_TRY:
        try:
            df = pd.read_csv(path, engine="python", sep=None, encoding=enc)
            if df.shape[1] == 1 and any(ch in str(df.columns[0]) for ch in [",",";","\t","|"]):
                continue
            return df, "auto", enc
        except Exception:
            pass

    # 2) Coba kombinasi encoding × delimiter
    for enc, sep in itertools.product(ENCODINGS_TO_TRY, DELIMS_TO_TRY):
        try:
            df = pd.read_csv(path, engine="python", sep=sep, encoding=enc)
            if df.shape[1] == 1 and any(ch in str(df.columns[0]) for ch in [",",";","\t","|"]):
                continue
            return df, (sep if sep is not None else "auto"), enc
        except Exception:
            pass

    # 3) Fallback: sniff delimiter
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            sample = f.read(4096)
        dialect = csv.Sniffer().sniff(sample, delimiters=";,|\t")
        sep = dialect.delimiter
        df = pd.read_csv(path, engine="python", sep=sep, encoding="utf-8", on_bad_lines="skip")
        return df, sep, "utf-8 (sniffed)"
    except Exception as e:
        raise RuntimeError(f"Gagal membaca file tabel: {path}. Detail: {e}")

def pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def extract_subject_number(s) -> Optional[int]:
    m = re.search(r"(\d+)", str(s))
    return int(m.group(1)) if m else None

def age_bucket_from_subject(s: str) -> str:
    """
    Mapping bucket usia berdasar ID Subject:
      - 20–30 : S1–S11
      - 31–50 : S12–S20
      - 50+   : S21–S34
    """
    n = extract_subject_number(s)
    if n is None:
        return "Unknown"
    if 1 <= n <= 11:
        return "20–30"
    if 12 <= n <= 20:
        return "31–50"
    if 21 <= n <= 34:
        return "50+"
    return "Unknown"

def sample_max_10_per_subject_condition(df: pd.DataFrame, subject_col: str, condition_col: str) -> pd.DataFrame:
    """Ambil maksimal 10 baris per (Subject × Condition)."""
    return (df.groupby([subject_col, condition_col], group_keys=False)
              .apply(lambda g: g.head(10))
              .reset_index(drop=True))

def plot_one_bucket(grp_full: pd.DataFrame, bucket: str,
                    all_labels: list[str], all_conditions: list[str], out_png: str):
    # Warna konsisten untuk tiap kondisi
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", None)
    cond_to_color = {cond: color_cycle[i % len(color_cycle)] for i, cond in enumerate(all_conditions)} if color_cycle else {}

    fig_w = max(10, len(all_labels) * 0.6)
    fig_h = 6
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    bar_width = 0.8 / max(1, len(all_conditions))
    x_base = np.arange(len(all_labels))

    sub = grp_full[grp_full["_AgeBucket"] == bucket]

    # Batang per kondisi
    for j, cond in enumerate(all_conditions):
        vals = sub[sub["_Condition"] == cond].set_index("_Label").reindex(all_labels)["count"].fillna(0).values
        x_pos = x_base + (j - (len(all_conditions)-1)/2.0) * bar_width
        cond_label = CONDITION_MAP.get(cond, cond)  # legenda kondisi (W/S/LD/AS -> penjelasan)
        ax.bar(x_pos, vals, width=bar_width, label=cond_label, color=cond_to_color.get(cond, None))

    ax.set_xticks(x_base)
    ax.set_xticklabels(all_labels, rotation=45, ha="right")
    ax.set_xlabel(f"Classification (Age {bucket})")
    ax.set_ylabel("Total Records")
    ax.set_title(f"Count by Label • Age {bucket}")

    # Legend 1: kondisi (warna)
    leg1 = ax.legend(title="Condition", frameon=False, loc="upper right")

    # Legend 2: kelas (teks)
    class_entries = [f"{k} = {v}" for k, v in CLASS_MAP.items()]
    class_handles = [Patch(facecolor='none', edgecolor='none', label=entry) for entry in class_entries]
    leg2 = ax.legend(handles=class_handles, loc="upper left", frameon=False, title="Class (Explanation)")
    ax.add_artist(leg1)

    plt.tight_layout()
    plt.savefig(out_png, dpi=FIG_DPI)
    plt.close()

def plot_combined(grp_full: pd.DataFrame, all_labels: list[str],
                  all_conditions: list[str], out_png: str):
    """Gabungkan tiga bucket dalam satu gambar."""
    buckets = ["20–30", "31–50", "50+"]
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", None)
    cond_to_color = {cond: color_cycle[i % len(color_cycle)] for i, cond in enumerate(all_conditions)} if color_cycle else {}

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    bar_width = 0.8 / max(1, len(all_conditions))
    x_base = np.arange(len(all_labels))

    for ax, bucket in zip(axes, buckets):
        sub = grp_full[grp_full["_AgeBucket"] == bucket]
        for j, cond in enumerate(all_conditions):
            vals = sub[sub["_Condition"] == cond].set_index("_Label").reindex(all_labels)["count"].fillna(0).values
            x_pos = x_base + (j - (len(all_conditions)-1)/2.0) * bar_width
            cond_label = CONDITION_MAP.get(cond, cond)
            ax.bar(x_pos, vals, width=bar_width, label=cond_label, color=cond_to_color.get(cond, None))

        ax.set_xticks(x_base)
        ax.set_xticklabels(all_labels, rotation=45, ha="right")
        ax.set_xlabel(f"Classification (Age {bucket})")
        ax.set_title(f"Age {bucket}")

    axes[0].set_ylabel("Total Records")

    # Legend kondisi (global)
    handles_c, labels_c = axes[0].get_legend_handles_labels()
    fig.legend(handles_c, labels_c, title="Condition", loc="upper center",
               ncol=min(4, len(labels_c)), frameon=False)

    # Legend kelas (teks)
    class_entries = [f"{k} = {v}" for k, v in CLASS_MAP.items()]
    class_handles = [Patch(facecolor='none', edgecolor='none', label=entry) for entry in class_entries]
    fig.legend(class_handles, [h.get_label() for h in class_handles],
               loc="lower center", ncol=3, frameon=False, title="Class (Explanation)")

    plt.tight_layout(rect=[0, 0.08, 1, 0.90])
    plt.savefig(out_png, dpi=FIG_DPI)
    plt.close()

def main():
    # ---- Load ----
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"File tidak ditemukan: {INPUT_CSV}")
    ensure_dir(OUTPUT_DIR)

    print(f"📂 Membaca: {INPUT_CSV}")
    df, used_sep, used_enc = load_table_robust(INPUT_CSV)
    print(f"✅ Terbaca: {df.shape[0]} baris × {df.shape[1]} kolom  | sep={used_sep}  enc={used_enc}")
    print(f"🧪 Kolom awal: {list(df.columns)[:8]}{' ...' if df.shape[1]>8 else ''}")

    # ---- Deteksi kolom ----
    subject_col   = pick_col(df, SUBJECT_CANDS)
    condition_col = pick_col(df, CONDITION_CANDS)
    label_col     = pick_col(df, LABEL_CANDS)
    if not all([subject_col, condition_col, label_col]):
        raise ValueError(f"Kolom wajib tidak ditemukan. Ditemukan: {list(df.columns)}")

    df["_Subject"]   = df[subject_col].astype(str)
    df["_Condition"] = df[condition_col].astype(str)
    df["_Label"]     = df[label_col].astype(str)

    # ---- Limit: max 10 baris per (Subject × Condition) ----
    df_limited = sample_max_10_per_subject_condition(df, "_Subject", "_Condition")
    print(f"🔎 Setelah pembatasan: {df_limited.shape[0]} baris")

    # ---- Bucket usia ----
    df_limited["_AgeBucket"] = df_limited["_Subject"].apply(age_bucket_from_subject)
    buckets = ["20–30", "31–50", "50+"]
    df_limited = df_limited[df_limited["_AgeBucket"].isin(buckets)].copy()

    # ---- Agregasi: by Age, Label, Condition ----
    grp = df_limited.groupby(["_AgeBucket", "_Label", "_Condition"]).size().reset_index(name="count")

    all_labels = sorted(df_limited["_Label"].unique())
    all_conditions = sorted(df_limited["_Condition"].unique())

    # Pastikan semua kombinasi ada (isi 0 jika kosong)
    idx = pd.MultiIndex.from_product([buckets, all_labels, all_conditions],
                                     names=["_AgeBucket", "_Label", "_Condition"])
    grp_full = grp.set_index(["_AgeBucket", "_Label", "_Condition"]).reindex(idx, fill_value=0).reset_index()

    # ---- Simpan CSV ringkasan ----
    out_csv = os.path.join(OUTPUT_DIR, "counts_by_age3_label_condition.csv")
    grp_full.to_csv(out_csv, index=False)
    print(f"💾 Disimpan: {out_csv}")

    # ---- Plot per bucket & gabungan ----
    out_20 = os.path.join(OUTPUT_DIR, "class_by_condition_age_20_30.png")
    out_31 = os.path.join(OUTPUT_DIR, "class_by_condition_age_31_50.png")
    out_50 = os.path.join(OUTPUT_DIR, "class_by_condition_age_50_plus.png")
    out_all = os.path.join(OUTPUT_DIR, "class_by_condition_all_ages_combined.png")

    plot_one_bucket(grp_full, "20–30", all_labels, all_conditions, out_20)
    plot_one_bucket(grp_full, "31–50", all_labels, all_conditions, out_31)
    plot_one_bucket(grp_full, "50+",   all_labels, all_conditions, out_50)
    plot_combined(grp_full, all_labels, all_conditions, out_all)

    print("✅ Selesai. File gambar:")
    print(f" - {out_20}\n - {out_31}\n - {out_50}\n - {out_all}")

if __name__ == "__main__":
    main()
