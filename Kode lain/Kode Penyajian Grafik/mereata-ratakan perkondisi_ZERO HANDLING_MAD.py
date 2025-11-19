# ========= avg_per_subject_condition_autosplit.py =========
import os, re
import numpy as np
import pandas as pd

# --- KONFIGURASI ---
CSV_PATH  = r"D:\EKG\Skripsi Willy\Data Akhir\Finale\Data Akhir Finale.csv"
OUT_PATH  = r"D:\EKG\Skripsi Willy\Data Akhir\Finale\avg_features_data akhir.csv"

# Pengaturan outlier
REMOVE_OUTLIERS       = True   # set False jika tidak ingin membuang outlier
ROBUST_Z_THRESH       = 3.5    # ambang |z_robust| > 3.5 dianggap outlier
IQR_K                 = 1.5    # fallback: di luar [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
MIN_GROUP_SIZE_FILTER = 5      # kelompok (subject, condition) dengan jumlah < 5: lewati filter outlier
# -------------------

def read_csv_flex(path: str) -> pd.DataFrame:
    for sep in [None, ",", ";", "\t", "|"]:
        try:
            df = pd.read_csv(path, sep=sep, engine="python")
            df = df.loc[:, ~df.columns.astype(str).str.contains("^Unnamed", case=False, regex=True)]
            return df
        except Exception:
            continue
    raise RuntimeError(f"Gagal membaca CSV: {path}")

def cols_by_regex(df: pd.DataFrame, patterns: list[str]) -> list[str]:
    """Pilih kolom yang cocok salah satu pola regex (case-insensitive)."""
    out = []
    for c in df.columns:
        lc = str(c).lower()
        if any(re.match(p, lc) for p in patterns):
            out.append(c)
    return out

# Sertakan 'B' (Baring) jika ada di data
PAT_COND = re.compile(r"^(?P<subject>.+?)_(?P<condition>DB|D|T|J|B)$", flags=re.IGNORECASE)

def try_split_subject_condition(df: pd.DataFrame) -> pd.DataFrame:
    cols = list(df.columns)

    # Normalisasi 'subject' (handle BOM)
    subject_col   = next((c for c in cols if str(c).lower().strip("\ufeff") == "subject"), None)
    condition_col = next((c for c in cols if str(c).lower() == "condition"), None)

    # 1) Sudah ada subject & condition
    if subject_col and condition_col:
        df = df.rename(columns={subject_col: "subject", condition_col: "condition"})
        df["condition"] = df["condition"].astype(str).str.upper().str.strip()
        return df

    # 2) Hanya subject gabungan -> pecah
    if subject_col and not condition_col:
        df = df.rename(columns={subject_col: "subject"})
        subjs, conds = [], []
        for val in df["subject"].astype(str):
            m = PAT_COND.match(val.strip())
            if m:
                subjs.append(m.group("subject"))
                conds.append(m.group("condition").upper())
            else:
                subjs.append(val)
                conds.append("UNKNOWN")
        df["subject"] = subjs
        df["condition"] = conds
        return df

    # 3) Cari kolom gabungan lain
    for c in cols:
        s = df[c].astype(str).head(50).fillna("")
        if s.apply(lambda x: bool(PAT_COND.match(x.strip()))).mean() >= 0.6:
            subjs, conds = [], []
            for val in df[c].astype(str):
                m = PAT_COND.match(val.strip())
                if m:
                    subjs.append(m.group("subject"))
                    conds.append(m.group("condition").upper())
                else:
                    subjs.append(val)
                    conds.append("UNKNOWN")
            df.insert(0, "subject", subjs)
            df.insert(1, "condition", conds)
            return df

    raise RuntimeError(
        "Tidak menemukan kolom 'subject'/'condition' dan juga tidak ada kolom gabungan "
        "(mis. 'Umar_D'). Pastikan ada kolom yang berisi pola '<nama>_D|T|J|DB|B'."
    )

def robust_outlier_mask(series: pd.Series) -> pd.Series:
    """
    True jika BUKAN outlier.
    Metode utama: robust Z-score berbasis MAD.
    Fallback: IQR ketika MAD==0 atau semua sama.
    """
    x = series.astype(float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))

    if np.isnan(med) or np.isnan(mad):
        # tidak bisa dihitung -> anggap semua valid
        return pd.Series(True, index=series.index)

    if mad > 0:
        z = 0.6745 * (x - med) / mad  # 0.6745 ~ normalizing constant
        keep = np.abs(z) <= ROBUST_Z_THRESH
        # NaN tetap dipertahankan (tidak dihitung outlier)
        keep = keep | x.isna()
        return keep.astype(bool)

    # Fallback IQR
    q1 = np.nanpercentile(x, 25)
    q3 = np.nanpercentile(x, 75)
    iqr = q3 - q1
    if iqr == 0 or np.isnan(iqr):
        return pd.Series(True, index=series.index)
    low  = q1 - IQR_K * iqr
    high = q3 + IQR_K * iqr
    keep = (x >= low) & (x <= high) | x.isna()
    return keep.astype(bool)

def filter_outliers_per_group(df_features: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    Buang baris outlier per (subject, condition) berdasarkan kumpulan feature_cols.
    Jika satu baris outlier pada salah satu fitur -> baris tersebut dibuang.
    """
    if not REMOVE_OUTLIERS:
        return df_features

    kept_parts = []
    removed_total = 0

    for (subj, cond), grp in df_features.groupby(["subject", "condition"], dropna=False):
        n = len(grp)
        if n < MIN_GROUP_SIZE_FILTER:
            kept_parts.append(grp)  # lewati filter kalau sample terlalu sedikit
            continue

        mask_all_keep = pd.Series(True, index=grp.index)
        for c in feature_cols:
            if c not in grp.columns:
                continue
            mask_keep_c = robust_outlier_mask(grp[c])
            mask_all_keep &= mask_keep_c

        removed = (~mask_all_keep).sum()
        removed_total += int(removed)
        kept_parts.append(grp.loc[mask_all_keep])

    out = pd.concat(kept_parts, axis=0).sort_index()
    print(f"🔎 Outlier removal: {removed_total} baris dihapus dari total {len(df_features)}.")
    return out

def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(CSV_PATH)

    df = read_csv_flex(CSV_PATH)
    df = try_split_subject_condition(df)

    # Label verbose opsional
    MAP_VERBOSE = {"LD": "Laying Down", "S": "Sitting", "W": "Walking", "AS": "Active Sitting"}
    df["condition_verbose"] = df["condition"].map(MAP_VERBOSE).fillna(df["condition"])

    # Pola regex per fitur (anti-konflik rr vs rr_std)
    feature_specs_regex = {
        "rr":        [r"^rr_(?!std|stdev|stddev)"],
        "rrstd":     [r"^rr_(?:std|stdev|stddev)_"],
        "qrs":       [r"^qs_", r"^qrs_"],
        "pr":        [r"^pr_"],
        "qtc":       [r"^qtc_|^qt_c_|^qt_corrected_"],
        "st_amplitude_mv":        [r"^st_amplitude_mv"],
        "st_deviation_mv":    [r"^st_deviation_mv"],
        "rs_ratio":  [r"^rs_ratio_|^r_s_ratio_"],
        "heartrate": [r"^heartrate_|^hr_"],
    }

    # Kumpulkan feature per baris (record) -> lalu filter outlier per (subject, condition)
    agg_df = df[["subject", "condition"]].copy()

    built_feature_cols = []  # nama kolom fitur hasil agregasi per baris
    for out_name, patterns in feature_specs_regex.items():
        cols = cols_by_regex(df, patterns)
        if not cols:
            agg_df[out_name] = np.nan
            built_feature_cols.append(out_name)
            continue

        sub = df[cols].apply(pd.to_numeric, errors="coerce")

        # Hitung mean per baris dari nilai valid (non-zero)
        valid = (sub != 0) & sub.notna()
        row_means = sub.where(valid).mean(axis=1, skipna=True)

        # IMPUTASI 0 per kolom agar align by index (hindari broadcasting error)
        imputed = sub.copy()
        for c in imputed.columns:
            mask_zero = (imputed[c] == 0) & row_means.notna()
            imputed.loc[mask_zero, c] = row_means[mask_zero]

        # Mean lintas lead per baris (setelah imputasi)
        agg_df[out_name] = imputed.mean(axis=1, skipna=True)
        built_feature_cols.append(out_name)

    # --- Buang outlier per (subject, condition) berdasarkan fitur yang tersedia ---
    agg_df_filtered = filter_outliers_per_group(agg_df, built_feature_cols)

    # --- Mean per (subject, condition) setelah outlier dibuang ---
    result = (
        agg_df_filtered
        .groupby(["subject", "condition"], as_index=False)
        .agg({k: "mean" for k in feature_specs_regex.keys()})
        .sort_values(["subject", "condition"])
    )

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    result.to_csv(OUT_PATH, index=False)
    print("✅ Tersimpan ke:", OUT_PATH)
    print("🧾 Shape:", result.shape)
    print(result.head(10))

if __name__ == "__main__":
    main()
