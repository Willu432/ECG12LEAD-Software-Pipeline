
import os, glob, math, sys, json
import numpy as np
import pandas as pd
import scipy.signal
import neurokit2 as nk
import joblib
import pywt
import xgboost as xgb 

# ====== KONFIGURASI ======
MAIN_DIR = r"D:\EKG\Skripsi Willy\Pengukuran Lansia"   # folder utama (berisi subjek)
OUT_CSV  = os.path.join(MAIN_DIR, "Hasil_Batch_fitur_Terfilter_Data lansia.csv")

# Path model
SCALER_PATH   = r"D:\EKG\Skripsi Willy\Model paling bagus\Model Final\scaler.joblib"
XGB_PATH      = r"D:\EKG\Skripsi Willy\Model paling bagus\Model Final\model_xgb_binary.json"
KMEANS_PATH   = r"D:\EKG\Skripsi Willy\Model paling bagus\Model Final\kmeans_abnormal_model.joblib"

# Nama kolom / lead
LEADS = ["I","II","III","AVR","AVL","AVF","V1","V2","V3","V4","V5","V6"]
FEATURE_FAMILIES = ['rr', 'rr_std', 'pr', 'qrs', 'qtc', 'st_amplitude_mv','st_deviation_mv', 'rs_ratio', 'heartrate']
EXPECTED_FEATURES = [f"{fam}_{lead.lower()}" for lead in LEADS for fam in FEATURE_FAMILIES]

# Monkey‐patch internal module name (mengikuti kode kamu)
sys.modules['numpy._core'] = np.core

# ====== LOAD MODEL ======
scaler          = joblib.load(SCALER_PATH)
# Inisialisasi model kosong terlebih dahulu
model_xgb = xgb.XGBClassifier() 
# Muat model dari file .json
model_xgb.load_model(XGB_PATH) 
kmeans_abnormal = joblib.load(KMEANS_PATH)

def predict_clusters_xgb_integrated(new_data, xgb_binary_model, kmeans_abnormal,
                                    abnormal_label_offset=1):
    """
    new_data: np.ndarray shape (n_samples, n_features), sudah diskalakan.
    Mengembalikan np.ndarray string label:
      0 -> "Normal"
      2 -> "Potential Slow Arrhytmia"
      1 -> "Potential Fast Arrhytmia"
    """
    # 1. Prediksi Normal(1) vs Abnormal(0)
    is_normal = xgb_binary_model.predict(new_data)

    # 2. Buat array integer awal
    labels_int = np.full(len(new_data), -999, dtype=int)

    # 3. Sampel normal → label 0
    idx_norm = np.where(is_normal == 1)[0]
    labels_int[idx_norm] = 0

    # 4. Sampel abnormal → kluster + offset
    idx_abn = np.where(is_normal == 0)[0]
    if len(idx_abn):
        abn_data     = new_data[idx_abn]
        cls_nums     = kmeans_abnormal.predict(abn_data) + abnormal_label_offset
        labels_int[idx_abn] = cls_nums

    # 5. Mapping ke nama
    name_map = {
        0: "Normal",
        1: "Potential Fast Arrhytmia",
        2: "Potential Slow Arrhytmia"
    }
    labels_str = np.array([name_map.get(i, f"Cluster {i}") for i in labels_int])

    # 6. Validasi
    if np.any(labels_int == -999):
        print("⚠️ Beberapa sampel belum terlabel!")

    return labels_str

# ================================ PROSES DATA =================================
# # <<< ADD: constants konversi ADC → mV (sama seperti real-time)
# VREF = 2.42       # volt
# GAIN = 1       # sesuai setelan ADS1293
# RESOLUTION = 24
# LSB_SIZE = VREF / (2**RESOLUTION - 1)

# def to_mV(adc):
#     return adc * LSB_SIZE * 1000.0 / GAIN

# ================================================================================

def to_mV(ecg_adc: int) -> float:
    """
    Mengonversi nilai ADC mentah 24-bit dari ADS1293 ke milivolt (mV).

    Fungsi ini menggunakan konstanta yang spesifik untuk konfigurasi perangkat keras Anda:
    - VREF: 2.4V (referensi internal tipikal)
    - GAIN: 3.5x (gain tetap dari Instrumentation Amplifier internal)
    - ADC_MAX: 12149900.0 (nilai spesifik dari Tabel 8 datasheet untuk R1=4, R2=5, R3=6, dan SDM 102.4 kHz).
    
    Rumus diadaptasi dari Persamaan 13 di datasheet ADS1293.

    Args:
        adc_code: Nilai mentah integer 32-bit (signed) yang dibaca dari register data ECG.

    Returns:
        Nilai tegangan dalam satuan milivolt (mV) sebagai float.
    """
    # --- Konstanta berdasarkan datasheet dan konfigurasi Anda ---
    VREF = 2.4
    GAIN = 3.5
    ADC_MAX = 12149900.0  # Sesuai dengan R2=5, R3=6 pada SDM 102.4kHz

    # --- Rumus Konversi ---
    # Rumus dasar: Voltage (V) = ((adc_code / ADC_MAX) - 0.5) * (2 * VREF / GAIN)
    voltage_in_volts = ((ecg_adc / ADC_MAX) - 0.5) * (2.0 * VREF / GAIN)
    
    # Konversi dari Volt ke miliVolt
    return voltage_in_volts * 1000.0

# <<< ADD: deteksi & perbaikan spike
def despike_hampel(x, k=7, nsigma=6.0):
    """
    Hampel filter di domain waktu untuk mendeteksi outlier tajam.
    k: half-window (total window = 2k+1)
    nsigma: threshold berbasis MAD
    """
    x = np.asarray(x, dtype=float)
    med = scipy.signal.medfilt(x, kernel_size=2*k+1)
    diff = np.abs(x - med)
    # MAD yang robust
    mad = scipy.signal.medfilt(diff, kernel_size=2*k+1)
    mad = np.maximum(mad, 1e-12)             # cegah div/zero
    mask = diff > (nsigma * 1.4826 * mad)    # 1.4826 ≈ k MAD→σ
    # interpolasi linear di sampel outlier
    xi = x.copy()
    if np.any(mask):
        idx = np.arange(len(x))
        xi[mask] = np.interp(idx[mask], idx[~mask], x[~mask])
    return xi, mask

# <<< ADD: deteksi saturasi dekat tepi ADC (±FS)
ADC_EDGE = 0.95 * (2**23 - 1)  # 95% full-scale 24-bit signed
def desaturate_edges(x):
    x = np.asarray(x, dtype=float)
    mask = (np.abs(x) >= ADC_EDGE)
    xi = x.copy()
    if np.any(mask):
        idx = np.arange(len(x))
        xi[mask] = np.interp(idx[mask], idx[~mask], x[~mask])
    return xi, mask

def _qrs_ms_from_Ronset_Roffset(beat_index, anchor_sample, waves_dwt, rpeaks, Fs):
    """
    Estimasi lebar QRS dari R_onset dan R_offset (ms) untuk beat terdekat.
    - beat_index: indeks iterasi saat ini (fallback jika anchor_sample tidak ada)
    - anchor_sample: sampel acuan di sekitar kompleks (mis. rata-rata Q dan S)
    """
    # Ambil array R-peak, R-onset, R-offset (buang NaN, cast ke int)
    rpeaks_arr  = np.array([x for x in rpeaks['ECG_R_Peaks']   if not np.isnan(x)], dtype=int)
    r_onsets    = np.array([x for x in waves_dwt['ECG_R_Onsets']  if not np.isnan(x)], dtype=int)
    r_offsets   = np.array([x for x in waves_dwt['ECG_R_Offsets'] if not np.isnan(x)], dtype=int)
    if len(rpeaks_arr) == 0 or len(r_onsets) == 0 or len(r_offsets) == 0:
        return None

    # Tentukan beat R yang relevan
    if anchor_sample is not None:
        j = int(np.argmin(np.abs(rpeaks_arr - int(anchor_sample))))
    else:
        j = min(max(0, beat_index), len(rpeaks_arr) - 1)
    rp = rpeaks_arr[j]

    # Cari onset terakhir sebelum R dan offset pertama sesudah R
    onset_candidates  = r_onsets[r_onsets <= rp]
    offset_candidates = r_offsets[r_offsets >= rp]
    if len(onset_candidates) == 0 or len(offset_candidates) == 0:
        return None

    onset  = int(onset_candidates.max())
    offset = int(offset_candidates.min())
    if offset <= onset:
        return None

    return ((offset - onset) / Fs) * 1000.0  # ms


def _calculate_qt_robust(r_onsets, t_offsets, t_peaks, fs, rr_ms_avg):
    """
    METODE UTAMA: Menghitung QT secara robust dengan penjodohan cerdas dan validasi fisiologis.
    """
    if r_onsets is None or len(r_onsets) == 0:
        return 0, 0, []

    t_marks = t_offsets if (t_offsets is not None and len(t_offsets) > 0) else t_peaks
    if t_marks is None or len(t_marks) == 0:
        return 0, 0, []

    valid_qts_ms = []
    t_idx = 0
    
    min_qt_ms = 200.0
    max_qt_ms = 550.0
    if rr_ms_avg and not np.isnan(rr_ms_avg) and rr_ms_avg > 0:
        min_qt_ms = max(min_qt_ms, 0.25 * rr_ms_avg) 
        max_qt_ms = min(max_qt_ms, 0.65 * rr_ms_avg)

    for r_onset in r_onsets:
        while t_idx < len(t_marks) and t_marks[t_idx] <= r_onset:
            t_idx += 1
        
        if t_idx < len(t_marks):
            qt_samples = t_marks[t_idx] - r_onset
            qt_ms = (qt_samples / fs) * 1000.0
            
            if min_qt_ms <= qt_ms <= max_qt_ms:
                valid_qts_ms.append(qt_ms)
    
    if not valid_qts_ms:
        return 0, 0, []

    qt_avg = np.median(valid_qts_ms)
    qtc_avg = 0
    if rr_ms_avg and not np.isnan(rr_ms_avg) and rr_ms_avg > 0:
        rr_sec = rr_ms_avg / 1000.0
        if rr_sec > 0:
            qtc_avg = qt_avg / (rr_sec ** (1/3)) # Rumus Fridericia

    return qt_avg, qtc_avg, valid_qts_ms


# <<< ADD: LAST-RESORT QT pairing helper >>>
def _qt_last_resort_pairing(R_onsets, T_marks, Fs, rr_ms=None,
                            min_abs_ms=160.0, max_abs_ms=600.0):
    """
    Pasangkan setiap R_onset dengan T_marks pertama yang > R_onset.
    Validasi secara fisiologis:
      - Jika RR diketahui: terima 20%–60% RR (dibatasi 160–600 ms)
      - Jika RR tidak ada: pakai 160–600 ms
    Return: (list_qt_ms, median_qt_ms atau np.nan)
    """
    R = np.asarray(R_onsets, dtype=float)
    T = np.asarray(T_marks, dtype=float)
    R = R[~np.isnan(R)]
    T = T[~np.isnan(T)]
    if len(R) == 0 or len(T) == 0:
        return [], np.nan

    # Gate fisiologis dinamis berbasis RR (opsional)
    if rr_ms is not None and np.isfinite(rr_ms):
        lo = max(min_abs_ms, 0.20 * rr_ms)
        hi = min(max_abs_ms, 0.60 * rr_ms)
    else:
        lo, hi = min_abs_ms, max_abs_ms

    qt_ms = []
    j = 0
    for r in R:
        # cari T pertama sesudah r
        while j < len(T) and T[j] <= r:
            j += 1
        if j >= len(T):
            break
        dt_ms = (T[j] - r) * 1000.0 / Fs
        if lo <= dt_ms <= hi:
            qt_ms.append(dt_ms)
        else:
            # coba T berikutnya jika kandidat pertama tidak masuk gate
            jj = j + 1
            chosen = False
            while jj < len(T):
                dt2 = (T[jj] - r) * 1000.0 / Fs
                if lo <= dt2 <= hi:
                    qt_ms.append(dt2)
                    j = jj
                    chosen = True
                    break
                jj += 1
            if not chosen:
                # biarkan j tetap, lanjut ke R berikutnya
                pass

    return qt_ms, (float(np.nanmedian(qt_ms)) if len(qt_ms) else np.nan)


# ### FITUR BARU: Fungsi Pencarian Baseline yang Adaptif ###
def find_iso_level(r_peak_index, waves_dwt, signal, fs):
    """
    Mencari level isoelektrik (baseline) untuk satu detak jantung.
    Prioritas:
    1. Segmen PR dari detak jantung saat ini.
    2. Segmen TP dari detak jantung sebelumnya.
    Mengembalikan nilai median dari segmen yang ditemukan.
    """
    p_onsets = waves_dwt['ECG_P_Onsets']
    r_onsets = waves_dwt['ECG_R_Onsets']
    t_offsets = waves_dwt['ECG_T_Offsets']

    # Prioritas 1: Cari segmen PR (P_Onset -> R_Onset)
    # Cari P_Onset dan R_Onset terakhir SEBELUM r_peak saat ini
    pr_start_candidates = p_onsets[p_onsets < r_peak_index]
    pr_end_candidates = r_onsets[r_onsets < r_peak_index]
    
    if len(pr_start_candidates) > 0 and len(pr_end_candidates) > 0:
        pr_start = int(pr_start_candidates[-1])
        pr_end = int(pr_end_candidates[-1])
        # Pastikan segmen valid
        if pr_end > pr_start and (pr_end - pr_start) > fs * 0.02: # minimal 20ms
            return np.median(signal[pr_start:pr_end])
            
    # Prioritas 2: Cari segmen TP (T_Offset -> P_Onset berikutnya)
    # Cari T_Offset terakhir SEBELUM r_peak saat ini
    tp_start_candidates = t_offsets[t_offsets < r_peak_index]
    if len(tp_start_candidates) > 0 and len(pr_start_candidates) > 0:
        tp_start = int(tp_start_candidates[-1])
        tp_end = int(pr_start_candidates[-1]) # P_Onset dari detak saat ini
        # Pastikan segmen valid
        if tp_end > tp_start and (tp_end - tp_start) > fs * 0.02:
            return np.median(signal[tp_start:tp_end])

    # Jika semua gagal, kembalikan None
    return None


def apply_iec_diagnostic_filter(signal, fs):
    """
    IEC 60601-2-25:2011 compliant diagnostic ECG filter (stabilized version)
    
    Tahapan:
    1. Baseline correction (detrending)
    2. High-pass filter (0.05 Hz)
    3. Low-pass filter (150 Hz)
    4. Notch filter (50 Hz)
    5. Edge padding untuk mencegah artefak
    """
    # Baseline correction (hilangkan offset DC)
    # signal = signal - np.mean(signal)

    # Padding sinyal untuk menghindari artefak filtfilt di tepi
    pad_len = int(fs * 2)  # 2 detik padding
    padded_signal = np.pad(signal, (pad_len, pad_len), mode='edge')

    # Nyquist frequency
    nyquist = fs / 2.0

    # --- Step 1: High-pass filter 0.05 Hz
    hp_cutoff = 0.05 / nyquist
    hp_cutoff = min(hp_cutoff, 0.99)
    b_hp, a_hp = scipy.signal.butter(4, hp_cutoff, btype='highpass')
    signal_hp = scipy.signal.filtfilt(b_hp, a_hp, padded_signal)

    # --- Step 2: Low-pass filter 150 Hz
    lp_cutoff = min(150 / nyquist, 0.99)
    b_lp, a_lp = scipy.signal.butter(4, lp_cutoff, btype='lowpass')
    signal_lp = scipy.signal.filtfilt(b_lp, a_lp, signal_hp)

    # --- Step 3: Notch filter 50 Hz
    f0 = 50.0
    Q = 30.0  # Quality factor
    if f0 < nyquist:
        w0 = f0 / nyquist
        b_notch, a_notch = scipy.signal.iirnotch(w0, Q)
        signal_filtered = scipy.signal.filtfilt(b_notch, a_notch, signal_lp)
    else:
        signal_filtered = signal_lp

    # --- Remove padding
    final_signal = signal_filtered[pad_len:-pad_len]

    return final_signal


def wavelet_denoise(
    x,
    fs,
    wavelet='sym6',
    level=None,
    mode='symmetric',
    method='soft',
    protect_low_hz=2.0,            # lindungi ≤ 2 Hz (ST/J-point)
    protect_band=(1.0, 40.0),      # lunakkan ambang di 1–40 Hz (P/T & QRS)
    protect_factor=0.5             # skala ambang di band yang dilindungi
):
    """
    Denoising berbasis wavelet untuk ECG, ramah ST-segment.
    - Tidak men-threshold koefisien aproksimasi (≈ DC–very-low).
    - Melewati threshold untuk detail ≤ protect_low_hz (default 2 Hz).
    - Melunakkan threshold pada band protect_band (default 1–40 Hz).
    - Soft-threshold berbasis MAD (VisuShrink).
    """
    if pywt is None:
        return np.asarray(x, dtype=float)

    x = np.asarray(x, dtype=float)
    w = pywt.Wavelet(wavelet)
    maxlev = pywt.dwt_max_level(len(x), w.dec_len)

    # Level default: 4–6 lazim; 5 cocok untuk Fs ≈ 250–500 Hz
    if level is None:
        level = min(5, maxlev)

    coeffs = pywt.wavedec(x, w, mode=mode, level=level)
    cA = coeffs[0]           # aproksimasi (≈ DC–very-low): JANGAN diubah
    details = coeffs[1:]     # detail level 1..level

    new_coeffs = [cA]
    lo_prot, hi_prot = protect_band

    for i, cD in enumerate(details, start=1):
        # Perkiraan pita frekuensi level-i (dyadik)
        f_low  = fs / (2.0 ** (i + 1))
        f_high = fs / (2.0 ** i)

        # Estimasi sigma via MAD (robust)
        sigma = (np.median(np.abs(cD - np.median(cD))) / 0.6745) + 1e-12
        thr_univ = sigma * np.sqrt(2.0 * np.log(max(cD.size, 2)))

        # Kebijakan proteksi:
        if f_high <= protect_low_hz:
            # ≤ ~2 Hz: biarkan utuh (jaga ST/J-point)
            cD_th = cD
        else:
            # Lunakkan di band yang tumpang tindih 1–40 Hz
            overlaps = (f_low <= hi_prot) and (f_high >= lo_prot)
            thr = thr_univ * (protect_factor if overlaps else 1.0)
            cD_th = pywt.threshold(cD, thr, mode=method)

        new_coeffs.append(cD_th)

    y = pywt.waverec(new_coeffs, w, mode=mode)
    return y[:len(x)]


# ====== PROSES SATU FILE ======
def extract_features_from_csv(csv_path):
    # Selalu kembalikan sesuatu meski gagal
    def _empty_row():
        return {f: 0.0 for f in EXPECTED_FEATURES}, "Unknown"

    try:
        dataset = pd.read_csv(csv_path, names=LEADS, sep=',', skiprows=1, dtype=float)
        # Fs = float(len(dataset) / 10)  # asumsi 10 detik rekaman
        Fs = 853  # tetap 853 Hz
        packed = {f: 0.0 for f in EXPECTED_FEATURES}  # lokal

        # ---- proses SEMUA lead ----
        for lead in LEADS:
            try:
                ecg_adc = dataset[lead].values.astype(float)
                ecg_adc, _ = desaturate_edges(ecg_adc)
                ecg_adc, _ = despike_hampel(ecg_adc, k=7, nsigma=6.0)
                ecgmv = to_mV(ecg_adc)

                # =================================================================
                # PIPELINE: Raw Signal → Baseline → IEC Filter → Wavelet Denoising
                # =================================================================
                
                # STEP 1: Baseline Correction 
                detr_ecg = scipy.signal.detrend(ecgmv,  axis=-1, type='constant')
                
                print(f"Applying IEC 60601-2-25:2011 filter for {lead}...")
                
                # STEP 2: Apply IEC 60601-2-25 Diagnostic Filter (TAMBAHAN)
                iec_filtered_signal = apply_iec_diagnostic_filter(detr_ecg, Fs)

    
                # STEP 3: Wavelet denoising (shrinkage) – tekan EMG tanpa merusak QRS
                y_filt = wavelet_denoise(iec_filtered_signal, Fs, wavelet='sym6', level=None, method='soft')
                
                # --- deteksi puncak & fitur (kode kamu) ---
                try:
                    # PQRST Peak Detection
                    _, rpeaks = nk.ecg_peaks(y_filt, sampling_rate=Fs)
                    signal_dwt, waves_dwt = nk.ecg_delineate(y_filt, rpeaks, sampling_rate=Fs, method="dwt")

                    # Remove Nan and change to ndarray int
                    peaksR = np.array([x for x in rpeaks['ECG_R_Peaks'] if math.isnan(x) is False]).astype(int)
                    peaksP = np.array([x for x in waves_dwt['ECG_P_Peaks'] if math.isnan(x) is False]).astype(int)
                    peaksQ = np.array([x for x in waves_dwt['ECG_Q_Peaks'] if math.isnan(x) is False]).astype(int)
                    peaksS = np.array([x for x in waves_dwt['ECG_S_Peaks'] if math.isnan(x) is False]).astype(int)
                    peaksT = np.array([x for x in waves_dwt['ECG_T_Peaks'] if math.isnan(x) is False]).astype(int)
                    peaksPOnsets = np.array([x for x in waves_dwt['ECG_P_Onsets'] if math.isnan(x) is False]).astype(int)
                    peaksPOffsets = np.array([x for x in waves_dwt['ECG_P_Offsets'] if math.isnan(x) is False]).astype(int)
                    peaksROnsets = np.array([x for x in waves_dwt['ECG_R_Onsets'] if math.isnan(x) is False]).astype(int)
                    peaksROffsets = np.array([x for x in waves_dwt['ECG_R_Offsets'] if math.isnan(x) is False]).astype(int)
                    peaksTOnsets = np.array([x for x in waves_dwt['ECG_T_Onsets'] if math.isnan(x) is False]).astype(int)
                    peaksTOffsets = np.array([x for x in waves_dwt['ECG_T_Offsets'] if math.isnan(x) is False]).astype(int)

                    # Remove Nan and change to ndarray int
                    rpeaks['ECG_R_Peaks'] = np.array([x for x in rpeaks['ECG_R_Peaks'] if math.isnan(x) is False]).astype(int)
                    waves_dwt['ECG_P_Peaks'] = np.array([x for x in waves_dwt['ECG_P_Peaks'] if math.isnan(x) is False]).astype(int)
                    waves_dwt['ECG_Q_Peaks'] = np.array([x for x in waves_dwt['ECG_Q_Peaks'] if math.isnan(x) is False]).astype(int)
                    waves_dwt['ECG_S_Peaks'] = np.array([x for x in waves_dwt['ECG_S_Peaks'] if math.isnan(x) is False]).astype(int)
                    waves_dwt['ECG_T_Peaks'] = np.array([x for x in waves_dwt['ECG_T_Peaks'] if math.isnan(x) is False]).astype(int)
                    waves_dwt['ECG_P_Onsets'] = np.array([x for x in waves_dwt['ECG_P_Onsets'] if math.isnan(x) is False]).astype(int)
                    waves_dwt['ECG_P_Offsets'] = np.array([x for x in waves_dwt['ECG_P_Offsets'] if math.isnan(x) is False]).astype(int)
                    waves_dwt['ECG_R_Onsets'] = np.array([x for x in waves_dwt['ECG_R_Onsets'] if math.isnan(x) is False]).astype(int)
                    waves_dwt['ECG_R_Offsets'] = np.array([x for x in waves_dwt['ECG_R_Offsets'] if math.isnan(x) is False]).astype(int)
                    waves_dwt['ECG_T_Onsets'] = np.array([x for x in waves_dwt['ECG_T_Onsets'] if math.isnan(x) is False]).astype(int)
                    waves_dwt['ECG_T_Offsets'] = np.array([x for x in waves_dwt['ECG_T_Offsets'] if math.isnan(x) is False]).astype(int)

                    # =================================================================
                    # CORRECTIONS EXACTLY LIKE CODE 2 - COMPLETE IMPLEMENTATION
                    # =================================================================

                    # Check if arrays have data before corrections
                    if len(rpeaks['ECG_R_Peaks']) > 0 and len(waves_dwt['ECG_P_Onsets']) > 0:
                        # Correcting first cycle - EXACT SAME LOGIC AS CODE 2
                        if rpeaks['ECG_R_Peaks'][0] < waves_dwt['ECG_P_Onsets'][0]:
                                rpeaks['ECG_R_Peaks'] = np.delete(rpeaks['ECG_R_Peaks'], 0)
                        if len(waves_dwt['ECG_P_Peaks']) > 0 and waves_dwt['ECG_P_Peaks'][0] < waves_dwt['ECG_P_Onsets'][0]:
                            waves_dwt['ECG_P_Peaks'] = np.delete(waves_dwt['ECG_P_Peaks'], 0)
                        if len(waves_dwt['ECG_Q_Peaks']) > 0 and waves_dwt['ECG_Q_Peaks'][0] < waves_dwt['ECG_P_Onsets'][0]:
                            waves_dwt['ECG_Q_Peaks'] = np.delete(waves_dwt['ECG_Q_Peaks'], 0)
                        if len(waves_dwt['ECG_S_Peaks']) > 0 and waves_dwt['ECG_S_Peaks'][0] < waves_dwt['ECG_P_Onsets'][0]:
                                waves_dwt['ECG_S_Peaks'] = np.delete(waves_dwt['ECG_S_Peaks'], 0)
                        if len(waves_dwt['ECG_T_Peaks']) > 0 and waves_dwt['ECG_T_Peaks'][0] < waves_dwt['ECG_P_Onsets'][0]:
                            waves_dwt['ECG_T_Peaks'] = np.delete(waves_dwt['ECG_T_Peaks'], 0)
                        if len(waves_dwt['ECG_P_Offsets']) > 0 and waves_dwt['ECG_P_Offsets'][0] < waves_dwt['ECG_P_Onsets'][0]:
                            waves_dwt['ECG_P_Offsets'] = np.delete(waves_dwt['ECG_P_Offsets'], 0)
                        if len(waves_dwt['ECG_R_Offsets']) > 0 and waves_dwt['ECG_R_Offsets'][0] < waves_dwt['ECG_P_Onsets'][0]:
                            waves_dwt['ECG_R_Offsets'] = np.delete(waves_dwt['ECG_R_Offsets'], 0)
                        if len(waves_dwt['ECG_T_Offsets']) > 0 and waves_dwt['ECG_T_Offsets'][0] < waves_dwt['ECG_P_Onsets'][0]:
                            waves_dwt['ECG_T_Offsets'] = np.delete(waves_dwt['ECG_T_Offsets'], 0)
                        if len(waves_dwt['ECG_R_Onsets']) > 0 and waves_dwt['ECG_R_Onsets'][0] < waves_dwt['ECG_P_Onsets'][0]:
                            waves_dwt['ECG_R_Onsets'] = np.delete(waves_dwt['ECG_R_Onsets'], 0)
                        if len(waves_dwt['ECG_T_Onsets']) > 0 and waves_dwt['ECG_T_Onsets'][0] < waves_dwt['ECG_P_Onsets'][0]:
                            waves_dwt['ECG_T_Onsets'] = np.delete(waves_dwt['ECG_T_Onsets'], 0)

                        # Additional corrections relative to R peaks
                        if len(rpeaks['ECG_R_Peaks']) > 0:
                            if len(waves_dwt['ECG_R_Offsets']) > 0 and waves_dwt['ECG_R_Offsets'][0] < rpeaks['ECG_R_Peaks'][0]:
                                waves_dwt['ECG_R_Offsets'] = np.delete(waves_dwt['ECG_R_Offsets'], 0)
                            if len(waves_dwt['ECG_T_Offsets']) > 0 and waves_dwt['ECG_T_Offsets'][0] < rpeaks['ECG_R_Peaks'][0]:
                                waves_dwt['ECG_T_Offsets'] = np.delete(waves_dwt['ECG_T_Offsets'], 0)
                            if len(waves_dwt['ECG_T_Onsets']) > 0 and waves_dwt['ECG_T_Onsets'][0] < rpeaks['ECG_R_Peaks'][0]:
                                waves_dwt['ECG_T_Onsets'] = np.delete(waves_dwt['ECG_T_Onsets'], 0)
                            if len(waves_dwt['ECG_S_Peaks']) > 0 and waves_dwt['ECG_S_Peaks'][0] < rpeaks['ECG_R_Peaks'][0]:
                                waves_dwt['ECG_S_Peaks'] = np.delete(waves_dwt['ECG_S_Peaks'], 0)
                            if len(waves_dwt['ECG_T_Peaks']) > 0 and waves_dwt['ECG_T_Peaks'][0] < rpeaks['ECG_R_Peaks'][0]:
                                waves_dwt['ECG_T_Peaks'] = np.delete(waves_dwt['ECG_T_Peaks'], 0)

                    # EXACT SAME LOW AMPLITUDE R PEAK CORRECTION AS CODE 2
                    if len(rpeaks['ECG_R_Peaks']) > 1:
                        if y_filt[rpeaks['ECG_R_Peaks']][0] < y_filt[rpeaks['ECG_R_Peaks']][1]/2:
                            rpeaks['ECG_R_Peaks'] = np.delete(rpeaks['ECG_R_Peaks'], 0)
                            if len(waves_dwt['ECG_P_Peaks']) > 0:
                                waves_dwt['ECG_P_Peaks'] = np.delete(waves_dwt['ECG_P_Peaks'], 0)
                            if len(waves_dwt['ECG_Q_Peaks']) > 0:
                                waves_dwt['ECG_Q_Peaks'] = np.delete(waves_dwt['ECG_Q_Peaks'], 0)
                            if len(waves_dwt['ECG_S_Peaks']) > 0:
                                waves_dwt['ECG_S_Peaks'] = np.delete(waves_dwt['ECG_S_Peaks'], 0)
                            if len(waves_dwt['ECG_T_Peaks']) > 0:
                                waves_dwt['ECG_T_Peaks'] = np.delete(waves_dwt['ECG_T_Peaks'], 0)
                            if len(waves_dwt['ECG_R_Onsets']) > 0:
                                waves_dwt['ECG_R_Onsets'] = np.delete(waves_dwt['ECG_R_Onsets'], 0)
                            if len(waves_dwt['ECG_R_Offsets']) > 0:
                                waves_dwt['ECG_R_Offsets'] = np.delete(waves_dwt['ECG_R_Offsets'], 0)
                            if len(waves_dwt['ECG_P_Onsets']) > 0:
                                waves_dwt['ECG_P_Onsets'] = np.delete(waves_dwt['ECG_P_Onsets'], 0)
                            if len(waves_dwt['ECG_P_Offsets']) > 0:
                                waves_dwt['ECG_P_Offsets'] = np.delete(waves_dwt['ECG_P_Offsets'], 0)
                            if len(waves_dwt['ECG_T_Onsets']) > 0:
                                waves_dwt['ECG_T_Onsets'] = np.delete(waves_dwt['ECG_T_Onsets'], 0)
                            if len(waves_dwt['ECG_T_Offsets']) > 0:
                                waves_dwt['ECG_T_Offsets'] = np.delete(waves_dwt['ECG_T_Offsets'], 0)

                    # EXACT SAME LAST CYCLE CORRECTIONS AS CODE 2
                    if len(rpeaks['ECG_R_Peaks']) > 0 and len(waves_dwt['ECG_T_Offsets']) > 0:
                        if rpeaks['ECG_R_Peaks'][len(rpeaks['ECG_R_Peaks']) - 1] > waves_dwt['ECG_T_Offsets'][len(waves_dwt['ECG_T_Offsets']) - 1]:
                            rpeaks['ECG_R_Peaks'] = np.delete(rpeaks['ECG_R_Peaks'], (len(rpeaks['ECG_R_Peaks']) - 1))
                        if len(waves_dwt['ECG_P_Peaks']) > 0 and waves_dwt['ECG_P_Peaks'][len(waves_dwt['ECG_P_Peaks']) - 1] > waves_dwt['ECG_T_Offsets'][len(waves_dwt['ECG_T_Offsets']) - 1]:
                            waves_dwt['ECG_P_Peaks'] = np.delete(waves_dwt['ECG_P_Peaks'], (len(waves_dwt['ECG_P_Peaks']) - 1))
                        if len(waves_dwt['ECG_Q_Peaks']) > 0 and waves_dwt['ECG_Q_Peaks'][len(waves_dwt['ECG_Q_Peaks']) - 1] > waves_dwt['ECG_T_Offsets'][len(waves_dwt['ECG_T_Offsets']) - 1]:
                            waves_dwt['ECG_Q_Peaks'] = np.delete(waves_dwt['ECG_Q_Peaks'], (len(waves_dwt['ECG_Q_Peaks']) - 1))
                        if len(waves_dwt['ECG_S_Peaks']) > 0 and waves_dwt['ECG_S_Peaks'][len(waves_dwt['ECG_S_Peaks']) - 1] > waves_dwt['ECG_T_Offsets'][len(waves_dwt['ECG_T_Offsets']) - 1]:
                            waves_dwt['ECG_S_Peaks'] = np.delete(waves_dwt['ECG_S_Peaks'], (len(waves_dwt['ECG_S_Peaks']) - 1))
                        if len(waves_dwt['ECG_T_Peaks']) > 0 and waves_dwt['ECG_T_Peaks'][len(waves_dwt['ECG_T_Peaks']) - 1] > waves_dwt['ECG_T_Offsets'][len(waves_dwt['ECG_T_Offsets']) - 1]:
                            waves_dwt['ECG_T_Peaks'] = np.delete(waves_dwt['ECG_T_Peaks'], (len(waves_dwt['ECG_T_Peaks']) - 1))
                        if len(waves_dwt['ECG_P_Onsets']) > 0 and waves_dwt['ECG_P_Onsets'][len(waves_dwt['ECG_P_Onsets']) - 1] > waves_dwt['ECG_T_Offsets'][len(waves_dwt['ECG_T_Offsets']) - 1]:
                            waves_dwt['ECG_P_Onsets'] = np.delete(waves_dwt['ECG_P_Onsets'], (len(waves_dwt['ECG_P_Onsets']) - 1))
                        if len(waves_dwt['ECG_T_Onsets']) > 0 and waves_dwt['ECG_T_Onsets'][len(waves_dwt['ECG_T_Onsets']) - 1] > waves_dwt['ECG_T_Offsets'][len(waves_dwt['ECG_T_Offsets']) - 1]:
                            waves_dwt['ECG_T_Onsets'] = np.delete(waves_dwt['ECG_T_Onsets'], (len(waves_dwt['ECG_T_Onsets']) - 1))
                        if len(waves_dwt['ECG_R_Onsets']) > 0 and waves_dwt['ECG_R_Onsets'][len(waves_dwt['ECG_R_Onsets']) - 1] > waves_dwt['ECG_T_Offsets'][len(waves_dwt['ECG_T_Offsets']) - 1]:
                            waves_dwt['ECG_R_Onsets'] = np.delete(waves_dwt['ECG_R_Onsets'], (len(waves_dwt['ECG_R_Onsets']) - 1))
                        if len(waves_dwt['ECG_R_Offsets']) > 0 and waves_dwt['ECG_R_Offsets'][len(waves_dwt['ECG_R_Offsets']) - 1] > waves_dwt['ECG_T_Offsets'][len(waves_dwt['ECG_T_Offsets']) - 1]:
                            waves_dwt['ECG_R_Offsets'] = np.delete(waves_dwt['ECG_R_Offsets'], (len(waves_dwt['ECG_R_Offsets']) - 1))

                    # Additional last cycle corrections relative to R peaks
                    if len(rpeaks['ECG_R_Peaks']) > 0:
                        if len(waves_dwt['ECG_P_Peaks']) > 0 and waves_dwt['ECG_P_Peaks'][len(waves_dwt['ECG_P_Peaks']) - 1] > rpeaks['ECG_R_Peaks'][len(rpeaks['ECG_R_Peaks']) - 1]:
                            waves_dwt['ECG_P_Peaks'] = np.delete(waves_dwt['ECG_P_Peaks'], (len(waves_dwt['ECG_P_Peaks']) - 1))
                        if len(waves_dwt['ECG_Q_Peaks']) > 0 and waves_dwt['ECG_Q_Peaks'][len(waves_dwt['ECG_Q_Peaks']) - 1] > rpeaks['ECG_R_Peaks'][len(rpeaks['ECG_R_Peaks']) - 1]:
                            waves_dwt['ECG_Q_Peaks'] = np.delete(waves_dwt['ECG_Q_Peaks'], (len(waves_dwt['ECG_Q_Peaks']) - 1))
                        if len(waves_dwt['ECG_P_Onsets']) > 0 and waves_dwt['ECG_P_Onsets'][len(waves_dwt['ECG_P_Onsets']) - 1] > rpeaks['ECG_R_Peaks'][len(rpeaks['ECG_R_Peaks']) - 1]:
                            waves_dwt['ECG_P_Onsets'] = np.delete(waves_dwt['ECG_P_Onsets'], (len(waves_dwt['ECG_P_Onsets']) - 1))
                        if len(waves_dwt['ECG_P_Offsets']) > 0 and waves_dwt['ECG_P_Offsets'][len(waves_dwt['ECG_P_Offsets']) - 1] > rpeaks['ECG_R_Peaks'][len(rpeaks['ECG_R_Peaks']) - 1]:
                            waves_dwt['ECG_P_Offsets'] = np.delete(waves_dwt['ECG_P_Offsets'], (len(waves_dwt['ECG_P_Offsets']) - 1))
                        if len(waves_dwt['ECG_R_Onsets']) > 0 and waves_dwt['ECG_R_Onsets'][len(waves_dwt['ECG_R_Onsets']) - 1] > rpeaks['ECG_R_Peaks'][len(rpeaks['ECG_R_Peaks']) - 1]:
                            waves_dwt['ECG_R_Onsets'] = np.delete(waves_dwt['ECG_R_Onsets'], (len(waves_dwt['ECG_R_Onsets']) - 1))


                    # =================================================================
                    # FEATURE EXTRACTION DENGAN ALGORITMA ORIGINAL
                    # =================================================================

                    # Initialize variables for this lead
                    RR_avg = 0
                    RR_stdev = 0
                    PR_avg = 0
                    qrs_avg = 0
                    QT_avg = 0
                    QTc_avg = 0
                    ST_avg_amplitude = 0
                    ST_avg_deviation = 0    
                    RS_ratio = 0
                    bpm = 0
                    RR_list = [] 


                    # 1. RR INTERVAL (ALGORITMA ORIGINAL)
                    if len(rpeaks['ECG_R_Peaks']) > 1:
                        RR_list = []
                        cnt = 0
                        while (cnt < (len(rpeaks['ECG_R_Peaks']) - 1)):
                            RR_interval = (rpeaks['ECG_R_Peaks'][cnt + 1] - rpeaks['ECG_R_Peaks'][cnt])
                            RRms_dist = ((RR_interval / Fs) * 1000.0)  # Convert sample distances to ms distances
                            RR_list.append(RRms_dist)
                            cnt += 1

                        if len(RR_list) > 0:
                            dfRR = pd.DataFrame(RR_list)
                            dfRR = dfRR.fillna(0)
                            RR_stdev = np.std(RR_list, axis=None)  # Save stdev

                            # Perhitungan average seperti original
                            sum_rr = 0.0
                            count_rr = 0.0
                            for index in range(len(RR_list)):
                                if (np.isnan(RR_list[index]) == True):
                                    continue
                                else:
                                    sum_rr += RR_list[index]
                                    count_rr += 1

                            if count_rr > 0:
                                RR_avg = (sum_rr / count_rr)
                                bpm = 60000 / np.mean(RR_list)  # BPM calculation

                    # 2. PR INTERVAL (ALGORITMA ORIGINAL)
                    if len(waves_dwt['ECG_R_Onsets']) > 1 and len(waves_dwt['ECG_P_Onsets']) > 1:
                        PR_peak_list = []
                        idex = ([x for x in range(0, len(waves_dwt['ECG_R_Onsets']) - 1)])
                        for i in idex:
                            if waves_dwt['ECG_R_Onsets'][i] < waves_dwt['ECG_P_Onsets'][i]:
                                cnt = 0
                                while (cnt < (len(waves_dwt['ECG_R_Onsets']) - 1)):
                                    if cnt < len(waves_dwt['ECG_Q_Peaks']):
                                        PR_peak_interval = (waves_dwt['ECG_Q_Peaks'][cnt] - waves_dwt['ECG_P_Onsets'][cnt])
                                        ms_dist = ((PR_peak_interval / Fs) * 1000.0)
                                        PR_peak_list.append(ms_dist)
                                    cnt += 1
                            else:
                                cnt = 0
                                while (cnt < (len(waves_dwt['ECG_R_Onsets']) - 1)):
                                    PR_peak_interval = (waves_dwt['ECG_R_Onsets'][cnt] - waves_dwt['ECG_P_Onsets'][cnt])
                                    ms_dist = ((PR_peak_interval / Fs) * 1000.0)
                                    PR_peak_list.append(ms_dist)
                                    cnt += 1

                        if len(PR_peak_list) > 0:
                            dfPR = pd.DataFrame(PR_peak_list)
                            dfPR = dfPR.fillna(0)

                            # Perhitungan average seperti original
                            sum_pr = 0.0
                            count_pr = 0.0
                            for index in range(len(PR_peak_list)):
                                if (np.isnan(PR_peak_list[index]) == True):
                                    continue
                                else:
                                    sum_pr += PR_peak_list[index]
                                    count_pr += 1

                            if count_pr > 0:
                                PR_avg = (sum_pr / count_pr)

                    # 3. qrs INTERVAL (ALGORITMA ORIGINAL)
                    if len(waves_dwt['ECG_S_Peaks']) > 1 and len(waves_dwt['ECG_Q_Peaks']) > 1:
                        qrs_peak_list = []
                        try:
                            idex = [x for x in range(0, len(waves_dwt['ECG_S_Peaks']) - 1)]
                            for i in idex:
                                # Hitung qrs berbasis Q & S seperti versi asli
                                if waves_dwt['ECG_S_Peaks'][i] < waves_dwt['ECG_Q_Peaks'][i]:
                                    qrs_samples = (waves_dwt['ECG_S_Peaks'][i + 1] - waves_dwt['ECG_Q_Peaks'][i])
                                else:
                                    qrs_samples = (waves_dwt['ECG_S_Peaks'][i] - waves_dwt['ECG_Q_Peaks'][i])

                                ms_dist = ((qrs_samples / Fs) * 1000.0)

                                # === FALLBACK: jika qrs = 0 ms, pakai R_offset - R_onset beat terkait ===
                                if not np.isnan(ms_dist) and (abs(ms_dist) < 1e-9):
                                    # Anchor di sekitar kompleks: pakai rata-rata Q & S (abaikan NaN)
                                    q_i = waves_dwt['ECG_Q_Peaks'][i]
                                    s_i = waves_dwt['ECG_S_Peaks'][i]
                                    anchor_vals = [v for v in [q_i, s_i] if not np.isnan(v)]
                                    anchor = float(np.nanmean(anchor_vals)) if len(anchor_vals) > 0 else None

                                    fallback_ms = _qrs_ms_from_Ronset_Roffset(i, anchor, waves_dwt, rpeaks, Fs)
                                    if fallback_ms is not None:
                                        ms_dist = fallback_ms  # gantikan dengan Roffset-Ronset

                                qrs_peak_list.append(ms_dist)

                            if len(qrs_peak_list) > 0:
                                dfqrs = pd.DataFrame(qrs_peak_list).fillna(0)

                                # Manual averaging seperti original
                                sum_qrs = 0.0
                                count_qrs = 0.0
                                for index in range(len(qrs_peak_list)):
                                    if np.isnan(qrs_peak_list[index]):
                                        continue
                                    else:
                                        sum_qrs += qrs_peak_list[index]
                                        count_qrs += 1

                                if count_qrs > 0:
                                    qrs_avg = (sum_qrs / count_qrs)
                        except:
                            print(f"QRS width Error for {lead}")

                    # =================================================================
                    # 4. QT/QTc INTERVAL (ROBUST + LAST RESORT)
                    # =================================================================
                    QT_avg = 0
                    QTc_avg = 0
                    
                    # --- TAHAP 1: Coba metode perhitungan robust (pilihan utama) ---
                    try:
                        QT_avg, QTc_avg, qt_list = _calculate_qt_robust(
                            waves_dwt.get('ECG_R_Onsets'),
                            waves_dwt.get('ECG_T_Offsets'),
                            waves_dwt.get('ECG_T_Peaks'),
                            Fs,
                            RR_avg 
                        )
                    except Exception as e:
                        print(f"[{lead}] Error pada metode QT robust: {e}")
                        QT_avg, QTc_avg = 0, 0

                    # --- TAHAP 2: Jika metode utama gagal (hasil=0), jalankan Last Resort ---
                    if QT_avg == 0 or not np.isfinite(QT_avg):
                        try:
                            print(f"[{lead}] Metode utama QT gagal, menjalankan Last Resort...")
                            
                            # Coba Last Resort dengan T-Offsets dulu
                            qt_list_fb, qt_avg_fb = _qt_last_resort_pairing(
                                waves_dwt['ECG_R_Onsets'],
                                waves_dwt['ECG_T_Offsets'],
                                Fs,
                                rr_ms=RR_avg
                            )

                            # Jika masih gagal, coba Last Resort dengan T-Peaks
                            if not np.isfinite(qt_avg_fb):
                                qt_list_fb, qt_avg_fb = _qt_last_resort_pairing(
                                    waves_dwt['ECG_R_Onsets'],
                                    waves_dwt['ECG_T_Peaks'],
                                    Fs,
                                    rr_ms=RR_avg
                                )
                            
                            # Jika Last Resort berhasil, update nilai QT dan hitung ulang QTc
                            if np.isfinite(qt_avg_fb):
                                print(f"[{lead}] Last Resort berhasil! QT ditemukan: {qt_avg_fb:.2f} ms")
                                QT_avg = float(qt_avg_fb)
                                if RR_avg is not None and RR_avg > 0:
                                    rr_sec = RR_avg / 1000.0
                                    QTc_avg = QT_avg / (rr_sec ** (1/3)) # Hitung ulang QTc Fridericia
                            else:
                                print(f"[{lead}] Last Resort juga gagal.")

                        except Exception as e_fb:
                            print(f"[{lead}] Error pada metode QT Last Resort: {e_fb}")
                            

                    # =================================================================
                    # 5. ST SEGMENT (ALGORITMA BARU YANG ADAPTIVE)
                    # =================================================================
                    # ### PERBAIKAN 2: Menggunakan Algoritma ST yang Baru dan Adaptif ###
                    st_amplitudes_list = []
                    st_deviations_list = []
                    
                    # J-point adalah R_Offsets
                    j_points = waves_dwt['ECG_R_Offsets'][~np.isnan(waves_dwt['ECG_R_Offsets'])]
                    # Titik pengukuran adalah 60ms setelah J-point
                    st_measurement_offset = int(0.060 * Fs)

                    for i, r_peak in enumerate(rpeaks['ECG_R_Peaks']):
                        # 1. Cari baseline untuk detak jantung ini
                        baseline_mv = find_iso_level(r_peak, waves_dwt, y_filt, Fs)
                        
                        # 2. Cari J-point yang berasosiasi dengan R-peak ini
                        #    (J-point pertama setelah R-peak)
                        associated_j_points = j_points[j_points > r_peak]
                        
                        # Jika baseline dan J-point ditemukan
                        if baseline_mv is not None and len(associated_j_points) > 0:
                            j_point = int(associated_j_points[0])
                            st_measurement_point = j_point + st_measurement_offset
                            
                            # 3. Pastikan titik pengukuran masih dalam rentang sinyal
                            if st_measurement_point < len(y_filt):
                                st_amplitude = y_filt[st_measurement_point]
                                st_deviation = st_amplitude - baseline_mv
                                
                                st_amplitudes_list.append(st_amplitude)
                                st_deviations_list.append(st_deviation)
                    
                    # Hitung rata-rata jika list berhasil diisi
                    if st_amplitudes_list:
                        ST_avg_amplitude = np.nanmean(st_amplitudes_list)
                        print(f"  ✅ ST Amplitude terhitung: {ST_avg_amplitude:.4f} mV (dari {len(st_amplitudes_list)} detak jantung)")
                    if st_deviations_list:
                        ST_avg_deviation = np.nanmean(st_deviations_list)
                        print(f"  ✅ ST Deviation terhitung: {ST_avg_deviation:.4f} mV (dari {len(st_deviations_list)} detak jantung)")

                    # 6. R/S RATIO (ALGORITMA ORIGINAL)
                    if 'ECG_S_Peaks' in waves_dwt.keys() and len(waves_dwt['ECG_S_Peaks']) > 0 and len(rpeaks['ECG_R_Peaks']) > 0:
                        try:
                            R_mean_amp = np.mean([y_filt[int(i)] for i in rpeaks['ECG_R_Peaks']])
                            S_mean_amp = np.mean([y_filt[int(i)] for i in waves_dwt['ECG_S_Peaks']])

                            if S_mean_amp != 0:  # Avoid division by zero
                                RS_ratio = (R_mean_amp) / abs(S_mean_amp)
                        except Exception as e:
                            print(f"Error calculating R/S ratio for {lead}: {e}")

                    # Store calculated features for this lead
                    lead_lower = lead.lower()
                    packed[f"rr_{lead_lower}"] = RR_avg
                    packed[f"rr_std_{lead_lower}"] = RR_stdev
                    packed[f"pr_{lead_lower}"] = PR_avg
                    packed[f"qrs_{lead_lower}"] = qrs_avg
                    packed[f"qtc_{lead_lower}"] = QTc_avg
                    packed[f"st_amplitude_mv_{lead_lower}"] = ST_avg_amplitude if np.isfinite(ST_avg_amplitude) else 0
                    packed[f"st_deviation_mv_{lead_lower}"] = ST_avg_deviation if np.isfinite(ST_avg_deviation) else 0
                    packed[f"rs_ratio_{lead_lower}"] = RS_ratio
                    packed[f"heartrate_{lead_lower}"] = bpm

                    # # Print results for this lead
                    # print(f'\n{lead} - HASIL DENGAN ALGORITMA ORIGINAL + IEC FILTER')
                    # print(f'{lead} - RR Interval (ms) - Mean: {RR_avg:.2f}, Standard Deviation: {RR_stdev:.2f}')
                    # print(f'{lead} - PR Interval (ms) - Mean: {PR_avg:.2f}')
                    # print(f'{lead} - qrs Interval (ms) - Mean: {qrs_avg:.2f}')
                    # print(f'{lead} - QT Interval (ms) - Mean: {QT_avg:.2f}')
                    # print(f'{lead} - QTc Interval (ms) - Mean: {QTc_avg:.2f}')
                    # print(f'{lead} - ST Segment (ms) - Mean: {ST_avg:.2f}')
                    # print(f'{lead} - R/S Ratio: {RS_ratio:.4f}')
                    # print(f'{lead} - BPM: {bpm:.2f}')

                    # # Plot PQRST peaks for ALL leads dengan algoritma original
                    # plt.figure(figsize=(10, 6))
                    # plt.title(f'Electrocardiogram Signal ({lead}) - PQRST Peaks Detection (Original Algorithm + IEC Filter)')
                    # plt.plot(t, y_filt, color='orange', label="Filtered Data")

                    # # Only plot peaks if they were successfully detected
                    # if len(rpeaks['ECG_R_Peaks']) > 0:
                    #     plt.plot(rpeaks['ECG_R_Peaks'], y_filt[rpeaks['ECG_R_Peaks']], "x", color='black', label="R Peak")
                    # if len(waves_dwt['ECG_P_Peaks']) > 0:
                    #     plt.plot(waves_dwt['ECG_P_Peaks'], y_filt[waves_dwt['ECG_P_Peaks']], "x", color='blue', label="P Peak")
                    # if len(waves_dwt['ECG_Q_Peaks']) > 0:
                    #     plt.plot(waves_dwt['ECG_Q_Peaks'], y_filt[waves_dwt['ECG_Q_Peaks']], "x", color='green', label="Q Peak")
                    # if len(waves_dwt['ECG_S_Peaks']) > 0:
                    #     plt.plot(waves_dwt['ECG_S_Peaks'], y_filt[waves_dwt['ECG_S_Peaks']], "x", color='red', label="S Peak")
                    # if len(waves_dwt['ECG_T_Peaks']) > 0:
                    #     plt.plot(waves_dwt['ECG_T_Peaks'], y_filt[waves_dwt['ECG_T_Peaks']], "x", color='purple', label="T Peak")

                    # plt.xlabel('Time [ms]')
                    # plt.ylabel('Amplitude [mV]')
                    # plt.legend(loc="lower left")

                    # # Add annotations with error handling (seperti original)
                    # try:
                    #     # Limit annotations to prevent overcrowding
                    #     max_annotations = 5

                    #     # Annotate P peaks
                    #     if len(waves_dwt['ECG_P_Peaks']) > 0:
                    #         for i, j in zip(waves_dwt['ECG_P_Peaks'][:max_annotations],
                    #                     y_filt[waves_dwt['ECG_P_Peaks'][:max_annotations]]):
                    #             plt.annotate('P', xy=(i, j))

                    #     # Annotate Q peaks
                    #     if len(waves_dwt['ECG_Q_Peaks']) > 0:
                    #         for i, j in zip(waves_dwt['ECG_Q_Peaks'][:max_annotations],
                    #                     y_filt[waves_dwt['ECG_Q_Peaks'][:max_annotations]]):
                    #             plt.annotate('Q', xy=(i, j))

                    #     # Annotate R peaks
                    #     if len(rpeaks['ECG_R_Peaks']) > 0:
                    #         for i, j in zip(rpeaks['ECG_R_Peaks'][:max_annotations],
                    #                     y_filt[rpeaks['ECG_R_Peaks'][:max_annotations]]):
                    #             plt.annotate('R', xy=(i, j))

                    #     # Annotate S peaks
                    #     if len(waves_dwt['ECG_S_Peaks']) > 0:
                    #         for i, j in zip(waves_dwt['ECG_S_Peaks'][:max_annotations],
                    #                     y_filt[waves_dwt['ECG_S_Peaks'][:max_annotations]]):
                    #             plt.annotate('S', xy=(i, j))

                    #     # Annotate T peaks
                    #     if len(waves_dwt['ECG_T_Peaks']) > 0:
                    #         for i, j in zip(waves_dwt['ECG_T_Peaks'][:max_annotations],
                    #                     y_filt[waves_dwt['ECG_T_Peaks'][:max_annotations]]):
                    #             plt.annotate('T', xy=(i, j))
                    # except Exception as e:
                    #     print(f"Error adding annotations for {lead}: {e}")

                    # plt.tight_layout()

                except Exception as e:
                    print(f"Error in peak detection for {lead}: {e}")
                    # # Create a basic plot for the lead even if PQRST detection failed
                    # plt.figure(figsize=(10, 6))
                    # plt.title(f'Electrocardiogram Signal ({lead}) - Filtered Signal (PQRST detection failed)')
                    # plt.plot(t, y_filt, color='orange')
                    # plt.xlabel('Time [ms]')
                    # plt.ylabel('Amplitude [mV]')
                    # plt.grid(True)
                    # plt.tight_layout()
                

            except Exception as e:
                print(f"Error processing {lead}: {e}")

            except Exception as e:
                print(f"Error processing {lead}: {e}")
                # lanjut ke lead berikutnya

        # # ---- integrasi & prediksi SESUDAH SEMUA LEAD ----
        try:
           # Bangun DataFrame dengan NAMA KOLOM yang konsisten
            X = pd.DataFrame(
                [{k: packed.get(k, 0.0) for k in EXPECTED_FEATURES}],
                columns=EXPECTED_FEATURES
            ).astype(float)

            # Samakan SET & URUTAN kolom persis seperti saat scaler di-fit
            if hasattr(scaler, "feature_names_in_"):
                missing = [c for c in scaler.feature_names_in_ if c not in X.columns]
                extra   = [c for c in X.columns if c not in scaler.feature_names_in_]
                if missing:
                    print("⚠️ Kolom yang diharapkan scaler tapi tidak ada:", missing)
                if extra:
                    print("ℹ️ Kolom ekstra (akan diabaikan):", extra)

                # Reindex sesuai urutan scaler; kolom yang hilang diisi 0.0
                X = X.reindex(columns=scaler.feature_names_in_, fill_value=0.0)

            X_scaled = scaler.transform(X)
            # feature_for_model = feature_vector
            label = predict_clusters_xgb_integrated(
                X_scaled, model_xgb, kmeans_abnormal
            )[0]
        except Exception as e:
            print(f"❌ Gagal melakukan prediksi klaster ({os.path.basename(csv_path)}): {e}")
            label = "Unknown"

        return packed, label

    except Exception as e:
        # biar barisnya di-SKIP dengan pesan jelas
        raise RuntimeError(f"Gagal memproses file {csv_path}: {e}")

# ====== KONFIGURASI BATCH ======
# OPSIONAL di atas run_batch (dekat konfigurasi)
OVERWRITE_OUTPUT = False  # set True kalau mau overwrite OUT_CSV setiap run

# ====== LOOP SEMUA SUBFOLDER & FILE ======
def run_batch():
    # kolom final yang konsisten
    cols = ["subject"]+ EXPECTED_FEATURES + ["diagnostic_class"]

    # siapkan output
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    wrote_header = False
    if os.path.exists(OUT_CSV):
        if OVERWRITE_OUTPUT:
            os.remove(OUT_CSV)
            wrote_header = False
        else:
            wrote_header = True  # append ke file lama

    # telusuri semua subfolder (nama folder = subjek)
    for subject_dir in sorted([d for d in glob.glob(os.path.join(MAIN_DIR, "*")) if os.path.isdir(d)]):
        subject = os.path.basename(subject_dir)

        # ambil maksimal 20 file raw per subjek
        csv_files = sorted(glob.glob(os.path.join(subject_dir, "Data_*_raw.csv")))[:25]
        if not csv_files:
            continue

        for csv_path in csv_files:
            try:
                packed, label = extract_features_from_csv(csv_path)  # harus return (dict, str)

                # bentuk satu baris & jaga urutan kolom
                row = {"subject": subject, **packed, "diagnostic_class": label}
                df_row = pd.DataFrame([row])

                # pastikan semua kolom ada
                for c in cols:
                    if c not in df_row.columns:
                        df_row[c] = 0.0 if c in EXPECTED_FEATURES else ""
                df_row = df_row[cols]

                # simpan APPEND per file
                df_row.to_csv(
                    OUT_CSV,
                    mode="a",
                    header=not wrote_header,
                    index=False,
                    encoding="utf-8-sig"
                )
                wrote_header = True
                print(f"[SAVED] {subject} → {os.path.basename(csv_path)} → {label}")

            except Exception as e:
                print(f"[SKIP] {subject} / {os.path.basename(csv_path)}: {e}")

    print(f"\nSelesai. Terkumpul di: {OUT_CSV}")

if __name__ == "__main__":
    run_batch()
