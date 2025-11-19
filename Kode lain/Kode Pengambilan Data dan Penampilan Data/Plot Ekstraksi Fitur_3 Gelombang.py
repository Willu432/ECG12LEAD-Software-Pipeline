import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import scipy.signal
import math
import neurokit2 as nk
import pywt
import os

# ==================================================================
# --- 1. KONFIGURASI ---
# ==================================================================

path_data_input = r"D:\EKG\Skripsi Willy\Perbandingan dengan alat klinis\willy_Tserial"
nama_file_input = "Data_2_raw.csv"
path_lengkap_input = os.path.join(path_data_input, nama_file_input)

MAX_BEATS_VISUAL = 3   # tampilkan maksimal 3 penandaan/segmen dalam window (boleh 2 jika mau)

# ==================================================================
# --- 2. FUNGSI-FUNGSI PEMROSESAN SINYAL ---
# ==================================================================

def to_mV(adc, vref=2.42, gain=200, resolution=24):
    lsb_size = vref / (2**resolution - 1)
    return adc * lsb_size * 1000.0 / gain

def despike_hampel(x, k=7, nsigma=6.0):
    x = np.asarray(x, dtype=float)
    med = scipy.signal.medfilt(x, kernel_size=2*k+1)
    diff = np.abs(x - med)
    mad = scipy.signal.medfilt(diff, kernel_size=2*k+1)
    mad = np.maximum(mad, 1e-12)
    mask = diff > (nsigma * 1.4826 * mad)
    xi = x.copy()
    if np.any(mask):
        idx = np.arange(len(x))
        xi[mask] = np.interp(idx[mask], idx[~mask], x[~mask])
    return xi

def desaturate_edges(x, adc_edge_factor=0.95, resolution=24):
    adc_edge = adc_edge_factor * (2**(resolution - 1) - 1)
    x = np.asarray(x, dtype=float)
    mask = (np.abs(x) >= adc_edge)
    xi = x.copy()
    if np.any(mask):
        idx = np.arange(len(x))
        xi[mask] = np.interp(idx[mask], idx[~mask], x[~mask])
    return xi

def apply_iec_diagnostic_filter(signal, fs):
    pad_len = int(fs * 2)
    padded_signal = np.pad(signal, (pad_len, pad_len), mode='edge')
    nyquist = fs / 2.0
    hp_cutoff = min(0.05 / nyquist, 0.99)
    b_hp, a_hp = scipy.signal.butter(4, hp_cutoff, btype='highpass')
    signal_hp = scipy.signal.filtfilt(b_hp, a_hp, padded_signal)
    lp_cutoff = min(150 / nyquist, 0.99)
    b_lp, a_lp = scipy.signal.butter(4, lp_cutoff, btype='lowpass')
    signal_lp = scipy.signal.filtfilt(b_lp, a_lp, signal_hp)
    f0 = 50.0
    if f0 < nyquist:
        w0 = f0 / nyquist
        b_notch, a_notch = scipy.signal.iirnotch(w0, Q=30.0)
        signal_filtered = scipy.signal.filtfilt(b_notch, a_notch, signal_lp)
    else:
        signal_filtered = signal_lp
    return signal_filtered[pad_len:-pad_len]

def wavelet_denoise(x, fs, wavelet='sym6', level=None):
    x = np.asarray(x, dtype=float)
    w = pywt.Wavelet(wavelet)
    maxlev = pywt.dwt_max_level(len(x), w.dec_len)
    level = min(5, maxlev) if level is None else level
    coeffs = pywt.wavedec(x, w, mode='symmetric', level=level)
    new_coeffs = [coeffs[0]]
    for cD in coeffs[1:]:
        sigma = (np.median(np.abs(cD - np.median(cD))) / 0.6745)
        thr = sigma * np.sqrt(2.0 * np.log(len(x)))
        cD_th = pywt.threshold(cD, thr, mode='soft')
        new_coeffs.append(cD_th)
    y = pywt.waverec(new_coeffs, w, mode='symmetric')
    return y[:len(x)]

# ==================================================================
# --- 3. UTIL EKSTRAKSI FITUR DARI FIDUSIAL ---
# ==================================================================

def compute_intervals_and_st(peaks, fs, signal):
    R = np.asarray(peaks.get('R', []), dtype=float)
    P = np.asarray(peaks.get('P', []), dtype=float)
    Q = np.asarray(peaks.get('Q', []), dtype=float)
    S = np.asarray(peaks.get('S', []), dtype=float)
    T = np.asarray(peaks.get('T', []), dtype=float)

    def valid_int(a):
        return np.array([int(x) for x in a if not np.isnan(x) and 0 <= x < len(signal)], dtype=int)

    R = valid_int(R); P = valid_int(P); Q = valid_int(Q); S = valid_int(S); T = valid_int(T)

    rr_ms = []
    for i in range(1, len(R)):
        rr_ms.append((R[i]-R[i-1]) * 1000.0/fs)

    pr_ms = []; pr_segments = []
    for r in R:
        p_candidates = P[P < r]
        q_candidates = Q[Q < r]
        if len(p_candidates) == 0:
            continue
        p = p_candidates[-1]
        q = q_candidates[-1] if len(q_candidates) > 0 else r
        dur = max(0, (q - p)*1000.0/fs)
        pr_ms.append(dur)
        pr_segments.append((p, q))

    qrs_ms = []; qrs_segments = []
    for r in R:
        q_candidates = Q[Q <= r]
        s_candidates = S[S >= r]
        if len(q_candidates)==0 or len(s_candidates)==0:
            continue
        q = q_candidates[-1]
        s = s_candidates[0]
        dur = max(0, (s - q)*1000.0/fs)
        qrs_ms.append(dur)
        qrs_segments.append((q, s))

    qt_ms = []; qt_segments = []
    for r in R:
        q_candidates = Q[Q <= r]
        t_candidates = T[T >= r]
        if len(q_candidates)==0 or len(t_candidates)==0:
            continue
        q = q_candidates[-1]
        t = t_candidates[0]
        dur = max(0, (t - q)*1000.0/fs)
        qt_ms.append(dur)
        qt_segments.append((q, t))

    qtc_ms = []
    if len(rr_ms) > 0 and len(qt_ms) > 0:
        rr_s = np.array(rr_ms)/1000.0
        for i in range(min(len(qt_ms), len(rr_s))):
            qtc_ms.append(qt_ms[i] / math.sqrt(rr_s[i]))

    hr_vals = []
    for rr in rr_ms:
        if rr > 0:
            hr_vals.append(60_000.0/rr)

    st_points = []; st_baselines = []
    for r in R:
        s_candidates = S[S >= r]
        j_idx = s_candidates[0] if len(s_candidates)>0 else r
        j60 = j_idx + int(0.060*fs)
        if j60 >= len(signal):
            continue
        q_candidates = Q[Q <= r]
        if len(q_candidates)==0:
            continue
        q = q_candidates[-1]
        pq_end = max(0, q-10)
        pq_start = max(0, pq_end - int(0.040*fs))
        if pq_end <= pq_start:
            continue
        baseline = np.mean(signal[pq_start:pq_end])
        st_amp = signal[j60] - baseline
        st_points.append((j60, st_amp))
        st_baselines.append((pq_start, pq_end, baseline))

    summary = {
        "RR_mean_ms": float(np.nanmean(rr_ms)) if len(rr_ms) else np.nan,
        "PR_mean_ms": float(np.nanmean(pr_ms)) if len(pr_ms) else np.nan,
        "QRS_mean_ms": float(np.nanmean(qrs_ms)) if len(qrs_ms) else np.nan,
        "QT_mean_ms": float(np.nanmean(qt_ms)) if len(qt_ms) else np.nan,
        "QTc_mean_ms": float(np.nanmean(qtc_ms)) if len(qtc_ms) else np.nan,
        "HR_mean_bpm": float(np.nanmean(hr_vals)) if len(hr_vals) else np.nan,
        "n_beats": len(rr_ms)+1 if len(rr_ms) else 0
    }

    segments = {"PR": pr_segments, "QRS": qrs_segments, "QT": qt_segments,
                "ST_points": st_points, "ST_baselines": st_baselines}
    return summary, segments

# ==================================================================
# --- 3b. FUNGSI: Pilih window 2 detak dari Lead II ---
# ==================================================================

def select_two_beat_window_from_leadII(df, fs, margin_ms=120, fallback_ms=2000):
    ecg_adc = df['II'].values
    ecg_desat = desaturate_edges(ecg_adc)
    ecg_despiked = despike_hampel(ecg_desat)
    ecg_mv = to_mV(ecg_despiked)
    ecg_pre = scipy.signal.detrend(ecg_mv, type='constant')
    ecg_iec = apply_iec_diagnostic_filter(ecg_pre, fs)
    ecg_final = wavelet_denoise(ecg_iec, fs)

    try:
        _, rpeaks = nk.ecg_peaks(ecg_final, sampling_rate=fs)
        R = rpeaks.get('ECG_R_Peaks', [])
    except Exception:
        R = []

    if R is None: R = []
    R = [int(r) for r in R if not np.isnan(r) and 0 <= r < len(ecg_final)]

    if len(R) >= 3:
        start = max(0, R[0] - int(margin_ms/1000*fs))
        end   = min(len(ecg_final), R[2] + int(margin_ms/1000*fs))
    else:
        end = min(len(ecg_final), int(fallback_ms/1000*fs))
        start = 0
    return start, end

# ==================================================================
# --- 4. FUNGSI UTAMA PROSES & PLOTTING (6-grid 2×3) + Notasi P/Q/R/S/T ---
# ==================================================================

def _box_axes(ax):
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(1.4)

def _annotate_peaks(ax, idx_win, signal, fs, label, max_labels=3):
    """
    Tulis huruf P/Q/R/S/T dekat marker 'x'. Dibatasi max_labels per lead.
    """
    if not idx_win:
        return
    # Ambil paling banyak max_labels titik (yang pertama kali muncul di window)
    idx_win = idx_win[:max_labels]
    for k in idx_win:
        x_ms = (k / fs) * 1000.0
        y = signal[k]
        # offset kecil agar teks tidak menimpa marker
        ax.annotate(label, xy=(x_ms, y),
                    xytext=(0, 10), textcoords='offset points',
                    fontsize=9, ha='center', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.7))

def process_and_plot_leads_6grid(dataset, fs, leads6, figure_title, start_idx, end_idx, max_beats_visual=2):
    assert len(leads6) == 6, "leads6 harus berisi tepat 6 lead"

    fig, axs = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
    axs = axs.ravel()
    fig.suptitle(f'{figure_title} (≤2 PQRST)', fontsize=18, fontweight='bold')

    t_ms = (np.arange(start_idx, end_idx) / fs) * 1000.0

    for i, lead_name in enumerate(leads6):
        ax = axs[i]

        # ---------- Pra-filter ----------
        ecg_adc = dataset[lead_name].values
        ecg_desat = desaturate_edges(ecg_adc)
        ecg_despiked = despike_hampel(ecg_desat)
        ecg_mv = to_mV(ecg_despiked)         # (bug-fix) jangan ditimpa lagi
        ecg_pre = scipy.signal.detrend(ecg_mv, type='constant')

        # ---------- Final ----------
        ecg_iec = apply_iec_diagnostic_filter(ecg_pre, fs)
        ecg_final = wavelet_denoise(ecg_iec, fs)

        # Window
        y_pre   = ecg_pre[start_idx:end_idx]
        y_final = ecg_final[start_idx:end_idx]

        # ---------- PQRST (untuk overlay & anotasi) ----------
        try:
            _, rpeaks = nk.ecg_peaks(ecg_final, sampling_rate=fs)
            _, waves_dwt = nk.ecg_delineate(ecg_final, rpeaks, sampling_rate=fs, method="dwt")
            peaks = {
                'P': waves_dwt.get('ECG_P_Peaks', []),
                'Q': waves_dwt.get('ECG_Q_Peaks', []),
                'R': rpeaks.get('ECG_R_Peaks', []),
                'S': waves_dwt.get('ECG_S_Peaks', []),
                'T': waves_dwt.get('ECG_T_Peaks', [])
            }
        except Exception:
            peaks = {'P': [], 'Q': [], 'R': [], 'S': [], 'T': []}

        summary, segments = compute_intervals_and_st(peaks, fs, ecg_final)

        # ---------- Plot ----------
        ax.plot(t_ms, y_pre,  color='0.7',   linewidth=1.0, label='Pre (detrend+Hampel)')
        ax.plot(t_ms, y_final, color='orange', linewidth=1.4, label='Final (IEC+Wavelet)')

        # Marker & Notasi puncak di dalam window
        pk_colors = {'P': 'blue', 'Q': 'green', 'R': 'black', 'S': 'red', 'T': 'purple'}
        for lbl, idxs in peaks.items():
            if idxs is None:
                continue
            idx_abs = [int(k) for k in idxs if 0 <= k < len(ecg_final)]
            idx_win = [k for k in idx_abs if start_idx <= k < end_idx]
            if idx_win:
                ax.plot((np.array(idx_win)/fs*1000.0), ecg_final[idx_win], "x", ms=5, color=pk_colors[lbl])
                # --> Tambah notasi P/Q/R/S/T
                _annotate_peaks(ax, idx_win, ecg_final, fs, lbl, max_labels=MAX_BEATS_VISUAL)

        # Shade segmen (maks MAX_BEATS_VISUAL)
        def _shade(seglist, alpha):
            shown = 0
            for (a,b) in seglist:
                if a >= start_idx and b <= end_idx:
                    ax.axvspan(a/fs*1000.0, b/fs*1000.0, alpha=alpha)
                    shown += 1
                    if shown >= MAX_BEATS_VISUAL:
                        break
        _shade(segments['PR'],  0.15)
        _shade(segments['QRS'], 0.20)
        _shade(segments['QT'],  0.12)

        # Titik ST (maks MAX_BEATS_VISUAL) + baseline PQ
        st_shown = 0
        for j, (j60, _st) in enumerate(segments['ST_points']):
            if start_idx <= j60 < end_idx:
                pq_start, pq_end, base = segments['ST_baselines'][j]
                ax.hlines(base, pq_start/fs*1000.0, pq_end/fs*1000.0, linestyles='dashed', lw=0.9, alpha=0.7)
                ax.plot(j60/fs*1000.0, ecg_final[j60], 's', ms=5)
                st_shown += 1
                if st_shown >= MAX_BEATS_VISUAL:
                    break

        rr = summary["RR_mean_ms"]; hr = summary["HR_mean_bpm"]
        ttl = lead_name
        if not np.isnan(hr): ttl += f"  | HR≈{hr:.1f} bpm"
        if not np.isnan(rr): ttl += f"  | RR≈{rr:.0f} ms"
        ax.set_title(ttl, fontsize=11)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("mV")
        ax.grid(True, linestyle='--', alpha=0.6)
        _box_axes(ax)

    # Legend global (opsional)
    handles, labels = axs[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper right', ncols=2, fontsize=9, frameon=True)

    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    return fig

# ==================================================================
# --- 5. EKSEKUSI UTAMA ---
# ==================================================================

try:
    dataset = pd.read_csv(
        path_lengkap_input,
        names=["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"],
        sep=',', skiprows=1, dtype=float
    )
    SAMPLING_RATE = float(len(dataset) / 10)   # Hz

    print(f"Dataset berhasil dimuat. Jumlah sampel: {len(dataset)}")
    print(f"Menggunakan Sampling Frequency (Fs) = {SAMPLING_RATE} Hz")

    # Tentukan window 2 detak dari lead II
    start_idx, end_idx = select_two_beat_window_from_leadII(dataset, SAMPLING_RATE)
    dur_ms = (end_idx-start_idx)/SAMPLING_RATE*1000.0
    print(f"Window 2-beat: start={start_idx}, end={end_idx} (durasi ≈ {dur_ms:.0f} ms)")

    limb_aug6 = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF']
    prec6     = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    print("\nMemulai plotting 6-grid (≤2 PQRST)...")
    print("-" * 50)

    process_and_plot_leads_6grid(dataset, SAMPLING_RATE, limb_aug6,
                                 "Limb + Augmented Leads", start_idx, end_idx, max_beats_visual=MAX_BEATS_VISUAL)

    process_and_plot_leads_6grid(dataset, SAMPLING_RATE, prec6,
                                 "Precordial Leads V1–V6", start_idx, end_idx, max_beats_visual=MAX_BEATS_VISUAL)

    print("-" * 50)
    print("✅ Semua plot berhasil dibuat. Menampilkan hasil...")
    plt.show()

except FileNotFoundError:
    print(f"❌ ERROR: File tidak ditemukan di '{path_lengkap_input}'")
except Exception as e:
    print(f"❌ Terjadi error: {e}")
