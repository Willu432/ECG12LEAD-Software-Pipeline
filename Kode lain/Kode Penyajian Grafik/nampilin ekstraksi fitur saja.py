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

# Path ke file .csv yang akan dianalisis
path_data_input = r"D:\EKG\Ekstraksi Fitur 12 Lead\Anak WJ\Test rekam 27 tidur(Willy)"
nama_file_input = "Data_5_raw.csv"
path_lengkap_input = os.path.join(path_data_input, nama_file_input)

# PENTING: Tentukan Sampling Frequency (Fs) alat Anda di sini!
# Ganti nilai 500 dengan Fs yang sebenarnya.
SAMPLING_RATE = 500  # Contoh: 500 Hz

# ==================================================================
# --- 2. FUNGSI-FUNGSI PEMROSESAN SINYAL (TIDAK PERLU DIUBAH) ---
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
# --- 3. FUNGSI UTAMA UNTUK PEMROSESAN & PLOTTING ---
# ==================================================================

def process_and_plot_leads(dataset, t_axis, leads_group, group_name, fs):
    """
    Fungsi utama untuk memproses dan membuat satu figure plot 
    untuk sekelompok lead (misal: Limb Leads).
    """
    fig, axs = plt.subplots(len(leads_group), 1, figsize=(18, 10), sharex=True)
    if len(leads_group) == 1:
        axs = [axs]
    fig.suptitle(f'{group_name} - Filtered Signal with PQRST Detection', fontsize=18, fontweight='bold')
    
    for i, lead_name in enumerate(leads_group):
        print(f"Memproses {lead_name}...")
        
        # --- Tahap 1 & 2: Pre-processing & Filtering ---
        ecg_adc = dataset[lead_name].values
        ecg_desaturated = desaturate_edges(ecg_adc)
        ecg_despiked = despike_hampel(ecg_desaturated)
        ecg_mv = to_mV(ecg_despiked)
        ecg_detrended = scipy.signal.detrend(ecg_mv, type='constant')
        ecg_iec_filtered = apply_iec_diagnostic_filter(ecg_detrended, fs)
        ecg_final = wavelet_denoise(ecg_iec_filtered, fs)

        # --- Tahap 3: Deteksi Puncak PQRST ---
        try:
            _, rpeaks = nk.ecg_peaks(ecg_final, sampling_rate=fs)
            _, waves_dwt = nk.ecg_delineate(ecg_final, rpeaks, sampling_rate=fs, method="dwt")
            peaks = {
                'P': waves_dwt.get('ECG_P_Peaks', []), 'Q': waves_dwt.get('ECG_Q_Peaks', []),
                'R': rpeaks.get('ECG_R_Peaks', []),    'S': waves_dwt.get('ECG_S_Peaks', []),
                'T': waves_dwt.get('ECG_T_Peaks', [])
            }
        except Exception as e:
            print(f"  -> Gagal mendeteksi PQRST untuk {lead_name}: {e}")
            peaks = {}

        # --- Tahap 4: Plotting (SESUAI PERMINTAAN) ---
        ax = axs[i]
        ax.plot(t_axis, ecg_final, color='orange', linewidth=1.5, label='Filtered Data')
        
        peak_colors = {'P': 'blue', 'Q': 'green', 'R': 'black', 'S': 'red', 'T': 'purple'}
        
        for label, indices in peaks.items():
            valid_indices = [int(idx) for idx in indices if not np.isnan(idx) and 0 <= idx < len(ecg_final)]
            if valid_indices:
                # Plot semua puncak dengan marker 'x'
                ax.plot(valid_indices, ecg_final[valid_indices], "x", color=peak_colors[label], markersize=7, label=f"{label} Peak")
                
                # Tambahkan anotasi simpel untuk setiap puncak
                for idx in valid_indices:
                    ax.annotate(label, xy=(idx, ecg_final[idx]), fontsize=10, fontweight='bold', ha='center', va='bottom')

        ax.set_ylabel(f'{lead_name} [mV]', fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc='upper right')

    axs[-1].set_xlabel('Sample Index', fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])

# ==================================================================
# --- 4. EKSEKUSI UTAMA ---
# ==================================================================

try:
    dataset = pd.read_csv(path_lengkap_input, names=["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"], sep=',', skiprows=1, dtype=float)
    t = np.arange(len(dataset))

    print(f"Dataset berhasil dimuat. Jumlah sampel: {len(dataset)}")
    print(f"Menggunakan Sampling Frequency (Fs) = {SAMPLING_RATE} Hz")

    limb_leads = ['I', 'II', 'III']
    augmented_leads = ['AVR', 'AVL', 'AVF']
    precordial_v1_v3 = ['V1', 'V2', 'V3']
    precordial_v4_v6 = ['V4', 'V5', 'V6']
    
    print("\nMemulai pemrosesan dan plotting sinyal EKG...")
    print("-" * 50)
    
    process_and_plot_leads(dataset, t, limb_leads, "Limb Leads", SAMPLING_RATE)
    process_and_plot_leads(dataset, t, augmented_leads, "Augmented Limb Leads", SAMPLING_RATE)
    process_and_plot_leads(dataset, t, precordial_v1_v3, "Precordial Leads (V1-V3)", SAMPLING_RATE)
    process_and_plot_leads(dataset, t, precordial_v4_v6, "Precordial Leads (V4-V6)", SAMPLING_RATE)
    
    print("-" * 50)
    print("✅ Semua plot berhasil dibuat. Menampilkan hasil...")
    
    plt.show()

except FileNotFoundError:
    print(f"❌ ERROR: File tidak ditemukan di '{path_lengkap_input}'")
except Exception as e:
    print(f"❌ Terjadi error: {e}")