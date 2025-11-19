import pandas as pd
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

def estimate_snr(signal, fs):
    """
    Mengestimasi SNR dengan asumsi noise berada di atas frekuensi 40 Hz.
    Metode ini digunakan untuk sinyal mentah dan sinyal terfilter.

    Args:
        signal (np.array): Sinyal input (bisa mentah atau terfilter).
        fs (int): Frekuensi sampling.

    Returns:
        float: Estimasi SNR dalam dB.
    """
    try:
        nyquist = fs / 2.0
        # Buat filter low-pass untuk mengisolasi 'sinyal' (di bawah 40Hz)
        # Ini adalah definisi 'sinyal' kita untuk metode ini
        b, a = scipy.signal.butter(4, 40 / nyquist, btype='low')
        estimated_signal_component = scipy.signal.filtfilt(b, a, signal)
        
        # Noise adalah sisa dari sinyal input
        estimated_noise_component = signal - estimated_signal_component
        
        power_signal = np.mean(estimated_signal_component ** 2)
        power_noise = np.mean(estimated_noise_component ** 2)

        if power_noise == 0:
            return float('inf') # Terjadi jika sinyal sudah sangat halus

        snr = 10 * np.log10(power_signal / power_noise)
        return snr
    except Exception as e:
        # Fallback jika ada error pada filter
        print(f"Error calculating SNR: {e}")
        return np.nan

# Load data
try:
    # Path diubah sesuai permintaan
    dataset = pd.read_csv(r"D:\EKG\Ekstraksi Fitur 12 Lead\Anak WJ\Test rekam 10 duduk (Jeffry)/Data_3_raw.csv", names=["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"], sep=',', skiprows=1)
except FileNotFoundError:
    print("File 'Data_3_raw.csv' tidak ditemukan. Menggunakan data dummy.")
    data = np.random.randn(5000, 12)
    dataset = pd.DataFrame(data, columns=["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"])

# Inisialisasi
leads = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
snr_results = []

def apply_iec_diagnostic_filter(signal, fs):
    signal = signal - np.mean(signal)
    pad_len = int(fs * 2)
    padded_signal = np.pad(signal, (pad_len, pad_len), mode='edge')
    nyquist = fs / 2.0
    
    hp_cutoff = 0.05 / nyquist
    b_hp, a_hp = scipy.signal.butter(4, min(hp_cutoff, 0.99), btype='highpass')
    signal_hp = scipy.signal.filtfilt(b_hp, a_hp, padded_signal)
    
    lp_cutoff = 150 / nyquist
    b_lp, a_lp = scipy.signal.butter(4, min(lp_cutoff, 0.99), btype='lowpass')
    signal_lp = scipy.signal.filtfilt(b_lp, a_lp, signal_hp)
    
    f0 = 50.0
    if f0 < nyquist:
        w0 = f0 / nyquist
        Q = 30.0
        b_notch, a_notch = scipy.signal.iirnotch(w0, Q)
        signal_filtered = scipy.signal.filtfilt(b_notch, a_notch, signal_lp)
    else:
        signal_filtered = signal_lp
        
    return signal_filtered[pad_len:-pad_len]

# Proses setiap lead
for lead in leads:
    ecgmv = (dataset[lead].astype(float)) * (2.42 / ((2**24))) * 1000
    Fs = int(len(dataset) / 10)
    if Fs < 81: Fs = 500 # Pastikan Fs cukup tinggi untuk estimasi

    # --- HITUNG SNR SEBELUM FILTER (SETELAH DETREND) ---
    # Menghilangkan baseline wander dari sinyal mentah sebelum estimasi SNR
    ecgmv_detrended = scipy.signal.detrend(ecgmv.values, type='linear')
    snr_before = estimate_snr(ecgmv_detrended, Fs)

    # --- PROSES FILTERING LENGKAP ---
    iec_filtered = apply_iec_diagnostic_filter(ecgmv.values, Fs)
    detr_ecg = scipy.signal.detrend(iec_filtered, type='linear')
    b, a = scipy.signal.butter(4, 0.6, 'low')
    tempf_butter = scipy.signal.filtfilt(b, a, detr_ecg)
    nyq_rate = Fs / 2.0
    width = 5.0 / nyq_rate
    ripple_db = 60.0
    O, beta = scipy.signal.kaiserord(ripple_db, width)
    if O % 2 == 0: O += 1
    taps = scipy.signal.firwin(O, 0.05 / nyq_rate, window=('kaiser', beta), pass_zero=False)
    final_filtered_signal = scipy.signal.lfilter(taps, 1.0, tempf_butter)

    # --- HITUNG SNR SESUDAH FILTER (MENGGUNAKAN METODE YANG SAMA) ---
    snr_after = estimate_snr(final_filtered_signal, Fs)

    # Simpan hasil
    snr_results.append({
        'Lead': lead,
        'SNR Sebelum Filter (dB)': snr_before,
        'SNR Sesudah Filter (dB)': snr_after
    })

# Tampilkan hasil dalam DataFrame
snr_df = pd.DataFrame(snr_results)
print("Perbandingan SNR (Metode Konsisten, Sinyal Mentah di-Detrend):")
print(snr_df.to_string(index=False))

# --- SIMPAN HASIL KE FILE EXCEL ---
# Pastikan Anda telah menginstal library openpyxl: pip install openpyxl
# Path diubah sesuai permintaan
output_filename_excel = r'D:\EKG\Skripsi Willy/hasil_snr_konsisten_detrended.xlsx'
snr_df.to_excel(output_filename_excel, index=False, float_format='%.2f', engine='openpyxl')
print(f"\nHasil telah disimpan ke file: {output_filename_excel}")
