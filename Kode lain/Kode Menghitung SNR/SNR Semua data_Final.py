import pandas as pd
import numpy as np
import scipy.signal
import scipy.fft
import matplotlib.pyplot as plt
import os
import pywt
import warnings

# Mencegah pesan warning yang tidak relevan saat data pendek
warnings.filterwarnings("ignore", category=UserWarning, message="Given sample size is too small for the given wavelet.")

# ==================================================================
# --- KONFIGURASI PATH ---
# ==================================================================
# 1. Tentukan path utama tempat semua folder data rekaman berada
BASE_PATH = r"D:\EKG\Skripsi Willy\Data ukur baru"

# 2. Tentukan path dan nama file untuk output Excel
OUTPUT_DIR = r"D:\EKG\Skripsi Willy\Data ukur baru\Hasil SNR"
OUTPUT_FILENAME = "Ringkasan_SNR_Seluruh_Folder.xlsx"
# ==================================================================


# ==================================================================
# --- FUNGSI HELPER (Tidak ada perubahan di sini) ---
# ==================================================================
VREF = 2.42
GAIN = 200
RESOLUTION = 24
LSB_SIZE = VREF / (2**RESOLUTION - 1)

def to_mV(adc):
    return adc * LSB_SIZE * 1000.0 / GAIN

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
    return xi, mask

ADC_EDGE = 0.95 * (2**23 - 1)
def desaturate_edges(x):
    x = np.asarray(x, dtype=float)
    mask = (np.abs(x) >= ADC_EDGE)
    xi = x.copy()
    if np.any(mask):
        idx = np.arange(len(x))
        xi[mask] = np.interp(idx[mask], idx[~mask], x[~mask])
    return xi, mask


def estimate_snr(signal, fs):
    """
    Mengestimasi SNR dengan asumsi noise berada di atas frekuensi 40 Hz.
    """
    try:
        nyquist = fs / 2.0
        b, a = scipy.signal.butter(4, 40 / nyquist, btype='low')
        estimated_signal_component = scipy.signal.filtfilt(b, a, signal)
        estimated_noise_component = signal - estimated_signal_component
        power_signal = np.mean(estimated_signal_component ** 2)
        power_noise = np.mean(estimated_noise_component ** 2)
        if power_noise == 0:
            return float('inf')
        snr = 10 * np.log10(power_signal / power_noise)
        return snr
    except Exception as e:
        print(f"Error calculating SNR: {e}")
        return np.nan

def apply_iec_diagnostic_filter(signal, fs):
    if len(signal) < int(fs * 4) + 1: # Cek jika sinyal terlalu pendek
        return signal
    pad_len = int(fs * 2)
    padded_signal = np.pad(signal, (pad_len, pad_len), mode='edge')
    nyquist = fs / 2.0
    
    hp_cutoff = 0.05 / nyquist
    b_hp, a_hp = scipy.signal.butter(4, min(hp_cutoff, 0.99), btype='highpass')
    signal_hp = scipy.signal.filtfilt(b_hp, a_hp, padded_signal)
    
    lp_cutoff = 150 / nyquist
    if (lp_cutoff >= 1.0): lp_cutoff = 0.99
    b_lp, a_lp = scipy.signal.butter(4, lp_cutoff, btype='lowpass')
    signal_lp = scipy.signal.filtfilt(b_lp, a_lp, signal_hp)
    
    f0 = 50.0
    if f0 < nyquist:
        w0 = f0 / nyquist
        b_notch, a_notch = scipy.signal.iirnotch(w0, 30.0)
        signal_filtered = scipy.signal.filtfilt(b_notch, a_notch, signal_lp)
    else:
        signal_filtered = signal_lp
        
    return signal_filtered[pad_len:-pad_len]

def wavelet_denoise(x, fs, wavelet='sym6', level=None):
    x = np.asarray(x, dtype=float)
    try:
        w = pywt.Wavelet(wavelet)
        maxlev = pywt.dwt_max_level(len(x), w.dec_len)
        if maxlev == 0: return x # Sinyal terlalu pendek untuk didekomposisi
        if level is None: level = min(5, maxlev)

        coeffs = pywt.wavedec(x, w, mode='symmetric', level=level)
        cA = coeffs[0]
        details = coeffs[1:]
        new_coeffs = [cA]
        
        for i, cD in enumerate(details, start=1):
            f_high = fs / (2.0 ** i)
            sigma = np.median(np.abs(cD - np.median(cD))) / 0.6745 + 1e-12
            thr = sigma * np.sqrt(2.0 * np.log(max(cD.size, 2)))
            
            if f_high <= 40.0:
                thr *= 0.5
            
            cD_th = pywt.threshold(cD, thr, mode='soft')
            new_coeffs.append(cD_th)

        y = pywt.waverec(new_coeffs, w, mode='symmetric')
        return y[:len(x)]
    except ValueError: # Menangani error jika sinyal terlalu pendek
        return x

# ==================================================================
# --- PEMROSESAN UTAMA UNTUK SEMUA FOLDER ---
# ==================================================================
os.makedirs(OUTPUT_DIR, exist_ok=True)
final_summary = []

# Iterasi melalui setiap item di base path
for folder_name in sorted(os.listdir(BASE_PATH)):
    folder_path = os.path.join(BASE_PATH, folder_name)
    
    # Lanjutkan hanya jika itu adalah sebuah folder
    if os.path.isdir(folder_path):
        print(f"--- Memproses Folder: {folder_name} ---")
        
        folder_snr_before = []
        folder_snr_after = []
        
        # Iterasi melalui setiap file di dalam folder
        for file_name in os.listdir(folder_path):
            # Lanjutkan hanya jika file adalah file CSV
            if file_name.lower().endswith('.csv'):
                file_path = os.path.join(folder_path, file_name)
                print(f"  -> Membaca file: {file_name}")
                
                try:
                    dataset = pd.read_csv(file_path, names=["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"], sep=',', skiprows=1)
                    leads = dataset.columns.tolist()

                    for lead in leads:
                        ecg_adc = dataset[lead].values.astype(float)
                        if len(ecg_adc) < 200: continue # Lewati data yang sangat pendek

                        ecg_adc, _ = desaturate_edges(ecg_adc)
                        ecg_adc, _ = despike_hampel(ecg_adc, k=7, nsigma=6.0)
                        ecgmv = to_mV(ecg_adc)
                        
                        Fs = int(len(dataset) / 10)
                        if Fs < 1: Fs = 500


                        # Tahap 1: Detrend
                        s1_detrended = scipy.signal.detrend(ecgmv, type='constant')
                        
                        # Hitung SNR dari sinyal mentah 
                        snr_before = estimate_snr(s1_detrended, Fs)
                        
                        # Tahap 2: IEC Filter
                        s2_iec_filtered = apply_iec_diagnostic_filter(s1_detrended, Fs)
                        
                        # Tahap 3: Wavelet Denoising
                        s3_final_wavelet = wavelet_denoise(s2_iec_filtered, Fs)

                        # Hitung SNR setelah semua filter
                        snr_after = estimate_snr(s3_final_wavelet, Fs)
                        
                        # Tambahkan hasil ke list folder jika valid
                        if np.isfinite(snr_before):
                            folder_snr_before.append(snr_before)
                        if np.isfinite(snr_after):
                            folder_snr_after.append(snr_after)

                except Exception as e:
                    print(f"    [ERROR] Gagal memproses file {file_name}: {e}")

        # Setelah semua file dalam satu folder diproses, hitung rata-ratanya
        if folder_snr_before and folder_snr_after:
            avg_snr_before = np.mean(folder_snr_before)
            avg_snr_after = np.mean(folder_snr_after)
            
            final_summary.append({
                'nama folder': folder_name,
                'snr sebelum filter': avg_snr_before,
                'snr sesudah filter': avg_snr_after
            })
            print(f"-> Rata-rata SNR Folder '{folder_name}': Sebelum={avg_snr_before:.2f} dB, Sesudah={avg_snr_after:.2f} dB\n")
        else:
            print(f"-> Tidak ada data CSV valid yang ditemukan di folder '{folder_name}'.\n")

# --- Konversi hasil akhir ke DataFrame dan simpan ke Excel ---
summary_df = pd.DataFrame(final_summary)
output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
summary_df.to_excel(output_path, index=False, float_format='%.2f')

print("\n--- SELESAI ---")
print(f"Ringkasan SNR untuk semua folder telah disimpan ke:\n{output_path}")
print("\nIsi file Excel:")
print(summary_df.to_string(index=False))

# Bagian plotting dan penyimpanan FFT/gambar individual telah dinonaktifkan
# agar skrip berjalan cepat tanpa menampilkan jendela plot.