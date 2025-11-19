import pandas as pd
import numpy as np
import scipy.signal
import scipy.fft
import matplotlib.pyplot as plt
import os
import pywt

# <<< ADD: constants konversi ADC → mV (sama seperti real-time)
VREF = 2.42       # volt
GAIN = 200        # sesuai setelan ADS1293
RESOLUTION = 24
LSB_SIZE = VREF / (2**RESOLUTION - 1)

def to_mV(adc):
    return adc * LSB_SIZE * 1000.0 / GAIN

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


def estimate_snr(signal, fs):
    """
    Mengestimasi SNR dengan asumsi noise berada di atas Frequency 40 Hz.
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

def calculate_fft(signal, fs):
    """
    Menghitung Fast Fourier Transform (FFT) dari sebuah sinyal.
    """
    N = len(signal)
    T = 1.0 / fs
    yf = scipy.fft.fft(signal)
    xf = scipy.fft.fftfreq(N, T)[:N//2]
    yf_abs = 2.0/N * np.abs(yf[0:N//2])
    return xf, yf_abs

def apply_iec_diagnostic_filter(signal, fs):
    """
    Menerapkan filter diagnostik EKG sesuai standar IEC 60601-2-25:2011.
    """
    # signal = signal - np.mean(signal)
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
        # Perkiraan pita Frequency level-i (dyadik)
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


# --- SETUP ---
output_dir = r'D:\EKG\Skripsi Willy'
os.makedirs(output_dir, exist_ok=True)

try:
    dataset = pd.read_csv(r"D:\EKG\Ekstraksi Fitur 12 Lead\Anak WJ\Test rekam 27 tidur(Willy)\Data_5_raw.csv", names=["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"], sep=',', skiprows=1)
except FileNotFoundError as e:
    print(f"Error: File tidak ditemukan di path yang diberikan.\n{e}")
    dataset = pd.DataFrame(np.random.randn(5000, 12), columns=["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"])
    print("Menggunakan data dummy untuk demonstrasi.")

leads = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
snr_results = []
fft_data_storage = {}

# --- PEMROSESAN UTAMA ---
for lead in leads:
    print(f"Memproses Lead {lead}...")
    # ecgmv = (dataset[lead].astype(float)) * (2.42 / ((2**24))) * 1000
    ecg_adc = dataset[lead].values.astype(float)
    ecg_adc, mask_sat = desaturate_edges(ecg_adc)        # <<< ADD: hapus sampel saturasi
    ecg_adc, mask_spk = despike_hampel(ecg_adc, k=7, nsigma=6.0)  # <<< ADD: hapus spike tajam
    ecgmv = to_mV(ecg_adc)                                # hasil dalam mV
    Fs = int(len(dataset) / 10)
    # if Fs < 81: Fs = 500

    # --- Tahap 0: Sinyal Mentah (After Detrend) ---
    s0_raw_detrended = scipy.signal.detrend(ecgmv, type='constant')
    snr_before = estimate_snr(s0_raw_detrended, Fs)

    # --- Tahap 1: IEC Filter ---
    s1_iec_filtered = apply_iec_diagnostic_filter(s0_raw_detrended, Fs)

    # --- Tahap 2: wavelet denoise (After IEC) ---
    s2_Wavelet = wavelet_denoise(s1_iec_filtered, Fs, wavelet='sym6', level=None, method='soft')


    snr_after = estimate_snr( s2_Wavelet, Fs)
    snr_results.append({
        'Lead': lead,
        'SNR Before Filter (dB)': snr_before,
        'SNR Sesudah Filter (dB)': snr_after
    })

    # --- Hitung FFT untuk semua tahapan ---
    fft_data_storage[lead] = {
        'r0': calculate_fft(ecgmv, Fs),  
        's0': calculate_fft(s0_raw_detrended, Fs),
        's1': calculate_fft(s1_iec_filtered, Fs),
        's2': calculate_fft(s2_Wavelet, Fs),
    }

# --- PENYIMPANAN DATA KE EXCEL ---
output_fft_excel = os.path.join(output_dir, 'hasil_fft_lengkap.xlsx')
with pd.ExcelWriter(output_fft_excel, engine='openpyxl') as writer:
    for lead in leads:
        data = fft_data_storage[lead]
        fft_df = pd.DataFrame({
            'Frequency (Hz)': data['s0'][0],
            'Amplitude_Raw_Detrended': data['s0'][1],
            'Amplitude_IEC_Filtered': data['s1'][1],
            'Amplitude_Wavelet Denoising': data['s2'][1],
        })
        fft_df.to_excel(writer, sheet_name=f'Lead_{lead}', index=False, float_format='%.5f')
print(f"\nData spektral lengkap telah disimpan ke: {output_fft_excel}")

snr_df = pd.DataFrame(snr_results)
output_snr_excel = os.path.join(output_dir, 'hasil_snr_konsisten_detrended.xlsx')
snr_df.to_excel(output_snr_excel, index=False, float_format='%.2f', engine='openpyxl')
print(f"Perbandingan SNR telah disimpan ke: {output_snr_excel}")
# --- PLOTTING: LEAD II, 4 FIGURE TERPISAH ---
lead = 'II'
data = fft_data_storage[lead]

Fs = int(len(dataset) / 10)
if Fs < 81:
    Fs = 500

# ===== Figure 1: RAW (Undetrended) vs Detrended =====
plt.figure(figsize=(12, 4))
plt.title('Phase 1: The Effect of Baseline Correction - Lead II')

# Perbandingan utama yang diminta:
plt.plot(data['r0'][0], data['r0'][1], label='R0: Raw (Undetrended)', alpha=0.8)
plt.plot(data['s0'][0], data['s0'][1], label='S0: After Baseline Correction', linewidth=1.5)

# # (Opsional) tampilkan juga S0 sebagai referensi tipis
# plt.plot(data['s0'][0], data['s0'][1], label='S0: Raw (Detrended)', alpha=0.4, linestyle='--')

# # Garis bantu cutoff
# plt.axvline(x=0.05, linestyle='--', label='HP Cutoff (0.05 Hz)')
# plt.axvline(x=150, linestyle='--')            # LP cutoff (150 Hz)
# plt.axvline(x=50, linestyle=':', label='Notch (50 Hz)')

plt.yscale('log'); plt.grid(True, which="both", ls="--"); plt.xlim(0, Fs/2)
plt.xlabel('Frequency (Hz)'); plt.ylabel('Amplitude (Log)'); plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'LeadII_Tahap1_RAW_vs_BaselineCorr.png'), dpi=200)
plt.show()

# ===== Figure 2: IEC vs Detrend =====
plt.figure(figsize=(12, 4))
plt.title('Phase 2: The Effect IEC Diagnostic Filter - Lead II')
plt.plot(data['s0'][0], data['s0'][1], label='S0: Before IEC Diagnostic Filter', alpha=0.7)
plt.plot(data['s1'][0], data['s1'][1], label='S1: After IEC Diagnostic Filter', linewidth=1.5)

plt.axvline(x=50, linestyle=':', color='gray', label='Notch (50 Hz)')
plt.axvline(x=0.05, linestyle='--', color='red', label='HP Cutoff (0.05 Hz)')
plt.axvline(x=150, linestyle='--', color='green', label='LP Cutoff (150 Hz)')
plt.yscale('log'); plt.grid(True, which="both", ls="--"); plt.xlim(0, Fs/2)
plt.xlabel('Frequency (Hz)'); plt.ylabel('Amplitude (Log)'); plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f'Lead{lead}_Tahap2_IEC.png.png'), dpi=200)
plt.show()


# ===== Figure 3: wavelet denoising =====
butterworth_cutoff_hz = 0.6 * (Fs / 2)  # 0.6 × Nyquist
plt.figure(figsize=(12, 4))
plt.title('Phase 3: The Effect of Wavelet Denoising - Lead II')
plt.plot(data['s1'][0], data['s1'][1], label='S2: Before Wavelet Denoising', alpha=0.7)
plt.plot(data['s2'][0], data['s2'][1], label='S3: After Wavelet Denoising', linewidth=1.5)
plt.yscale('log'); plt.grid(True, which="both", ls="--"); plt.xlim(0, Fs/2)
plt.xlabel('Frequency (Hz)'); plt.ylabel('Amplitude (Log)'); plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'LeadII_Tahap3_Wavelet Denoising.png'), dpi=200)
plt.show()

print("\n--- Selesai ---")
print("Perbandingan SNR (Metode Konsisten, Sinyal Mentah di-Detrend):")
print(snr_df.to_string(index=False))
