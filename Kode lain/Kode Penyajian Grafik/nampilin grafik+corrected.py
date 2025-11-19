import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import scipy.signal

# ==================================================================
# --- FUNGSI-FUNGSI PEMROSESAN SINYAL ---
# ==================================================================

def despike_hampel(x, k=7, nsigma=6.0):
    """
    Hampel filter untuk menghilangkan lonjakan (spike) tajam.
    """
    x = np.asarray(x, dtype=float)
    med = scipy.signal.medfilt(x, kernel_size=2*k+1)
    diff = np.abs(x - med)
    mad = scipy.signal.medfilt(diff, kernel_size=2*k+1)
    mad = np.maximum(mad, 1e-12)  # Mencegah pembagian dengan nol
    mask = diff > (nsigma * 1.4826 * mad)
    
    xi = x.copy()
    if np.any(mask):
        idx = np.arange(len(x))
        # Interpolasi linear pada titik-titik outlier
        xi[mask] = np.interp(idx[mask], idx[~mask], x[~mask])
    return xi

def desaturate_edges(x, resolution=24, edge_factor=0.95):
    """
    Mendeteksi dan memperbaiki sampel yang mengalami saturasi ADC.
    """
    # Menghitung batas saturasi (misal 95% dari nilai maksimum 24-bit)
    ADC_EDGE = edge_factor * (2**(resolution - 1) - 1)
    x = np.asarray(x, dtype=float)
    mask = (np.abs(x) >= ADC_EDGE)
    
    xi = x.copy()
    if np.any(mask):
        idx = np.arange(len(x))
        # Interpolasi linear pada titik-titik saturasi
        xi[mask] = np.interp(idx[mask], idx[~mask], x[~mask])
    return xi

# ==================================================================
# --- EKSEKUSI UTAMA ---
# ==================================================================

file_path = r"D:\EKG\Ekstraksi Fitur 12 Lead\Anak WJ\Test rekam 27 tidur(Willy)\Data_5_raw.csv"

try:
    # 1. Membaca data mentah dari CSV
    dataset = pd.read_csv(file_path, names=["I", "II", "III", "AVR", "AVL", "AVF", 
                                             "V1", "V2", "V3", "V4", "V5", "V6"], 
                                   sep=',', skiprows=1, dtype=float)
    t = np.arange(0, len(dataset))
    print(f"Data mentah berhasil dimuat. Jumlah sampel: {len(dataset)}")

    # 2. Menerapkan koreksi Desaturate dan Despike pada setiap lead
    dataset_corrected = pd.DataFrame()
    for lead in dataset.columns:
        signal_raw = dataset[lead].values
        # Terapkan perbaikan secara berurutan: pertama saturasi, lalu spike
        signal_desaturated = desaturate_edges(signal_raw)
        signal_despiked = despike_hampel(signal_desaturated)
        dataset_corrected[lead] = signal_despiked
    print("Proses Desaturate dan Despike selesai untuk semua 12 lead.")

    # 3. Plotting data yang sudah dikoreksi
    
    # Fungsi helper untuk format plot
    def disable_sci_not(axs):
        for ax in axs:
            ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))
            ax.ticklabel_format(style='plain', axis='y')
            ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.0f}'))

    # --- Window 1: Limb Leads ---
    fig1, axs1 = plt.subplots(3, 1, figsize=(15, 9), sharex=True)
    fig1.suptitle('Limb Leads (I, II, III) - Raw', fontsize=24)
    axs1[0].plot(t, dataset_corrected['I'], label='I', color='blue'); axs1[0].set_ylabel('I ADC Value'); axs1[0].legend(); axs1[0].grid(True)
    axs1[1].plot(t, dataset_corrected['II'], label='II', color='green'); axs1[1].set_ylabel('II ADC Value'); axs1[1].legend(); axs1[1].grid(True)
    axs1[2].plot(t, dataset_corrected['III'], label='III', color='red'); axs1[2].set_ylabel('III ADC Value'); axs1[2].set_xlabel('Sample'); axs1[2].legend(); axs1[2].grid(True)
    disable_sci_not(axs1)
    fig1.tight_layout(rect=[0, 0.03, 1, 0.95])

    # --- Window 2: Augmented Limb Leads ---
    fig2, axs2 = plt.subplots(3, 1, figsize=(15, 9), sharex=True)
    fig2.suptitle('Augmented Limb Leads (AVR, AVL, AVF) - Raw', fontsize=24)
    axs2[0].plot(t, dataset_corrected['AVR'], label='AVR', color='blue'); axs2[0].set_ylabel('AVR ADC Value'); axs2[0].legend(); axs2[0].grid(True)
    axs2[1].plot(t, dataset_corrected['AVL'], label='AVL', color='green'); axs2[1].set_ylabel('AVL ADC Value'); axs2[1].legend(); axs2[1].grid(True)
    axs2[2].plot(t, dataset_corrected['AVF'], label='AVF', color='red'); axs2[2].set_ylabel('AVF ADC Value'); axs2[2].set_xlabel('Sample'); axs2[2].legend(); axs2[2].grid(True)
    disable_sci_not(axs2)
    fig2.tight_layout(rect=[0, 0.03, 1, 0.95])

    # --- Window 3: Precordial Leads (V1-V3) ---
    fig3, axs3 = plt.subplots(3, 1, figsize=(15, 9), sharex=True)
    fig3.suptitle('Precordial Leads (V1, V2, V3) - Raw', fontsize=24)
    axs3[0].plot(t, dataset_corrected['V1'], label='V1', color='blue'); axs3[0].set_ylabel('V1 ADC Value'); axs3[0].legend(); axs3[0].grid(True)
    axs3[1].plot(t, dataset_corrected['V2'], label='V2', color='green'); axs3[1].set_ylabel('V2 ADC Value'); axs3[1].legend(); axs3[1].grid(True)
    axs3[2].plot(t, dataset_corrected['V3'], label='V3', color='red'); axs3[2].set_ylabel('V3 ADC Value'); axs3[2].set_xlabel('Sample'); axs3[2].legend(); axs3[2].grid(True)
    disable_sci_not(axs3)
    fig3.tight_layout(rect=[0, 0.03, 1, 0.95])

    # --- Window 4: Precordial Leads (V4-V6) ---
    fig4, axs4 = plt.subplots(3, 1, figsize=(15, 9), sharex=True)
    fig4.suptitle('Precordial Leads (V4, V5, V6) - Raw', fontsize=24)
    axs4[0].plot(t, dataset_corrected['V4'], label='V4', color='blue'); axs4[0].set_ylabel('V4 ADC Value'); axs4[0].legend(); axs4[0].grid(True)
    axs4[1].plot(t, dataset_corrected['V5'], label='V5', color='green'); axs4[1].set_ylabel('V5 ADC Value'); axs4[1].legend(); axs4[1].grid(True)
    axs4[2].plot(t, dataset_corrected['V6'], label='V6', color='red'); axs4[2].set_ylabel('V6 ADC Value'); axs4[2].set_xlabel('Sample'); axs4[2].legend(); axs4[2].grid(True)
    disable_sci_not(axs4)
    fig4.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.show()

except FileNotFoundError:
    print(f"Error: File '{file_path}' tidak ditemukan.")
except Exception as e:
    print(f"An error occurred: {e}")