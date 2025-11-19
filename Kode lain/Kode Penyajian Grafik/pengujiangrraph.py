import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

file_path = r"D:\EKG\Skripsi Willy\Perbandingan dengan alat klinis\Jeffry_Tserial\Data_3_raw.csv"
try:
    dataset = pd.read_csv(file_path, names=["i", "ii", "iii", "avr", "avl", "avf", 
                                            "v1", "v2", "v3", "v4", "v5", "v6"], 
                           sep=',', skiprows=1, dtype=float)
    t = np.arange(0, len(dataset))

    # Function to disable scientific notation and show full integers
    def disable_sci_not(axs):
        for ax in axs:
            ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))
            ax.ticklabel_format(style='plain', axis='y')  # no scientific notation
            ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.0f}'))  # no decimals

    # --- Window 1: Limb Leads ---
    fig1, axs1 = plt.subplots(3, 1, figsize=(15, 9), sharex=True)
    fig1.suptitle('Limb Leads (I, II, III)', fontsize=24)
    axs1[0].plot(t, dataset['i'], label='I', color='blue'); axs1[0].set_ylabel('I (mV)'); axs1[0].legend(); axs1[0].grid(True)
    axs1[1].plot(t, dataset['ii'], label='II', color='green'); axs1[1].set_ylabel('II (mV)'); axs1[1].legend(); axs1[1].grid(True)
    axs1[2].plot(t, dataset['iii'], label='III', color='red'); axs1[2].set_ylabel('III (mV)'); axs1[2].set_xlabel('Sample'); axs1[2].legend(); axs1[2].grid(True)
    disable_sci_not(axs1)
    fig1.tight_layout(rect=[0, 0.03, 1, 0.95])

    # --- Window 2: Augmented Limb Leads ---
    fig2, axs2 = plt.subplots(3, 1, figsize=(15, 9), sharex=True)
    fig2.suptitle('Augmented Limb Leads (AVR, AVL, AVF)', fontsize=24)
    axs2[0].plot(t, dataset['avr'], label='AVR', color='blue'); axs2[0].set_ylabel('AVR (mV)'); axs2[0].legend(); axs2[0].grid(True)
    axs2[1].plot(t, dataset['avl'], label='AVL', color='green'); axs2[1].set_ylabel('AVL (mV)'); axs2[1].legend(); axs2[1].grid(True)
    axs2[2].plot(t, dataset['avf'], label='AVF', color='red'); axs2[2].set_ylabel('AVF (mV)'); axs2[2].set_xlabel('Sample'); axs2[2].legend(); axs2[2].grid(True)
    disable_sci_not(axs2)
    fig2.tight_layout(rect=[0, 0.03, 1, 0.95])

    # --- Window 3: Precordial Leads (V1-V3) ---
    fig3, axs3 = plt.subplots(3, 1, figsize=(15, 9), sharex=True)
    fig3.suptitle('Precordial Leads (V1, V2, V3)', fontsize=24)
    axs3[0].plot(t, dataset['v1'], label='V1', color='blue'); axs3[0].set_ylabel('V1 (mV)'); axs3[0].legend(); axs3[0].grid(True)
    axs3[1].plot(t, dataset['v2'], label='V2', color='green'); axs3[1].set_ylabel('V2 (mV)'); axs3[1].legend(); axs3[1].grid(True)
    axs3[2].plot(t, dataset['v3'], label='V3', color='red'); axs3[2].set_ylabel('V3 (mV)'); axs3[2].set_xlabel('Sample'); axs3[2].legend(); axs3[2].grid(True)
    disable_sci_not(axs3)
    fig3.tight_layout(rect=[0, 0.03, 1, 0.95])

    # --- Window 4: Precordial Leads (V4-V6) ---
    fig4, axs4 = plt.subplots(3, 1, figsize=(15, 9), sharex=True)
    fig4.suptitle('Precordial Leads (V4, V5, V6)', fontsize=24)
    axs4[0].plot(t, dataset['v4'], label='V4', color='blue'); axs4[0].set_ylabel('V4 (mV)'); axs4[0].legend(); axs4[0].grid(True)
    axs4[1].plot(t, dataset['v5'], label='V5', color='green'); axs4[1].set_ylabel('V5 (mV)'); axs4[1].legend(); axs4[1].grid(True)
    axs4[2].plot(t, dataset['v6'], label='V6', color='red'); axs4[2].set_ylabel('V6 (mV)'); axs4[2].set_xlabel('Sample'); axs4[2].legend(); axs4[2].grid(True)
    disable_sci_not(axs4)
    fig4.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.show()

except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")
