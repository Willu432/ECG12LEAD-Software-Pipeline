import serial
import time
import pandas as pd
import os
import sys
import numpy as np
from time import sleep
import struct # <-- Pastikan library ini diimpor

# ==================================================================
# --- PATH CONFIGURATION ---
# ==================================================================
# 1. Direktori utama untuk menyimpan rekaman
path_utama = r"D:\EKG\Skripsi Willy\Perbandingan dengan alat klinis"
# 2. Nama sub-folder untuk sesi perekaman ini
nama_subjek = "Jeffry_TSerial" # Ganti nama folder
# 3. Path lengkap tempat file CSV akan disimpan
save_path = os.path.join(path_utama, nama_subjek)

# ==================================================================
# --- SERIAL COMMUNICATION SETUP ---
# ==================================================================
# Delimiter frame Start of Text (STX) dan End of Text (ETX)
STX, ETX = 0x02, 0x03
# Ukuran data EKG yang diharapkan (12 integer * 4 byte/integer)
EXPECTED_FRAME_SIZE = 48
# Ukuran payload = data + checksum (48 data + 1 checksum)
EXPECTED_PAYLOAD_SIZE = 49

# Buffer untuk menampung data serial yang masuk
buf = bytearray()

# --- Buka Port Serial ---
# PENTING: Ganti 'COM10' dengan COM port ESP32 Anda.
try:
    serialPort = serial.Serial(port='COM23', baudrate=921600)
except serial.SerialException as e:
    print(f"Error opening serial port: {e}")
    print("Please check the COM port and ensure the device is connected.")
    sys.exit(1) # Keluar dari script jika port tidak bisa dibuka

# ==================================================================
# --- DATA CONVERSION SETUP ---
# ==================================================================
# Konstanta dari datasheet ADS1293 untuk konversi ADC ke milivolt
VREF = 2.42      # Tegangan referensi dalam volt
GAIN = 200       # Gain
RESOLUTION = 24  # Resolusi 24-bit
LSB_SIZE = VREF / (2**RESOLUTION - 1)

def convert_to_millivolts(adc_value):
    """Mengonversi nilai ADC integer mentah ke milivolt (mV)."""
    return adc_value * LSB_SIZE * 1000 / GAIN

# ==================================================================
# --- MAIN PROGRAM ---
# ==================================================================
try:
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"Created directory: {save_path}")

    print("Program started. Listening for data...")
    timerecord = 10
    iteration = 1
    
    # Variabel untuk melacak status sinkronisasi
    frames_processed = 0
    checksum_errors = 0
    framing_errors = 0

    while True:
        raw_data = {"I": [], "II": [], "III": [], "AVR": [], "AVL": [], "AVF": [], "V1": [], "V2": [],"V3": [], "V4": [], "V5": [], "V6": [] }
        converted_data = {"I-mV": [], "II-mV": [], "III-mV": [], "AVR-mV": [], "AVL-mV": [], "AVF-mV": [], "V1-mV": [], "V2-mV": [], "V3-mV": [], "V4-mV": [], "V5-mV": [], "V6-mV": []}

        print(f"\n--- Starting Iteration {iteration} (Recording for {timerecord} seconds) ---")
        t_end = time.time() + timerecord

        while time.time() < t_end:
            chunk = serialPort.read(serialPort.in_waiting or 4096)
            if not chunk:
                continue
            buf.extend(chunk)

            while True:
                start_index = buf.find(STX)
                if start_index == -1:
                    break # Tidak ada STX, tunggu data lagi

                # Posisi akhir frame yang diharapkan (STX + PAYLOAD + ETX)
                end_index = start_index + EXPECTED_PAYLOAD_SIZE + 1
                if len(buf) <= end_index:
                    break # Data belum cukup untuk satu frame penuh
                
                if buf[end_index] == ETX:
                    # Struktur frame (STX...ETX) valid, sekarang validasi checksum
                    payload = buf[start_index + 1 : end_index]
                    frame_data = payload[:EXPECTED_FRAME_SIZE]
                    received_checksum = payload[-1]

                    # Hitung checksum dari data yang diterima
                    calculated_checksum = sum(frame_data) & 0xFF

                    if calculated_checksum == received_checksum:
                        # Checksum cocok! Data valid.
                        del buf[:end_index + 1] # Hapus frame dari buffer
                        frames_processed += 1
                        try:
                            vals = struct.unpack('<12i', frame_data)
                            print(f"Received values: {vals}") # <-- PENAMBAHAN BARIS INI
                            
                            (v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12) = vals

                            raw_data["I"].append(v1); raw_data["II"].append(v2); raw_data["III"].append(v3)
                            raw_data["AVR"].append(v4); raw_data["AVL"].append(v5); raw_data["AVF"].append(v6)
                            raw_data["V1"].append(v7); raw_data["V2"].append(v8); raw_data["V3"].append(v9)
                            raw_data["V4"].append(v10); raw_data["V5"].append(v11); raw_data["V6"].append(v12)

                            converted_data["I-mV"].append(convert_to_millivolts(v1))
                            converted_data["II-mV"].append(convert_to_millivolts(v2))
                            converted_data["III-mV"].append(convert_to_millivolts(v3))
                            converted_data["AVR-mV"].append(convert_to_millivolts(v4))
                            converted_data["AVL-mV"].append(convert_to_millivolts(v5))
                            converted_data["AVF-mV"].append(convert_to_millivolts(v6))
                            converted_data["V1-mV"].append(convert_to_millivolts(v7))
                            converted_data["V2-mV"].append(convert_to_millivolts(v8))
                            converted_data["V3-mV"].append(convert_to_millivolts(v9))
                            converted_data["V4-mV"].append(convert_to_millivolts(v10))
                            converted_data["V5-mV"].append(convert_to_millivolts(v11))
                            converted_data["V6-mV"].append(convert_to_millivolts(v12))
                        except Exception as e:
                            print(f"Error processing valid frame: {e}")
                    else:
                        # Checksum tidak cocok! Data rusak. Buang frame ini.
                        checksum_errors += 1
                        del buf[:start_index + 1]
                else:
                    # Framing error: STX ditemukan tapi ETX tidak pada posisi yang benar.
                    # Buang byte STX palsu ini dan cari lagi.
                    framing_errors += 1
                    del buf[:start_index + 1]

        # --- SIMPAN DATA UNTUK ITERASI YANG SELESAI ---
        print(f"Iteration {iteration} finished.")
        print(f"  - Frames processed: {frames_processed}")
        print(f"  - Checksum errors: {checksum_errors}")
        print(f"  - Framing errors: {framing_errors}")
        
        # Reset counters for next iteration
        frames_processed, checksum_errors, framing_errors = 0, 0, 0

        if any(len(v) > 0 for v in raw_data.values()):
            df_raw = pd.DataFrame(raw_data)
            file_path_raw = os.path.join(save_path, f'Data_{iteration}_raw.csv')
            df_raw.to_csv(file_path_raw, index=False)
            print(f'✅ Raw data saved ({len(df_raw)} samples) at {file_path_raw}.')

            df_converted = pd.DataFrame(converted_data)
            file_path_converted = os.path.join(save_path, f'Data_{iteration}_converted.csv')
            df_converted.to_csv(file_path_converted, index=False)
            print(f'✅ Converted mV data saved at {file_path_converted}.')
        else:
            print(f"⚠️ No data was collected in iteration {iteration}. Skipping file save.")

        iteration += 1
        sleep(2)

except KeyboardInterrupt:
    print("\nProgram stopped by user.")
except Exception as e:
    print(f"\nAn unexpected error occurred: {e}")
finally:
    if 'serialPort' in locals() and serialPort.is_open:
        serialPort.close()
        print("Serial port closed.")
    print("Program finished.")

