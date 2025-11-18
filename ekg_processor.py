# File: ekg_processor.py

import socket
import struct
import time
import pandas as pd
import numpy as np
import scipy.signal
import math
import neurokit2 as nk
import joblib
import os
import pywt
import xgboost as xgb 
from time import sleep

# ==================================================================
# --- KELAS UNTUK MENGELOLA PEMROSESAN EKG ---
# ==================================================================
class EKGProcessor:
    def __init__(self, status_callback, log_callback):
        self.client_socket = None
        self.is_running = False
        self.buf = bytearray()
        
        self.scaler = None
        self.model_xgb = None
        self.kmeans_abnormal = None

        self.update_status = status_callback
        self.log_message = log_callback
        # TAMBAHAN: Callback untuk live plot
        self.plot_callback = None


    def _load_models(self, model_path):
        try:
            self.log_message("Memuat model machine learning...")
            scaler_path = os.path.join(model_path, 'scaler.joblib')
            xgb_path = os.path.join(model_path, 'model_xgb_binary.json')
            kmeans_path = os.path.join(model_path, 'kmeans_abnormal_model.joblib')

            self.scaler = joblib.load(scaler_path)
            # Catatan: Jika .json adalah model XGBoost, joblib mungkin tidak bisa membukanya.
            # Anda mungkin perlu library xgboost:
            # import xgboost as xgb
            self.model_xgb = xgb.XGBClassifier()
            self.model_xgb.load_model(xgb_path)
            # self.model_xgb = joblib.load(xgb_path)
            self.kmeans_abnormal = joblib.load(kmeans_path)

            self.log_message("Model berhasil dimuat.")
            self.update_status("Model berhasil dimuat.")
            return True
        except FileNotFoundError as e:
            msg = f"Error: File model tidak ditemukan - {e}"
            self.log_message(msg)
            self.update_status(msg)
            return False
        except Exception as e:
            msg = f"Error saat memuat model: {e}"
            self.log_message(msg)
            self.update_status(msg)
            return False

    def connect_to_esp32(self, ip, port):
        try:
            self.log_message(f"Menghubungkan ke {ip}:{port}...")
            self.update_status(f"Menghubungkan ke {ip}:{port}...")
            self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client_socket.settimeout(5.0)
            self.client_socket.connect((ip, port))
            self.client_socket.settimeout(1.0)
            msg = "✅ Koneksi berhasil!"
            self.log_message(msg)
            self.update_status(msg)
            return True
        except socket.error as e:
            msg = f"Error koneksi: {e}"
            self.log_message(msg)
            self.update_status(msg)
            self.client_socket = None
            return False

    def stop_processing(self):
        self.is_running = False
        self.log_message("Proses perekaman dihentikan oleh pengguna.")
        self.update_status("Proses perekaman dihentikan.")

    def close_connection(self):
        if self.client_socket:
            self.client_socket.close()
            self.client_socket = None
            self.log_message("Koneksi TCP ditutup.")

    def start_main_loop(self, config):
        self.is_running = True
        
        if not self._load_models(config['path_model']):
            self.is_running = False
            return

        iteration = 1
        while self.is_running:
            self.log_message(f"\n--- Memulai Iterasi {iteration} ({config['timerecord']} detik) ---")
            self.update_status(f"Iterasi {iteration}: Mengambil data...")
            
            raw_data, converted_data = self._gather_data(config['timerecord'])
            
            if not self.is_running: break
            if not any(len(v) > 0 for v in raw_data.values()):
                self.log_message(f"⚠️ Tidak ada data pada iterasi {iteration}. Melewati...")
                iteration += 1
                sleep(2)
                continue

            save_path = os.path.join(config['path_utama'], config['nama_subjek'])
            if not os.path.exists(save_path): os.makedirs(save_path)
            
            df_raw = pd.DataFrame(raw_data)
            df_raw.to_csv(os.path.join(save_path, f'Data_{iteration}_raw.csv'), index=False)
            df_converted = pd.DataFrame(converted_data)
            df_converted.to_csv(os.path.join(save_path, f'Data_{iteration}_converted.csv'), index=False)
            self.log_message(f"Data mentah iterasi {iteration} disimpan.")

            self.update_status(f"Iterasi {iteration}: Memproses sinyal...")
            packedData = self._process_all_leads(converted_data)

            self.update_status(f"Iterasi {iteration}: Melakukan prediksi...")
            cluster_label = self._predict_cluster(packedData)
            if cluster_label:
                packedData["cluster_label"] = cluster_label
                self.log_message(f"-> Hasil Prediksi: {cluster_label}")

            self.update_status(f"Iterasi {iteration}: Menyimpan hasil...")
            path_lengkap_excel = os.path.join(save_path, config['nama_file_excel'])
            self._save_to_excel(packedData, iteration, path_lengkap_excel)
            self.log_message(f"Hasil iterasi {iteration} disimpan ke Excel.")

            iteration += 1
            sleep(2)

        self.close_connection()
        self.update_status("Selesai. Siap untuk memulai rekaman baru.")
    
    def _gather_data(self, timerecord):
        Raw_data = { "I": [], "II": [], "III": [], "AVR": [], "AVL": [], "AVF": [], "V1": [], "V2": [],"V3": [], "V4": [], "V5": [], "V6": [] }
        converted_data = { "I-mV": [], "II-mV": [], "III-mV": [], "AVR-mV": [], "AVL-mV": [], "AVF-mV": [], "V1-mV": [], "V2-mV": [], "V3-mV": [], "V4-mV": [], "V5-mV": [], "V6-mV": [] }
        t_end = time.time() + timerecord
        STX, ETX, EXPECTED_FRAME_SIZE, EXPECTED_PAYLOAD_SIZE = 0x02, 0x03, 48, 49
        
        while time.time() < t_end and self.is_running:
            try:
                chunk = self.client_socket.recv(4096)
                if not chunk:
                    self.log_message("Koneksi ditutup oleh perangkat.")
                    self.is_running = False
                    break
                
                self.buf.extend(chunk)

                while True:
                    start_index = self.buf.find(STX)
                    if start_index == -1: break
                    
                    end_index = start_index + EXPECTED_PAYLOAD_SIZE + 1
                    if len(self.buf) <= end_index: break

                    if self.buf[end_index] == ETX:
                        payload = self.buf[start_index + 1 : end_index]
                        frame_data = payload[:EXPECTED_FRAME_SIZE]
                        received_checksum = payload[-1]
                        calculated_checksum = sum(frame_data) & 0xFF

                        if calculated_checksum == received_checksum:
                            del self.buf[:end_index + 1]
                            vals = struct.unpack('<12i', frame_data)
                            (v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12) = vals
                            Raw_data["I"].append(v1); Raw_data["II"].append(v2); Raw_data["III"].append(v3); Raw_data["AVR"].append(v4); Raw_data["AVL"].append(v5); Raw_data["AVF"].append(v6); Raw_data["V1"].append(v7); Raw_data["V2"].append(v8); Raw_data["V3"].append(v9); Raw_data["V4"].append(v10); Raw_data["V5"].append(v11); Raw_data["V6"].append(v12)
                            # TAMBAHAN: Kirim data ke live plot callback
                            if self.plot_callback:
                                try:
                                    plot_data = {
                                        "I": v1, "II": v2, "III": v3, "AVR": v4,
                                        "AVL": v5, "AVF": v6, "V1": v7, "V2": v8,
                                        "V3": v9, "V4": v10, "V5": v11, "V6": v12
                                    }
                                    self.plot_callback(plot_data)
                                except Exception as e:
                                    pass  # Jangan ganggu proses utama jika plot error
                            converted_data["I-mV"].append(convert_to_millivolts(v1)); converted_data["II-mV"].append(convert_to_millivolts(v2)); converted_data["III-mV"].append(convert_to_millivolts(v3)); converted_data["AVR-mV"].append(convert_to_millivolts(v4)); converted_data["AVL-mV"].append(convert_to_millivolts(v5)); converted_data["AVF-mV"].append(convert_to_millivolts(v6)); converted_data["V1-mV"].append(convert_to_millivolts(v7)); converted_data["V2-mV"].append(convert_to_millivolts(v8)); converted_data["V3-mV"].append(convert_to_millivolts(v9)); converted_data["V4-mV"].append(convert_to_millivolts(v10)); converted_data["V5-mV"].append(convert_to_millivolts(v11)); converted_data["V6-mV"].append(convert_to_millivolts(v12))
                            # # TAMBAHAN: Kirim data ke live plot callback dalam mV
                            # if self.plot_callback:
                            #     try:
                            #         plot_data = {
                            #             "I": convert_to_millivolts(v1), 
                            #             "II": convert_to_millivolts(v2), 
                            #             "III": convert_to_millivolts(v3), 
                            #             "AVR": convert_to_millivolts(v4),
                            #             "AVL": convert_to_millivolts(v5), 
                            #             "AVF": convert_to_millivolts(v6), 
                            #             "V1": convert_to_millivolts(v7), 
                            #             "V2": convert_to_millivolts(v8),
                            #             "V3": convert_to_millivolts(v9), 
                            #             "V4": convert_to_millivolts(v10), 
                            #             "V5": convert_to_millivolts(v11), 
                            #             "V6": convert_to_millivolts(v12)
                            #         }
                            #         self.plot_callback(plot_data)
                            #     except Exception as e:
                            #         pass  # Jangan ganggu proses utama jika plot error
                        else: del self.buf[:start_index + 1]
                        
                    else: del self.buf[:start_index + 1]
            except socket.timeout: continue
            except Exception as e:
                self.log_message(f"Error saat mengambil data: {e}")
                break
        return Raw_data, converted_data

    def _process_all_leads(self, converted_data):
        self.log_message("\nMemulai ekstraksi fitur untuk 12 Lead...")
        packedData = {}
        Fs = 853
        leads = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]

        for lead in leads:
            channel = f"{lead}-mV"
            lead_lower = lead.lower()
            ecgmv = np.asarray(converted_data[channel], dtype=float)
            
            if len(ecgmv) < Fs / 2: # Skip jika data kurang dari 0.5 detik
                self.log_message(f"Data {lead} tidak cukup untuk diproses.")
                continue

            # --- FULL PROCESSING PIPELINE ---
            num_spike = np.sum(np.abs(ecgmv - despike_pipeline(ecgmv, Fs)) > 1e-9)
            if num_spike > 0:
                self.log_message(f"⚠️  {channel}: {num_spike} sampel dikoreksi (spike/saturasi).")

            ecgmv_clean = despike_pipeline(ecgmv, Fs)
            # ecgmv_clean = ecgmv # no despike
            detr_ecg = scipy.signal.detrend(ecgmv_clean, axis=-1, type='constant')
            iec_filtered_signal = apply_iec_diagnostic_filter(detr_ecg, Fs)
            y_filt = wavelet_denoise(iec_filtered_signal, Fs)
            
            # --- FEATURE EXTRACTION ---
            try:
                _, rpeaks = nk.ecg_peaks(y_filt, sampling_rate=Fs)
                _, waves_dwt = nk.ecg_delineate(y_filt, rpeaks, sampling_rate=Fs, method="dwt")

                # --- The entire correction and feature extraction logic from your script ---
                (RR_avg, RR_stdev, PR_avg, qrs_avg, QT_avg, QTc_avg, 
                 ST_avg_amplitude, ST_avg_deviation, RS_ratio, bpm) = self._extract_features_for_lead(y_filt, Fs, rpeaks, waves_dwt, lead)

                # Store features
                packedData[f"rr_{lead_lower}"] = RR_avg
                packedData[f"rr_std_{lead_lower}"] = RR_stdev
                packedData[f"pr_{lead_lower}"] = PR_avg
                packedData[f"qrs_{lead_lower}"] = qrs_avg
                packedData[f"qtc_{lead_lower}"] = QTc_avg
                packedData[f"st_amplitude_mv_{lead_lower}"] = ST_avg_amplitude
                packedData[f"st_deviation_mv_{lead_lower}"] = ST_avg_deviation
                packedData[f"rs_ratio_{lead_lower}"] = RS_ratio
                packedData[f"heartrate_{lead_lower}"] = bpm

                # Log results to GUI
                self.log_message(f'\n--- Hasil Ekstraksi Fitur untuk Lead {lead} ---')
                self.log_message(f'  RR Interval (ms) - Mean: {RR_avg:.2f}, Std Dev: {RR_stdev:.2f}')
                self.log_message(f'  PR Interval (ms) - Mean: {PR_avg:.2f}')
                self.log_message(f'  QRS Interval (ms) - Mean: {qrs_avg:.2f}')
                self.log_message(f'  QTc Interval (ms) - Mean: {QTc_avg:.2f}')
                self.log_message(f'  ST Amplitude (mV) - Mean: {ST_avg_amplitude:.5f}')
                self.log_message(f'  ST Deviation (mV) - Mean: {ST_avg_deviation:.5f}')
                self.log_message(f'  R/S Ratio: {RS_ratio:.4f}')
                self.log_message(f'  BPM: {bpm:.2f}')

            except Exception as e:
                self.log_message(f"Error saat memproses {lead}: {e}")
        
        return packedData

    def _extract_features_for_lead(self, y_filt, Fs, rpeaks, waves_dwt, lead_name):
        # This is where the core logic from your script goes.
        
        # Cleanup NaNs from delineation results
        for key in waves_dwt:
            if isinstance(waves_dwt[key], list):
                waves_dwt[key] = np.array([x for x in waves_dwt[key] if not np.isnan(x)]).astype(int)
        if 'ECG_R_Peaks' in rpeaks:
             rpeaks['ECG_R_Peaks'] = np.array([x for x in rpeaks['ECG_R_Peaks'] if not np.isnan(x)]).astype(int)

        # --- CORRECTIONS ---
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

        # Initialize variables
        RR_avg, RR_stdev, PR_avg, qrs_avg, QT_avg, QTc_avg, ST_avg_amplitude, ST_avg_deviation, RS_ratio, bpm = (0,)*10
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

        # 3. QRS DURATION (ALGORITMA ORIGINAL)
        if len(waves_dwt['ECG_S_Peaks']) > 1 and len(waves_dwt['ECG_Q_Peaks']) > 1:
            qrs_peak_list = []
            try:
                idex = [x for x in range(0, len(waves_dwt['ECG_S_Peaks']) - 1)]
                for i in idex:
                    # Hitung QRSberbasis Q & S seperti versi asli
                    if waves_dwt['ECG_S_Peaks'][i] < waves_dwt['ECG_Q_Peaks'][i]:
                        qrs_samples = (waves_dwt['ECG_S_Peaks'][i + 1] - waves_dwt['ECG_Q_Peaks'][i])
                    else:
                        qrs_samples = (waves_dwt['ECG_S_Peaks'][i] - waves_dwt['ECG_Q_Peaks'][i])

                    ms_dist = ((qrs_samples / Fs) * 1000.0)

                    # === FALLBACK: jika QRS= 0 ms, pakai R_offset - R_onset beat terkait ===
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
                    dfQRS= pd.DataFrame(qrs_peak_list).fillna(0)

                    # Manual averaging seperti original
                    sum_QRS= 0.0
                    count_QRS= 0.0
                    for index in range(len(qrs_peak_list)):
                        if np.isnan(qrs_peak_list[index]):
                            continue
                        else:
                            sum_QRS+= qrs_peak_list[index]
                            count_QRS+= 1

                    if count_QRS> 0:
                        qrs_avg = (sum_QRS/ count_QRS)
            except:
                self.log_message(f"QRS width Error pada lead {lead_name}")

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
            self.log_message(f"[{lead_name}] Error pada metode QT robust: {e}")
            QT_avg, QTc_avg = 0, 0

        # --- TAHAP 2: Jika metode utama gagal (hasil=0), jalankan Last Resort ---
        if QT_avg == 0 or not np.isfinite(QT_avg):
            try:
                self.log_message(f"[{lead_name}] Metode utama QT gagal, menjalankan Last Resort...")
                
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
                    self.log_message(f"[{lead_name}] Last Resort berhasil! QT ditemukan: {qt_avg_fb:.2f} ms")
                    QT_avg = float(qt_avg_fb)
                    if RR_avg is not None and RR_avg > 0:
                        rr_sec = RR_avg / 1000.0
                        QTc_avg = QT_avg / (rr_sec ** (1/3)) # Hitung ulang QTc Fridericia
                else:
                    self.log_message(f"[{lead_name}] Last Resort juga gagal.")

            except Exception as e_fb:
                self.log_message(f"[{lead_name}] Error pada metode QT Last Resort: {e_fb}")
                
        
        # =================================================================
        # 5. ST SEGMENT (ALGORITMA BARU YANG ADAPTIF)
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
            self.log_message(f"  ✅ ST Amplitude terhitung: {ST_avg_amplitude:.4f} mV (dari {len(st_amplitudes_list)} detak jantung)")
        if st_deviations_list:
            ST_avg_deviation = np.nanmean(st_deviations_list)
            self.log_message(f"  ✅ ST Deviation terhitung: {ST_avg_deviation:.4f} mV (dari {len(st_deviations_list)} detak jantung)")

        # 6. R/S RATIO (ALGORITMA ORIGINAL)
        if 'ECG_S_Peaks' in waves_dwt.keys() and len(waves_dwt['ECG_S_Peaks']) > 0 and len(rpeaks['ECG_R_Peaks']) > 0:
            try:
                R_mean_amp = np.mean([y_filt[int(i)] for i in rpeaks['ECG_R_Peaks']])
                S_mean_amp = np.mean([y_filt[int(i)] for i in waves_dwt['ECG_S_Peaks']])

                if S_mean_amp != 0:  # Avoid division by zero
                    RS_ratio = (R_mean_amp) / abs(S_mean_amp)
            except Exception as e:
                self.log_message(f"Error calculating R/S ratio for {lead_name}: {e}")
        return RR_avg, RR_stdev, PR_avg, qrs_avg, QT_avg, QTc_avg, ST_avg_amplitude, ST_avg_deviation, RS_ratio, bpm

    def _predict_cluster(self, packedData):
        try:
            leads = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
            FEATURE_FAMILIES = ['rr', 'rr_std', 'pr', 'qrs', 'qtc', 'st_amplitude_mv', 'st_deviation_mv', 'rs_ratio', 'heartrate']
            EXPECTED_FEATURES = [f"{fam}_{lead.lower()}" for lead in leads for fam in FEATURE_FAMILIES]
            
            X = pd.DataFrame([{k: packedData.get(k, 0.0) for k in EXPECTED_FEATURES}], columns=EXPECTED_FEATURES).astype(float)
            
            if hasattr(self.scaler, "feature_names_in_"):
                 X = X.reindex(columns=self.scaler.feature_names_in_, fill_value=0.0)
            
            X_scaled = self.scaler.transform(X)
            label = predict_clusters_xgb_integrated(X_scaled, self.model_xgb, self.kmeans_abnormal)[0]
            return label
        except Exception as e:
            self.log_message(f"❌ Gagal melakukan prediksi: {e}")
            return None

    def _save_to_excel(self, packedData, iteration, filepath):
        df = pd.DataFrame([packedData])
        try:
            if iteration == 1 or not os.path.exists(filepath):
                df.to_excel(filepath, index=False)
            else:
                with pd.ExcelWriter(filepath, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
                    reader = pd.read_excel(filepath)
                    df.to_excel(writer, startrow=len(reader) + 1, index=False, header=False)
        except Exception as e:
            self.log_message(f"Gagal menyimpan ke Excel: {e}")

# ==================================================================
# --- SEMUA FUNGSI HELPERS DARI KODE ASLI ---
# ==================================================================
# VREF = 2.42
# GAIN = 3.5
# RESOLUTION = 24
# LSB_SIZE = VREF / (2**RESOLUTION - 1)

# def convert_to_millivolts(adc_value):
#     return adc_value * LSB_SIZE * 1000 / GAIN
def convert_to_millivolts(adc_value: int) -> float:
    """
    Mengonversi nilai ADC mentah 24-bit dari ADS1293 ke milivolt (mV).

    Fungsi ini menggunakan konstanta yang spesifik untuk konfigurasi perangkat keras Anda:
    - VREF: 2.4V (referensi internal tipikal)
    - GAIN: 3.5x (gain tetap dari Instrumentation Amplifier internal)
    - ADC_MAX: 12149040.0 (nilai spesifik dari Tabel 8 datasheet untuk R1=4, R2=5, R3=6, dan SDM 102.4 kHz).
    
    Rumus diadaptasi dari Persamaan 13 di datasheet ADS1293.

    Args:
        adc_code: Nilai mentah integer 32-bit (signed) yang dibaca dari register data ECG.

    Returns:
        Nilai tegangan dalam satuan milivolt (mV) sebagai float.
    """
    # --- Konstanta berdasarkan datasheet dan konfigurasi Anda ---
    VREF = 2.4
    GAIN = 3.5
    ADC_MAX = 12149040.0  # Sesuai dengan R2=5, R3=6 pada SDM 102.4kHz

    # --- Rumus Konversi ---
    # Rumus dasar: Voltage (V) = ((adc_code / ADC_MAX) - 0.5) * (2 * VREF / GAIN)
    voltage_in_volts = ((adc_value / ADC_MAX) - 0.5) * (2.0 * VREF / GAIN)
    
    # Konversi dari Volt ke miliVolt
    return voltage_in_volts * 1000.0

def predict_clusters_xgb_integrated(new_data, xgb_binary_model, kmeans_abnormal, abnormal_label_offset=1):
    is_normal = xgb_binary_model.predict(new_data)
    labels_int = np.full(len(new_data), -999, dtype=int)
    idx_norm = np.where(is_normal == 1)[0]
    labels_int[idx_norm] = 0
    idx_abn = np.where(is_normal == 0)[0]
    if len(idx_abn):
        abn_data = new_data[idx_abn]
        cls_nums = kmeans_abnormal.predict(abn_data) + abnormal_label_offset
        labels_int[idx_abn] = cls_nums
    name_map = {0: "Normal", 2: "Potential Slow Arrhytmia", 1: "Potential Fast Arrhytmia"}
    return np.array([name_map.get(i, f"Cluster {i}") for i in labels_int])

def hampel_despike(x, fs, window_ms=120, n_sigma=4.0):
    x = np.asarray(x, dtype=float); n = len(x)
    w = max(1, int((window_ms/1000.0)*fs)); y = x.copy(); k = 1.4826
    for i in range(n):
        i0, i1 = max(0, i-w), min(n, i+w+1); m = np.median(x[i0:i1])
        s = k * np.median(np.abs(x[i0:i1] - m)) + 1e-12
        if np.abs(x[i] - m) > n_sigma * s: y[i] = m
    return y

def saturasi_interpolasi(x, adc_abs_max=8388607, frac=0.995, guard=5):
    x = np.asarray(x, dtype=float); y = x.copy()
    sat = np.abs(x) >= (frac * adc_abs_max)
    if not np.any(sat): return y
    idx = np.where(sat)[0]; seg, start = [], idx[0]
    for a,b in zip(idx[:-1], idx[1:]):
        if b != a+1: seg.append((start, a)); start = b
    seg.append((start, idx[-1]))
    for s,e in seg:
        i0, i1 = max(0, s-guard), min(len(y)-1, e+guard)
        left, right = i0-1, i1+1
        if left < 0 or right >= len(y): y[i0:i1+1] = np.median(y[max(0,i0-50):min(len(y),i1+50)])
        else: y[i0:i1+1] = np.linspace(y[left], y[right], i1-i0+1)
    return y

def despike_pipeline(x, fs):
    y = saturasi_interpolasi(x, adc_abs_max=8388607, frac=0.995, guard=6)
    return hampel_despike(y, fs, window_ms=100, n_sigma=4.0)

def _qrs_ms_from_Ronset_Roffset(beat_index, anchor_sample, waves_dwt, rpeaks, Fs):
    rpeaks_arr  = np.array([x for x in rpeaks.get('ECG_R_Peaks', []) if not np.isnan(x)], dtype=int)
    r_onsets    = np.array([x for x in waves_dwt.get('ECG_R_Onsets', []) if not np.isnan(x)], dtype=int)
    r_offsets   = np.array([x for x in waves_dwt.get('ECG_R_Offsets', []) if not np.isnan(x)], dtype=int)
    if len(rpeaks_arr) == 0 or len(r_onsets) == 0 or len(r_offsets) == 0: return None
    j = int(np.argmin(np.abs(rpeaks_arr - int(anchor_sample)))) if anchor_sample is not None else min(max(0, beat_index), len(rpeaks_arr) - 1)
    rp = rpeaks_arr[j]
    onset_candidates  = r_onsets[r_onsets <= rp]
    offset_candidates = r_offsets[r_offsets >= rp]
    if len(onset_candidates) == 0 or len(offset_candidates) == 0: return None
    onset  = int(onset_candidates.max()); offset = int(offset_candidates.min())
    if offset <= onset: return None
    return ((offset - onset) / Fs) * 1000.0

def _calculate_qt_robust(r_onsets, t_offsets, t_peaks, fs, rr_ms_avg):
    if r_onsets is None or len(r_onsets) == 0: return 0, 0, []
    t_marks = t_offsets if (t_offsets is not None and len(t_offsets) > 0) else t_peaks
    if t_marks is None or len(t_marks) == 0: return 0, 0, []
    valid_qts_ms = []; t_idx = 0
    min_qt_ms, max_qt_ms = 200.0, 550.0
    if rr_ms_avg and not np.isnan(rr_ms_avg) and rr_ms_avg > 0:
        min_qt_ms = max(min_qt_ms, 0.25 * rr_ms_avg) 
        max_qt_ms = min(max_qt_ms, 0.65 * rr_ms_avg)
    for r_onset in r_onsets:
        while t_idx < len(t_marks) and t_marks[t_idx] <= r_onset: t_idx += 1
        if t_idx < len(t_marks):
            qt_samples = t_marks[t_idx] - r_onset
            qt_ms = (qt_samples / fs) * 1000.0
            if min_qt_ms <= qt_ms <= max_qt_ms: valid_qts_ms.append(qt_ms)
    if not valid_qts_ms: return 0, 0, []
    qt_avg = np.median(valid_qts_ms); qtc_avg = 0
    if rr_ms_avg and not np.isnan(rr_ms_avg) and rr_ms_avg > 0:
        rr_sec = rr_ms_avg / 1000.0
        if rr_sec > 0: qtc_avg = qt_avg / (rr_sec ** (1/3))
    return qt_avg, qtc_avg, valid_qts_ms

def _qt_last_resort_pairing(R_onsets, T_marks, Fs, rr_ms=None, min_abs_ms=160.0, max_abs_ms=600.0):
    R = np.asarray(R_onsets, dtype=float); T = np.asarray(T_marks, dtype=float)
    R = R[~np.isnan(R)]; T = T[~np.isnan(T)]
    if len(R) == 0 or len(T) == 0: return [], np.nan
    lo, hi = (max(min_abs_ms, 0.20 * rr_ms), min(max_abs_ms, 0.60 * rr_ms)) if rr_ms and np.isfinite(rr_ms) else (min_abs_ms, max_abs_ms)
    qt_ms, j = [], 0
    for r in R:
        while j < len(T) and T[j] <= r: j += 1
        if j >= len(T): break
        dt_ms = (T[j] - r) * 1000.0 / Fs
        if lo <= dt_ms <= hi: qt_ms.append(dt_ms)
    return qt_ms, (float(np.nanmedian(qt_ms)) if len(qt_ms) else np.nan)

def find_iso_level(r_peak_index, waves_dwt, signal, fs):
    p_onsets = waves_dwt.get('ECG_P_Onsets', []); r_onsets = waves_dwt.get('ECG_R_Onsets', []); t_offsets = waves_dwt.get('ECG_T_Offsets', [])
    pr_start_candidates = p_onsets[p_onsets < r_peak_index]; pr_end_candidates = r_onsets[r_onsets < r_peak_index]
    if len(pr_start_candidates) > 0 and len(pr_end_candidates) > 0:
        pr_start, pr_end = int(pr_start_candidates[-1]), int(pr_end_candidates[-1])
        if pr_end > pr_start and (pr_end - pr_start) > fs * 0.02: return np.median(signal[pr_start:pr_end])
    tp_start_candidates = t_offsets[t_offsets < r_peak_index]
    if len(tp_start_candidates) > 0 and len(pr_start_candidates) > 0:
        tp_start, tp_end = int(tp_start_candidates[-1]), int(pr_start_candidates[-1])
        if tp_end > tp_start and (tp_end - tp_start) > fs * 0.02: return np.median(signal[tp_start:tp_end])
    return None

def apply_iec_diagnostic_filter(signal, fs):
    pad_len = int(fs * 2); padded_signal = np.pad(signal, (pad_len, pad_len), mode='edge')
    nyquist = fs / 2.0
    hp_cutoff = min(0.05 / nyquist, 0.99); b_hp, a_hp = scipy.signal.butter(4, hp_cutoff, btype='highpass')
    signal_hp = scipy.signal.filtfilt(b_hp, a_hp, padded_signal)
    lp_cutoff = min(150 / nyquist, 0.99); b_lp, a_lp = scipy.signal.butter(4, lp_cutoff, btype='lowpass')
    signal_lp = scipy.signal.filtfilt(b_lp, a_lp, signal_hp)
    f0, Q = 50.0, 30.0
    if f0 < nyquist:
        w0 = f0 / nyquist; b_notch, a_notch = scipy.signal.iirnotch(w0, Q)
        signal_filtered = scipy.signal.filtfilt(b_notch, a_notch, signal_lp)
    else: signal_filtered = signal_lp
    return signal_filtered[pad_len:-pad_len]

def wavelet_denoise(x, fs, wavelet='sym6', level=None, mode='symmetric', method='soft', protect_low_hz=2.0, protect_band=(1.0, 40.0), protect_factor=0.5):
    if pywt is None: return np.asarray(x, dtype=float)
    x = np.asarray(x, dtype=float); w = pywt.Wavelet(wavelet); maxlev = pywt.dwt_max_level(len(x), w.dec_len)
    if level is None: level = min(5, maxlev)
    coeffs = pywt.wavedec(x, w, mode=mode, level=level); cA = coeffs[0]; details = coeffs[1:]
    new_coeffs = [cA]; lo_prot, hi_prot = protect_band
    for i, cD in enumerate(details, start=1):
        f_low, f_high = fs / (2.0 ** (i + 1)), fs / (2.0 ** i)
        sigma = (np.median(np.abs(cD - np.median(cD))) / 0.6745) + 1e-12
        thr_univ = sigma * np.sqrt(2.0 * np.log(max(cD.size, 2)))
        if f_high <= protect_low_hz: cD_th = cD
        else:
            overlaps = (f_low <= hi_prot) and (f_high >= lo_prot)
            thr = thr_univ * (protect_factor if overlaps else 1.0)
            cD_th = pywt.threshold(cD, thr, mode=method)
        new_coeffs.append(cD_th)
    y = pywt.waverec(new_coeffs, w, mode=mode)
    return y[:len(x)]
