# ECG12LEAD-Software-Pipeline
Proyek Pengembangan Pipeline dan Software 12 Lead
# ⚡️ Aplikasi Analisis Sinyal EKG 12 Lead ⚡️

### Versi 1.0 - 2025
#### Departemen Teknik Elektro - Universitas Padjadjaran

---

## 📄 Deskripsi Proyek

Aplikasi **Standalone** berbasis desktop (Windows 64-bit) untuk akuisisi, visualisasi, dan analisis otomatis sinyal **Elektrokardiogram (EKG) 12 Lead**. Aplikasi ini dirancang untuk memudahkan penelitian dan studi akademis dalam bidang analisis EKG.

---

## 🚀 Fitur Utama

* **Perekaman Real-Time:** Akuisisi data EKG langsung dari perangkat **ESP32** via koneksi WiFi.
* **Analisis 12 Lead:** Pemrosesan dan visualisasi seluruh 12 lead EKG.
* **Deteksi Otomatis:** Deteksi otomatis gelombang P, QRS, dan T.
* **Perhitungan Parameter:** Perhitungan otomatis parameter klinis EKG (HR, PR Interval, QRS Duration, dll.).
* **Ekspor Data:** Ekspor hasil analisis dan parameter ke _file_ **Excel** (`.xlsx`).
* **Visualisasi Interaktif:** Grafik EKG interaktif dengan mode _fullscreen_ (F11).

---

## 👨‍💻 PANDUAN DEVELOPER (BUILD & DEPLOY)

Bagian ini ditujukan untuk pengembang yang ingin memodifikasi kode sumber atau melakukan _re-build_ aplikasi. Konfigurasi build menggunakan `build_exe_release.spec` yang telah disesuaikan untuk menangani _hidden imports_ yang kompleks.

### 1. Prasyarat (Prerequisites)
Pastikan lingkungan pengembangan Anda memiliki:
* **OS:** Windows 10/11 (64-bit)
* **Python:** Versi 3.9 atau lebih baru
* **Git:** Untuk manajemen versi

### 2. Setup Environment
Buka terminal/CMD di direktori proyek, lalu jalankan perintah berikut:

#### 1. Buat Virtual Environment
python -m venv venv

#### 2. Aktifkan Virtual Environment
venv\Scripts\activate

#### 3. Upgrade pip
python -m pip install --upgrade pip

#### 4. Install Library Dependencies (Sesuai dengan kode pada gui_main.py, ekg_processor.py, dan ekg_ekstrak_data.py)
* pip install numpy pandas scipy matplotlib scikit-learn xgboost neurokit2 PyWavelets joblib Pillow pyserial openpyxl tkinter threading socket struct
* kalau kurang atau lebih silahkan disesuaikan

#### 5. Install PyInstaller
pip install pyinstaller

#### 6. Deploy
* Download semua file disini
* Pada terminal, arahkan ke direktori tempat file berasal, CD "D:\EKG\Skripsi Willy\Software EKG 12 Lead"
* pyinstaller --clean build_exe_release.spec

#### 7. Hasil Build
file aplikasi akan muncul di folder dist/AplikasiEKG.exe

---

## 📥 Panduan Instalasi Pertama Kali User (WAJIB!)

Aplikasi ini bersifat **STANDALONE** (tidak perlu _install_ Python, Library, atau _Environment_).

### Langkah 1: Instalasi Visual C++ Redistributable
Ini wajib untuk menjalankan aplikasi.

1.  Jalankan _file_ **`vcredist_x64.exe`** yang disertakan.
2.  Ikuti instruksi instalasi.
3.  **_Restart_ komputer** jika diminta setelah instalasi selesai.

### Langkah 2: Jalankan Aplikasi

1.  _Double-click_ **`AplikasiEKG.exe`**.
2.  Jika muncul peringatan **Windows Defender SmartScreen**, klik **"More info"** lalu **"Run anyway"**.

---

## ⚙️ Cara Menggunakan

### 📊 Mode Pengukuran Real-Time

Digunakan untuk merekam sinyal secara langsung dari perangkat keras ESP32.

1.  Pastikan **ESP32** sudah terhubung ke komputer via **WiFi** (sesuai panduan perangkat keras).
2.  Di aplikasi, klik **"Lakukan Pengukuran Real-Time"**.
3.  Pilih _folder_ untuk penyimpanan hasil rekaman.
4.  Masukkan nama subjek dan durasi rekam yang diinginkan.
5.  Klik **"Mulai Perekaman"**.

### 📁 Mode Ekstrak & Analisis Data CSV

Digunakan untuk menganalisis _file_ data EKG yang sudah terekam (_offline_).

1.  Di aplikasi, klik **"Ekstrak & Analisis Data CSV"**.
2.  Pilih _file_ **CSV** yang ingin Anda proses.
3.  Pilih _folder_ _output_ untuk hasil analisis **Excel**.
4.  Klik **"Mulai Proses Ekstraksi"**.
5.  Visualisasi grafik dan hasil analisis akan ditampilkan di antarmuka aplikasi.

---

## 🛠️ Troubleshooting

| Masalah | Solusi |
| :--- | :--- |
| **Error** "_MSVCP140.dll not found_" | **Install `vcredist_x64.exe`** yang disertakan. |
| Aplikasi tidak bisa berjalan | Coba _Run as Administrator_ (`Klik kanan` > `Run as Administrator`). Pastikan antivirus tidak memblokir. |
| **Error** "_Model not found_" | Pastikan _folder_ **`Model Final 2`** berada di direktori yang sama dengan `AplikasiEKG.exe`. |
| ESP32 tidak terdeteksi | Cek koneksi WiFi. Pastikan **IP address** sudah benar (default: `192.168.4.1`). |
| Grafik tidak muncul | Pastikan format _file_ CSV sudah benar. Cek _log aktivitas_ di aplikasi untuk detail error. |

---

## 💻 System Requirements

| | Minimum | Recommended |
| :--- | :--- | :--- |
| **OS** | Windows 10/11 (64-bit) | Windows 11 (64-bit) |
| **Processor** | Intel Core i3 atau setara | Intel Core i5 atau lebih tinggi |
| **RAM** | 4GB | 8GB atau lebih |
| **Storage** | 2GB _free space_ | 5GB _free space_ |
| **Display** | 1366x768 | 1920x1080 (Full HD) |

---

## 👥 Tim Pengembang & Support

### Pengembang
* **Software:** Willy Juliansyah
* **Hardware:** Jeffry Fane

### Supervisor
* Arjon Turnip, Ph.D.
* Fikri Rida Fadillah, S.T.

### Kontak
© 2025 **Lab Cogno-Technology & AI**
Universitas Padjadjaran

---

## 📜 Lisensi

Aplikasi ini dikembangkan untuk keperluan **Akademis dan Penelitian** di lingkungan Universitas Padjadjaran. Dilarang mendistribusikan ulang atau menggunakan untuk kepentingan komersial tanpa izin resmi dari pengembang.

**Versi:** 1.0
**Build Date:** [19-10-2025]
