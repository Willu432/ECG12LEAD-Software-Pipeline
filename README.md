# ECG12LEAD-Software-Pipeline
Proyek Pengembangan Pipeline dan Software 12 Lead

╔══════════════════════════════════════════════════════════════╗
  
  APLIKASI ANALISIS SINYAL EKG 12 LEAD                        
  Versi 1.0 - 2025                                            
  Departemen Teknik Elektro - Universitas Padjadjaran  
  
╚══════════════════════════════════════════════════════════════╝

INSTALASI PERTAMA KALI:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. INSTALL VISUAL C++ REDISTRIBUTABLE (WAJIB!)
   ✓ Jalankan file "vcredist_x64.exe" yang disertakan
   ✓ Ikuti petunjuk instalasi
   ✓ Restart komputer jika diminta

2. EXTRACT SEMUA FILE
   ✓ Extract folder ZIP ke lokasi pilihan Anda
   ✓ Contoh: C:\Program Files\AplikasiEKG\

3. JANGAN PISAHKAN FILE!
   ⚠️ PENTING: Jangan pindah atau hapus folder "Model Final 2"
   ⚠️ File .exe HARUS dalam folder yang sama dengan "Model Final 2"

4. JALANKAN APLIKASI
   ✓ Double-click "AplikasiEKG.exe"
   ✓ Jika muncul Windows Defender, klik "More info" > "Run anyway"


CARA MENGGUNAKAN:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 PENGUKURAN REAL-TIME:
   1. Hubungkan ESP32 ke komputer via WiFi
   2. Klik "Lakukan Pengukuran Real-Time"
   3. Pilih folder penyimpanan hasil
   4. Masukkan nama subjek dan durasi rekam
   5. Klik "Mulai Perekaman"

📁 EKSTRAK DATA CSV:
   1. Klik "Ekstrak & Analisis Data CSV"
   2. Pilih file CSV yang ingin dianalisis
   3. Pilih folder output untuk hasil Excel
   4. Klik "Mulai Proses Ekstraksi"
   5. Lihat grafik dan hasil di aplikasi


TROUBLESHOOTING:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

❌ Error "MSVCP140.dll not found"
   → Install vcredist_x64.exe yang disertakan

❌ Aplikasi tidak mau jalan
   → Klik kanan AplikasiEKG.exe > Run as Administrator
   → Pastikan antivirus tidak memblokir

❌ Error "Model not found"
   → Pastikan folder "Model Final 2" ada di lokasi yang sama
   → Jangan ubah nama folder atau isi file model

❌ ESP32 tidak terdeteksi
   → Cek koneksi WiFi ESP32
   → Pastikan IP address benar (default: 192.168.4.1)

❌ Grafik tidak muncul
   → Pastikan file CSV format sudah benar
   → Cek log aktivitas untuk detail error


SYSTEM REQUIREMENTS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Minimum:
- Windows 10/11 (64-bit)
- Processor: Intel Core i3 atau setara
- RAM: 4GB
- Storage: 2GB free space
- Display: 1366x768

Recommended:
- Windows 11 (64-bit)
- Processor: Intel Core i5 atau lebih tinggi
- RAM: 8GB atau lebih
- Storage: 5GB free space
- Display: 1920x1080 atau lebih tinggi


FITUR APLIKASI:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✓ Perekaman real-time dari ESP32
✓ Analisis 12 lead EKG
✓ Deteksi otomatis P, Q, R, S, T waves
✓ Perhitungan parameter EKG (HR, PR interval, dll)
✓ Export hasil ke Excel
✓ Visualisasi grafik interaktif
✓ Mode fullscreen (F11)


KONTAK & SUPPORT:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Pengembang Hardware: Jeffry Fane
Pengembang Software: Willy Juliansyah

Supervisor:
- Arjon Turnip, Ph.D.
- Fikri Rida Fadillah, S.T.

© 2025 Lab Cogno-Technology & AI
Universitas Padjadjaran


LISENSI:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Aplikasi ini dikembangkan untuk keperluan akademis dan penelitian.
Dilarang mendistribusikan ulang tanpa izin.

Versi: 1.0
Build Date: [19-10-2025]
