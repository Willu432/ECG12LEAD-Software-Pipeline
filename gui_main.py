# File: gui_main.py

import matplotlib
matplotlib.use('TkAgg')  # Backend untuk real-time plotting
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import os
import sys
import threading
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np # baru ditambahkan untuk plot

# Impor kelas-kelas prosesor dari file-file terpisah
from ekg_processor import EKGProcessor
from ekg_ekstrak_data import EKGExtractor

# --- Definisi Tema dan Font ---
FONT_PRIMARY = ("Segoe UI", 10)
FONT_BOLD = ("Segoe UI", 10, "bold")
FONT_TITLE = ("Segoe UI", 32, "bold")
FONT_SUBTITLE = ("Segoe UI", 18, "bold")
FONT_BUTTON = ("Segoe UI", 11, "bold")
FONT_LOG = ("Consolas", 9)

# Skema warna modern dengan kontras yang baik
COLOR_BG = "#F5F7FA"
COLOR_FRAME = "#FFFFFF"
COLOR_PRIMARY = "#2563EB"  # Biru modern
COLOR_PRIMARY_DARK = "#1E40AF"
COLOR_ACCENT = "#10B981"  # Hijau emerald
COLOR_ACCENT_DARK = "#059669"
COLOR_DANGER = "#EF4444"
COLOR_DANGER_DARK = "#DC2626"
COLOR_TEXT = "#1F2937"
COLOR_TEXT_SECONDARY = "#6B7280"
COLOR_LIGHT_TEXT = "#FFFFFF"
COLOR_BORDER = "#E5E7EB"
COLOR_HOVER = "#F3F4F6"

class App(tk.Tk):
    """Kelas utama aplikasi yang bertindak sebagai controller untuk semua halaman."""
    def __init__(self):
        super().__init__()
        self.title("12 Lead ECG Signal Analyzer - Padjadjaran University")
        self.geometry("1400x800")
        self.minsize(1200, 700)
        self.configure(bg=COLOR_BG)
        
        # Fullscreen toggle
        self.is_fullscreen = False
        self.bind("<F11>", self.toggle_fullscreen)
        self.bind("<Escape>", self.exit_fullscreen)

        # Konfigurasi style dengan tema modern
        style = ttk.Style(self)
        style.theme_use('clam')
        
        # Style dasar
        style.configure(".", background=COLOR_BG, foreground=COLOR_TEXT, font=FONT_PRIMARY)
        style.configure("TFrame", background=COLOR_BG)
        style.configure("TLabel", background=COLOR_BG, foreground=COLOR_TEXT)
        
        # Card-style frame dengan shadow effect
        style.configure("Card.TFrame", background=COLOR_FRAME, relief="flat", borderwidth=0)
        style.configure("Card.TLabelframe", background=COLOR_FRAME, bordercolor=COLOR_BORDER, 
                       relief="solid", borderwidth=1)
        style.configure("Card.TLabelframe.Label", background=COLOR_FRAME, 
                       foreground=COLOR_PRIMARY, font=FONT_BOLD, padding=(5, 0))
        
        # Button styles dengan state yang jelas
        style.configure("Primary.TButton", background=COLOR_PRIMARY, foreground=COLOR_LIGHT_TEXT, 
                       font=FONT_BUTTON, padding=(20, 12), borderwidth=0, relief="flat")
        style.map("Primary.TButton", 
                 background=[('active', COLOR_PRIMARY_DARK), ('disabled', COLOR_BORDER)],
                 foreground=[('disabled', COLOR_TEXT_SECONDARY)])
        
        style.configure("Accent.TButton", background=COLOR_ACCENT, foreground=COLOR_LIGHT_TEXT, 
                       font=FONT_BUTTON, padding=(20, 12), borderwidth=0, relief="flat")
        style.map("Accent.TButton", 
                 background=[('active', COLOR_ACCENT_DARK), ('disabled', COLOR_BORDER)],
                 foreground=[('disabled', COLOR_TEXT_SECONDARY)])
        
        style.configure("Danger.TButton", background=COLOR_DANGER, foreground=COLOR_LIGHT_TEXT, 
                       font=FONT_BUTTON, padding=(20, 12), borderwidth=0, relief="flat")
        style.map("Danger.TButton", 
                 background=[('active', COLOR_DANGER_DARK), ('disabled', COLOR_BORDER)],
                 foreground=[('disabled', COLOR_TEXT_SECONDARY)])
        
        # Small Danger Button untuk exit di header
        style.configure("DangerSmall.TButton", background=COLOR_DANGER, foreground=COLOR_LIGHT_TEXT, 
                       font=FONT_PRIMARY, padding=(12, 8), borderwidth=0, relief="flat")
        style.map("DangerSmall.TButton", 
                 background=[('active', COLOR_DANGER_DARK)],
                 foreground=[('active', COLOR_LIGHT_TEXT)])
        
        style.configure("Secondary.TButton", background=COLOR_FRAME, foreground=COLOR_TEXT, 
                       font=FONT_PRIMARY, padding=(15, 10), borderwidth=1, relief="solid")
        style.map("Secondary.TButton", 
                 background=[('active', COLOR_HOVER), ('disabled', COLOR_BG)],
                 bordercolor=[('active', COLOR_PRIMARY)])
        
        style.configure("Link.TButton", padding=(5, 5), relief="flat", background=COLOR_BG, 
                       foreground=COLOR_PRIMARY, font=("Segoe UI", 10, "underline"))
        style.map("Link.TButton", 
                 background=[('active', COLOR_BG)],
                 foreground=[('active', COLOR_PRIMARY_DARK)])
        
        style.configure("MainMenu.TButton", font=("Segoe UI", 13, "bold"), padding=(25, 18))
        
        # Entry style
        style.configure("TEntry", fieldbackground=COLOR_FRAME, bordercolor=COLOR_BORDER, 
                       lightcolor=COLOR_BORDER, darkcolor=COLOR_BORDER, padding=8)
        style.map("TEntry", fieldbackground=[('readonly', COLOR_HOVER)],
                 bordercolor=[('focus', COLOR_PRIMARY)])
        
        # Radiobutton style
        style.configure("TRadiobutton", background=COLOR_FRAME, foreground=COLOR_TEXT, 
                       font=FONT_PRIMARY, padding=5)
        style.map("TRadiobutton", 
                 background=[('active', COLOR_HOVER), ('disabled', COLOR_FRAME)],
                 foreground=[('disabled', COLOR_TEXT_SECONDARY)])
        
        container = ttk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}
        for F in (MainMenuFrame, MeasurementFrame, ExtractionFrame, InfoFrame):
            page_name = F.__name__
            frame = F(parent=container, controller=self)
            self.frames[page_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("MainMenuFrame")
        self.protocol("WM_DELETE_WINDOW", self.quit_app)
        
        # Bind event untuk maximize button
        self.bind("<Map>", self.on_window_state_change)

    def show_frame(self, page_name):
        frame = self.frames[page_name]
        frame.tkraise()
    
    def toggle_fullscreen(self, event=None):
        self.is_fullscreen = not self.is_fullscreen
        self.attributes("-fullscreen", self.is_fullscreen)
        return "break"
    
    def exit_fullscreen(self, event=None):
        if self.is_fullscreen:
            self.is_fullscreen = False
            self.attributes("-fullscreen", False)
        return "break"
    
    def on_window_state_change(self, event=None):
        """Otomatis masuk fullscreen saat window di-maximize"""
        # Check jika window dalam state zoomed (maximized)
        if self.state() == 'zoomed' and not self.is_fullscreen:
            # Delay sedikit untuk memastikan window sudah fully maximized
            self.after(100, self.enter_fullscreen_from_maximize)
    
    def enter_fullscreen_from_maximize(self):
        """Helper function untuk masuk fullscreen dari maximize"""
        if not self.is_fullscreen:
            self.is_fullscreen = True
            self.attributes("-fullscreen", True)
        
    def quit_app(self):
        measurement_frame = self.frames.get("MeasurementFrame")
        if measurement_frame and measurement_frame.processing_thread and measurement_frame.processing_thread.is_alive():
            if messagebox.askyesno("Exit App", 
                                  "The recording process is in progress. Are you sure you want to exit?",
                                  icon='warning'):
                measurement_frame.stop_recording()
                self.destroy()
        else:
            if messagebox.askyesno("Exit Application", 
                                  "Are you sure you want to exit the app?"):
                self.destroy()

# ===================================================================
# --- Halaman Menu Utama ---
# ===================================================================
class MainMenuFrame(ttk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.configure(style="TFrame")

        # Canvas untuk scrolling
        self.canvas = tk.Canvas(self, bg=COLOR_BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        
        # Container utama
        main_container = ttk.Frame(self.canvas)
        
        main_container.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        canvas_window = self.canvas.create_window((0, 0), window=main_container, anchor="n")
        self.canvas.configure(yscrollcommand=scrollbar.set)
        
        # Bind untuk update width saat window resize
        def _on_canvas_configure(event):
            self.canvas.itemconfig(canvas_window, width=event.width)
        self.canvas.bind("<Configure>", _on_canvas_configure)
        
        # Pack canvas dan scrollbar
        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Bind mousewheel untuk scrolling - PERBAIKAN
        self.canvas.bind("<Enter>", self._bind_mousewheel)
        self.canvas.bind("<Leave>", self._unbind_mousewheel)

        # Main wrapper untuk centering
        wrapper = ttk.Frame(main_container)
        wrapper.pack(expand=True, fill="both", pady=40)
        
        # Content frame yang akan di-center
        content_frame = ttk.Frame(wrapper)
        content_frame.pack(expand=True, anchor="center")

        # Header dengan logo
        header_frame = ttk.Frame(content_frame)
        header_frame.pack(pady=(20, 30))

        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            logo_path = os.path.join(script_dir, "logo_unpad.png")
            img = Image.open(logo_path).resize((140, 140), Image.Resampling.LANCZOS)
            self.logo = ImageTk.PhotoImage(img)
            logo_label = ttk.Label(header_frame, image=self.logo, background=COLOR_BG)
            logo_label.pack(pady=(0, 15))
        except Exception as e:
            print(f"Error Logo : {e}")
        
        # Judul aplikasi - centered
        title_label = ttk.Label(header_frame, text="12 Lead ECG Signal Analyzer", 
                               font=FONT_TITLE, background=COLOR_BG, foreground=COLOR_PRIMARY,
                               justify=tk.CENTER)
        title_label.pack()
        
        subtitle_label = ttk.Label(header_frame, 
                                   text="Departement of Electrical Engineering • Padjadjaran University", 
                                   font=("Segoe UI", 11), background=COLOR_BG, 
                                   foreground=COLOR_TEXT_SECONDARY,
                                   justify=tk.CENTER)
        subtitle_label.pack(pady=(5, 0))

        # Menu buttons dengan ikon dan deskripsi - centered
        buttons_frame = ttk.Frame(content_frame)
        buttons_frame.pack(pady=20)
        
        btn_width = 45
        
        # Tombol Pengukuran Real-Time
        btn1 = ttk.Button(buttons_frame, 
                         text="📊  Start Real-Time Measurement", 
                         command=lambda: controller.show_frame("MeasurementFrame"), 
                         style="MainMenu.TButton", width=btn_width)
        btn1.pack(pady=8, ipady=5)
        desc1 = ttk.Label(buttons_frame, 
                         text="Record and analyze ECG signals directly from the ESP32 device", 
                         font=("Segoe UI", 9), foreground=COLOR_TEXT_SECONDARY, background=COLOR_BG,
                         justify=tk.CENTER)
        desc1.pack(pady=(0, 15))
        
        # Tombol Ekstraksi Data
        btn2 = ttk.Button(buttons_frame, 
                         text="📁  Extract & Analyze CSV Data", 
                         command=lambda: controller.show_frame("ExtractionFrame"), 
                         style="MainMenu.TButton", width=btn_width)
        btn2.pack(pady=8, ipady=5)
        desc2 = ttk.Label(buttons_frame, 
                         text="Process and analyze ECG data already stored in CSV format", 
                         font=("Segoe UI", 9), foreground=COLOR_TEXT_SECONDARY, background=COLOR_BG,
                         justify=tk.CENTER)
        desc2.pack(pady=(0, 15))
        
        # Tombol Info
        btn3 = ttk.Button(buttons_frame, 
                         text="ℹ️  About the Developer", 
                         command=lambda: controller.show_frame("InfoFrame"), 
                         style="MainMenu.TButton", width=btn_width)
        btn3.pack(pady=8, ipady=5)
        desc3 = ttk.Label(buttons_frame, 
                         text="Information about the development team and project supervisors", 
                         font=("Segoe UI", 9), foreground=COLOR_TEXT_SECONDARY, background=COLOR_BG,
                         justify=tk.CENTER)
        desc3.pack(pady=(0, 25))
        
        # Separator line
        separator_frame = ttk.Frame(buttons_frame)
        separator_frame.pack(fill=tk.X, pady=(10, 20))
        ttk.Separator(separator_frame, orient="horizontal").pack(fill=tk.X, padx=50)
        
        # Tombol Fullscreen
        fullscreen_frame = ttk.Frame(buttons_frame)
        fullscreen_frame.pack(pady=(0, 10))
        
        btn_fullscreen = ttk.Button(fullscreen_frame, 
                         text="🖥️  Fullscreen Mode (F11)", 
                         command=controller.toggle_fullscreen, 
                         style="Secondary.TButton", width=30)
        btn_fullscreen.pack()
        
        fullscreen_hint = ttk.Label(fullscreen_frame, 
                         text="Press F11 or Esc to exit full screen", 
                         font=("Segoe UI", 8), foreground=COLOR_TEXT_SECONDARY, background=COLOR_BG,
                         justify=tk.CENTER)
        fullscreen_hint.pack(pady=(3, 15))
        
        # Tombol Keluar
        btn4 = ttk.Button(buttons_frame, 
                         text="🚪  Exit App", 
                         command=controller.quit_app, 
                         style="Danger.TButton", width=btn_width)
        btn4.pack(pady=8, ipady=5)

        # Footer - centered
        footer = ttk.Label(content_frame, text="© 2025 Lab Cogno-Technology & AI", 
                          font=("Segoe UI", 9), foreground=COLOR_TEXT_SECONDARY, 
                          background=COLOR_BG,
                          justify=tk.CENTER)
        footer.pack(pady=20)
    
    def _bind_mousewheel(self, event):
        """Bind mousewheel saat mouse masuk ke canvas"""
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        # Untuk Linux
        self.canvas.bind_all("<Button-4>", self._on_mousewheel)
        self.canvas.bind_all("<Button-5>", self._on_mousewheel)
    
    def _unbind_mousewheel(self, event):
        """Unbind mousewheel saat mouse keluar dari canvas"""
        self.canvas.unbind_all("<MouseWheel>")
        # Untuk Linux
        self.canvas.unbind_all("<Button-4>")
        self.canvas.unbind_all("<Button-5>")
    
    def _on_mousewheel(self, event):
        """Handle mousewheel scrolling"""
        if event.num == 5 or event.delta < 0:
            self.canvas.yview_scroll(1, "units")
        elif event.num == 4 or event.delta > 0:
            self.canvas.yview_scroll(-1, "units")

# ===================================================================
# --- Halaman Pengukuran Real-Time ---
# ===================================================================
class MeasurementFrame(ttk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        
        # Gunakan path yang kompatibel dengan PyInstaller
        if getattr(sys, 'frozen', False):
            # Running sebagai executable
            script_dir = sys._MEIPASS
        else:
            # Running sebagai script
            script_dir = os.path.dirname(os.path.abspath(__file__))
        
        self.fixed_model_path = os.path.join(script_dir, "Model Final 2")

        self.processor = EKGProcessor(status_callback=self.update_status, 
                                     log_callback=self.log_message)
        self.processing_thread = None

        self.path_utama_var = tk.StringVar()
        self.nama_subjek_var = tk.StringVar(value="Sample_Subject_Name")
        self.ip_var = tk.StringVar(value="192.168.4.1")
        self.port_var = tk.StringVar(value="8888")
        self.duration_var = tk.StringVar(value="10")
        self.status_var = tk.StringVar(value="Ready to start measuring")
        # ==============================================================================================
        # TAMBAHAN: Variabel untuk live plotting
        self.plot_active = False
        self.plot_data = {
            "I": deque(maxlen=2000), "II": deque(maxlen=2000),
            "III": deque(maxlen=2000), "AVR": deque(maxlen=2000),
            "AVL": deque(maxlen=2000), "AVF": deque(maxlen=2000),
            "V1": deque(maxlen=2000), "V2": deque(maxlen=2000),
            "V3": deque(maxlen=2000), "V4": deque(maxlen=2000),
            "V5": deque(maxlen=2000), "V6": deque(maxlen=2000)
        }
        self.selected_channels = {
            "I": tk.BooleanVar(value=True),
            "II": tk.BooleanVar(value=True),
            "III": tk.BooleanVar(value=False),
            "AVR": tk.BooleanVar(value=False),
            "AVL": tk.BooleanVar(value=False),
            "AVF": tk.BooleanVar(value=False),
            "V1": tk.BooleanVar(value=False),
            "V2": tk.BooleanVar(value=False),
            "V3": tk.BooleanVar(value=False),
            "V4": tk.BooleanVar(value=False),
            "V5": tk.BooleanVar(value=False),
            "V6": tk.BooleanVar(value=False)
        }
        self.fig = None
        self.axes = None
        self.lines = {}
        self.anim = None
        # ================================================================================
        self._create_widgets()

    def _create_widgets(self):
        # Header dengan tombol kembali
        header_frame = ttk.Frame(self, style="Card.TFrame", height=70)
        header_frame.pack(fill=tk.X, padx=15, pady=15)
        header_frame.pack_propagate(False)
        
        back_button = ttk.Button(header_frame, text="← Return to Main Menu", 
                                command=lambda: self.controller.show_frame("MainMenuFrame"), 
                                style="Link.TButton")
        back_button.pack(side=tk.LEFT, padx=15, pady=15)
        
        title_label = ttk.Label(header_frame, text="Real-Time Measurement", 
                               font=FONT_SUBTITLE, background=COLOR_FRAME, 
                               foreground=COLOR_PRIMARY)
        title_label.pack(side=tk.LEFT, padx=10, pady=15)

        # Canvas untuk scrolling content
        self.canvas = tk.Canvas(self, bg=COLOR_BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        
        # Content area dengan scroll
        content_frame = ttk.Frame(self.canvas)
        
        content_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        canvas_window = self.canvas.create_window((0, 0), window=content_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)
        
        # Bind untuk auto-resize lebar canvas
        def _configure_canvas(event):
            self.canvas.itemconfig(canvas_window, width=event.width)
        self.canvas.bind("<Configure>", _configure_canvas)
        
        # Pack canvas
        self.canvas.pack(side="left", fill="both", expand=True, padx=15, pady=(0, 15))
        scrollbar.pack(side="right", fill="y", pady=(0, 15))
        
        # Bind mousewheel
        self.canvas.bind("<Enter>", self._bind_mousewheel)
        self.canvas.bind("<Leave>", self._unbind_mousewheel)
        
        # Card untuk pengaturan path
        path_card = ttk.LabelFrame(content_frame, text="📂 Save Location", 
                                   padding=20, style="Card.TLabelframe")
        path_card.pack(fill=tk.X, pady=(0, 10))
        
        path_inner = ttk.Frame(path_card, style="Card.TFrame")
        path_inner.pack(fill=tk.X)
        
        ttk.Label(path_inner, text="Output Folder :", 
                 background=COLOR_FRAME).grid(row=0, column=0, sticky=tk.W, pady=(0, 8))
        
        path_entry_frame = ttk.Frame(path_inner, style="Card.TFrame")
        path_entry_frame.grid(row=1, column=0, sticky=tk.EW, pady=0)
        path_inner.columnconfigure(0, weight=1)
        
        self.path_entry = ttk.Entry(path_entry_frame, textvariable=self.path_utama_var, 
                                    state="readonly", font=FONT_PRIMARY)
        self.path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        ttk.Button(path_entry_frame, text="Pilih Folder", 
                  command=self._select_main_path, 
                  style="Secondary.TButton").pack(side=tk.LEFT)

        # Card untuk parameter
        param_card = ttk.LabelFrame(content_frame, text="⚙️ Measurement Parameters", 
                                    padding=20, style="Card.TLabelframe")
        param_card.pack(fill=tk.X, pady=(0, 10))
        
        param_inner = ttk.Frame(param_card, style="Card.TFrame")
        param_inner.pack(fill=tk.X)
        param_inner.columnconfigure(1, weight=1)
        param_inner.columnconfigure(3, weight=1)
        
        # Nama Subjek
        ttk.Label(param_inner, text="Subject Name:", 
                 background=COLOR_FRAME).grid(row=0, column=0, sticky=tk.W, pady=8, padx=(0, 15))
        self.subjek_entry = ttk.Entry(param_inner, textvariable=self.nama_subjek_var)
        self.subjek_entry.grid(row=0, column=1, columnspan=3, sticky=tk.EW, pady=8)
        
        # IP dan Port
        ttk.Label(param_inner, text="ESP32 IP:", 
                 background=COLOR_FRAME).grid(row=1, column=0, sticky=tk.W, pady=8, padx=(0, 15))
        self.ip_entry = ttk.Entry(param_inner, textvariable=self.ip_var)
        self.ip_entry.grid(row=1, column=1, sticky=tk.EW, pady=8, padx=(0, 20))
        
        ttk.Label(param_inner, text="Port:", 
                 background=COLOR_FRAME).grid(row=1, column=2, sticky=tk.W, padx=(0, 15))
        self.port_entry = ttk.Entry(param_inner, textvariable=self.port_var, width=15)
        self.port_entry.grid(row=1, column=3, sticky=tk.W, pady=8)
        
        # Durasi
        ttk.Label(param_inner, text="Recording Duration:", 
                 background=COLOR_FRAME).grid(row=2, column=0, sticky=tk.W, pady=8, padx=(0, 15))
        duration_frame = ttk.Frame(param_inner, style="Card.TFrame")
        duration_frame.grid(row=2, column=1, sticky=tk.W, pady=8)
        self.duration_entry = ttk.Entry(duration_frame, textvariable=self.duration_var, width=10)
        self.duration_entry.pack(side=tk.LEFT, padx=(0, 5))
        ttk.Label(duration_frame, text="second", background=COLOR_FRAME).pack(side=tk.LEFT)

        # Tombol kontrol
        control_frame = ttk.Frame(content_frame, style="Card.TFrame")
        control_frame.pack(fill=tk.X, pady=10)
        
        self.start_button = ttk.Button(control_frame, text="▶ Start Recording", 
                                       command=self.start_recording, 
                                       style="Accent.TButton")
        self.start_button.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 10))
        
        self.stop_button = ttk.Button(control_frame, text="⏹ Stop Recording", 
                                      command=self.stop_recording, 
                                      style="Danger.TButton", state="disabled")
        self.stop_button.pack(side=tk.LEFT, expand=True, fill=tk.X)
        # ===============================================================================================
        # TAMBAHAN: Card untuk Live Plot Control (setelah control_frame)
        plot_control_card = ttk.LabelFrame(content_frame, text="📈 Real-Time Plot Control", 
                                          padding=20, style="Card.TLabelframe")
        plot_control_card.pack(fill=tk.X, pady=(0, 10))
        
        plot_inner = ttk.Frame(plot_control_card, style="Card.TFrame")
        plot_inner.pack(fill=tk.X)
        
        # Label instruksi
        ttk.Label(plot_inner, text="Select channels to display:", 
                 background=COLOR_FRAME, font=FONT_BOLD).pack(anchor=tk.W, pady=(0, 10))
        
        # Frame untuk checkbox channels (2 rows x 6 columns)
        checkbox_frame = ttk.Frame(plot_inner, style="Card.TFrame")
        checkbox_frame.pack(fill=tk.X)
        
        leads = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        for idx, lead in enumerate(leads):
            row = idx // 6
            col = idx % 6
            cb = ttk.Checkbutton(checkbox_frame, text=f"Lead {lead}", 
                                variable=self.selected_channels[lead],
                                style="TCheckbutton")
            cb.grid(row=row, column=col, sticky=tk.W, padx=10, pady=5)
        
        # Tombol kontrol plot
        plot_button_frame = ttk.Frame(plot_inner, style="Card.TFrame")
        plot_button_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.start_plot_button = ttk.Button(plot_button_frame, text="📊 Start Live Plot", 
                                           command=self.start_plotting, 
                                           style="Primary.TButton", state="disabled")
        self.start_plot_button.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 5))
        
        self.stop_plot_button = ttk.Button(plot_button_frame, text="⏹ Stop Plot", 
                                          command=self.stop_plotting, 
                                          style="Secondary.TButton", state="disabled")
        self.stop_plot_button.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(5, 0))

        # TAMBAHAN: Area untuk Live Plot (setelah log_card, sebelum status bar)
        plot_card = ttk.LabelFrame(content_frame, text="📈 Live Signal Plot", 
                                  padding=10, style="Card.TLabelframe")
        plot_card.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Set minimum height untuk plot area
        plot_card.config(height=400)  # Sama tinggi dengan log area
        
        # Placeholder untuk plot
        self.plot_frame = ttk.Frame(plot_card, style="Card.TFrame")
        self.plot_frame.pack(fill=tk.BOTH, expand=True)
        
        self.plot_placeholder = ttk.Label(self.plot_frame, 
                                         text="Live plot will appear here during recording\nClick 'Start Live Plot' after starting recording", 
                                         font=("Segoe UI", 11), 
                                         foreground=COLOR_TEXT_SECONDARY,
                                         background=COLOR_FRAME)
        self.plot_placeholder.pack(expand=True, pady=50)

        # ===============================================================================================
        
        # Log area dengan ukuran lebih besar
        log_card = ttk.LabelFrame(content_frame, text="📋 Activity Log", 
                                 padding=15, style="Card.TLabelframe")
        log_card.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        log_inner = ttk.Frame(log_card, style="Card.TFrame")
        log_inner.pack(fill=tk.BOTH, expand=True)
        
        # Tinggi log diperbesar
        self.log_text = tk.Text(log_inner, height=20, state="disabled", wrap=tk.WORD, 
                               font=FONT_LOG, bg="#1E1E1E", fg="#D4D4D4", 
                               insertbackground="white", relief="flat", 
                               padx=10, pady=10, borderwidth=0)
        log_scrollbar = ttk.Scrollbar(log_inner, orient="vertical", command=self.log_text.yview)
        self.log_text.config(yscrollcommand=log_scrollbar.set)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Status bar
        status_frame = ttk.Frame(self, style="Card.TFrame", height=40)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=15, pady=(0, 15))
        status_frame.pack_propagate(False)
        
        status_indicator = ttk.Label(status_frame, text="●", font=("Segoe UI", 16), 
                                    foreground=COLOR_ACCENT, background=COLOR_FRAME)
        status_indicator.pack(side=tk.LEFT, padx=(15, 5))
        
        self.status_label = ttk.Label(status_frame, textvariable=self.status_var, 
                                     font=FONT_PRIMARY, background=COLOR_FRAME, 
                                     foreground=COLOR_TEXT)
        self.status_label.pack(side=tk.LEFT, padx=(0, 15), pady=10)

    def _bind_mousewheel(self, event):
        """Bind mousewheel saat mouse masuk ke canvas"""
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>", self._on_mousewheel)
        self.canvas.bind_all("<Button-5>", self._on_mousewheel)
    
    def _unbind_mousewheel(self, event):
        """Unbind mousewheel saat mouse keluar dari canvas"""
        self.canvas.unbind_all("<MouseWheel>")
        self.canvas.unbind_all("<Button-4>")
        self.canvas.unbind_all("<Button-5>")
    
    def _on_mousewheel(self, event):
        """Handle mousewheel scrolling"""
        if event.num == 5 or event.delta < 0:
            self.canvas.yview_scroll(1, "units")
        elif event.num == 4 or event.delta > 0:
            self.canvas.yview_scroll(-1, "units")

    def _select_main_path(self):
        path = filedialog.askdirectory(title="Choose Primary Save Folder")
        if path:
            self.path_utama_var.set(path)
            
    def update_status(self, message):
        self.status_var.set(message)
        
    def log_message(self, message):
        self.after(0, self._update_log_widget, message)
        
    def _update_log_widget(self, message):
        self.log_text.config(state="normal")
        self.log_text.insert(tk.END, str(message) + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state="disabled")

    def start_recording(self):
        if not all([self.path_utama_var.get(), self.nama_subjek_var.get()]):
            messagebox.showerror("Input is incomplete", 
                               "Please select a Save Location and enter the Subject Name.",
                               parent=self)
            return
        if not os.path.isdir(self.fixed_model_path):
            messagebox.showwarning("Model folder not found", 
                                 f"Model Folder '{os.path.basename(self.fixed_model_path)}' is not found.",
                                 parent=self)
            return

        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        
        config = {
            'path_utama': self.path_utama_var.get(),
            'path_model': self.fixed_model_path,
            'nama_subjek': self.nama_subjek_var.get(),
            'timerecord': int(self.duration_var.get()),
            'nama_file_excel': f"Hasil_{self.nama_subjek_var.get()}.xlsx"
        }
        ip, port = self.ip_var.get(), int(self.port_var.get())

        if not self.processor.connect_to_esp32(ip, port):
            self.start_button.config(state="normal")
            self.stop_button.config(state="disabled")
            return
        
        self.processing_thread = threading.Thread(
            target=self.processor.start_main_loop, 
            args=(config,), 
            daemon=True
        )
        # TAMBAHAN: Enable tombol start plot setelah recording dimulai
        self.start_plot_button.config(state="normal")
        
        # TAMBAHAN: Set plot callback ke processor
        self.processor.plot_callback = self.update_plot_data
        
        self.processing_thread.start()

    def stop_recording(self):
        # TAMBAHAN: Stop plot jika aktif
        if self.plot_active:
            self.stop_plotting()
            
        if self.processor:
            self.processor.stop_processing()
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        # TAMBAHAN: Disable tombol plot
        self.start_plot_button.config(state="disabled")
        self.stop_plot_button.config(state="disabled")
        
# ==============================================================================================
# ===== METHOD BARU UNTUK LIVE PLOTTING =====
    
    def start_plotting(self):
        """Mulai live plotting"""
        if self.plot_active:
            return
        
        self.plot_active = True
        self.start_plot_button.config(state="disabled")
        self.stop_plot_button.config(state="normal")
        
        # Clear placeholder
        self.plot_placeholder.pack_forget()
        
        # Setup matplotlib figure
        self.fig, self.axes = plt.subplots(
            nrows=sum(1 for v in self.selected_channels.values() if v.get()),
            ncols=1,
            figsize=(10, 8),
            sharex=True
        )
        
        # Jika hanya 1 channel, axes bukan array
        if not isinstance(self.axes, np.ndarray):
            self.axes = [self.axes]
        
        # Setup axes untuk setiap channel yang dipilih
        ax_idx = 0
        for lead, var in self.selected_channels.items():
            if var.get():
                ax = self.axes[ax_idx]
                line, = ax.plot([], [], label=f'Lead {lead}', linewidth=0.8)
                self.lines[lead] = line
                ax.set_ylabel(f'{lead} (Raw)', fontsize=9)
                # ax.set_ylabel(f'{lead} (mV)', fontsize=9)
                ax.legend(loc='upper right', fontsize=8)
                ax.grid(True, alpha=0.3)
                ax_idx += 1
        
        self.axes[-1].set_xlabel('Sample', fontsize=9)
        self.fig.tight_layout()
        
        # Embed plot ke tkinter
        self.canvas_plot = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas_plot.draw()
        self.canvas_plot.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Start animation
        self.anim = FuncAnimation(self.fig, self._animate_plot, interval=50, blit=False)
        
        self.log_message("✅ Live plot started")
    
    def stop_plotting(self):
        """Stop live plotting"""
        if not self.plot_active:
            return
        
        self.plot_active = False
        self.start_plot_button.config(state="normal")
        self.stop_plot_button.config(state="disabled")
        
        # Stop animation
        if self.anim:
            self.anim.event_source.stop()
            self.anim = None
        
        # Clear plot
        if self.fig:
            plt.close(self.fig)
            self.fig = None
            self.axes = None
            self.lines = {}
        
        # Remove canvas
        if hasattr(self, 'canvas_plot'):
            self.canvas_plot.get_tk_widget().destroy()
            delattr(self, 'canvas_plot')
        
        # Show placeholder again
        self.plot_placeholder.pack(expand=True, pady=50)
        
        self.log_message("⏹ Live plot stopped")

    def update_plot_data(self, new_data):
        """Update data untuk plotting (dipanggil dari processor thread)"""
        if not self.plot_active:
            return
        
        # Append data ke deque
        for lead, value in new_data.items():
            if lead in self.plot_data:
                self.plot_data[lead].append(value)

    def _animate_plot(self, frame):
        """Animate function untuk FuncAnimation"""
        if not self.plot_active:
            return
        
        # Update setiap line dengan data terbaru
        for lead, line in self.lines.items():
            if lead in self.plot_data and len(self.plot_data[lead]) > 0:
                data = list(self.plot_data[lead])
                x_data = list(range(len(data)))
                line.set_data(x_data, data)
        
        # Auto-scale axes
        for ax in self.axes:
            ax.relim()
            ax.autoscale_view()
        
        return list(self.lines.values())
    
# ============================================================================================

# ===================================================================
# --- Halaman Ekstraksi Data dari File ---
# ===================================================================
class ExtractionFrame(ttk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        
        # Gunakan path yang kompatibel dengan PyInstaller
        if getattr(sys, 'frozen', False):
            script_dir = sys._MEIPASS
        else:
            script_dir = os.path.dirname(os.path.abspath(__file__))
        
        self.fixed_model_path = os.path.join(script_dir, "Model Final 2")
        
        self.input_file_path_var = tk.StringVar()
        self.output_dir_var = tk.StringVar()
        self.selected_lead_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Ready to start extracting")
        
        self.plot_figures = {}
        self.canvas_widget = None
        self.extractor = EKGExtractor(status_callback=self.update_status, 
                                     log_callback=self.log_message)
        
        self._create_widgets()

    def _create_widgets(self):
        # Header
        header_frame = ttk.Frame(self, style="Card.TFrame", height=70)
        header_frame.pack(fill=tk.X, padx=15, pady=15)
        header_frame.pack_propagate(False)
        
        back_button = ttk.Button(header_frame, text="← Back to Main Menu", 
                                command=lambda: self.controller.show_frame("MainMenuFrame"), 
                                style="Link.TButton")
        back_button.pack(side=tk.LEFT, padx=15, pady=15)
        
        title_label = ttk.Label(header_frame, text="Extract & Analyze CSV Data", 
                               font=FONT_SUBTITLE, background=COLOR_FRAME, 
                               foreground=COLOR_PRIMARY)
        title_label.pack(side=tk.LEFT, padx=10, pady=15)

        # Canvas untuk scrolling content
        self.main_canvas = tk.Canvas(self, bg=COLOR_BG, highlightthickness=0)
        main_scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.main_canvas.yview)
        
        # Content wrapper dengan scroll
        content_wrapper = ttk.Frame(self.main_canvas)
        
        content_wrapper.bind(
            "<Configure>",
            lambda e: self.main_canvas.configure(scrollregion=self.main_canvas.bbox("all"))
        )
        
        canvas_window = self.main_canvas.create_window((0, 0), window=content_wrapper, anchor="nw")
        self.main_canvas.configure(yscrollcommand=main_scrollbar.set)
        
        # Bind untuk auto-resize lebar canvas
        def _configure_main_canvas(event):
            self.main_canvas.itemconfig(canvas_window, width=event.width)
        self.main_canvas.bind("<Configure>", _configure_main_canvas)
        
        # Pack canvas
        self.main_canvas.pack(side="left", fill="both", expand=True, padx=15, pady=(0, 15))
        main_scrollbar.pack(side="right", fill="y", pady=(0, 15))
        
        # Bind mousewheel
        self.main_canvas.bind("<Enter>", self._bind_mousewheel)
        self.main_canvas.bind("<Leave>", self._unbind_mousewheel)

        # Top section - File settings
        top_section = ttk.Frame(content_wrapper)
        top_section.pack(fill=tk.X, pady=(0, 10))
        
        # Card untuk file input/output
        file_card = ttk.LabelFrame(top_section, text="📄 File Setting", 
                                   padding=20, style="Card.TLabelframe")
        file_card.pack(fill=tk.X)
        
        file_inner = ttk.Frame(file_card, style="Card.TFrame")
        file_inner.pack(fill=tk.X)
        file_inner.columnconfigure(1, weight=1)
        
        # Input file
        ttk.Label(file_inner, text="CSV Input File:", 
                 background=COLOR_FRAME).grid(row=0, column=0, sticky=tk.W, pady=8, padx=(0, 15))
        input_frame = ttk.Frame(file_inner, style="Card.TFrame")
        input_frame.grid(row=0, column=1, sticky=tk.EW, pady=8, padx=(0, 10))
        ttk.Entry(input_frame, textvariable=self.input_file_path_var, 
                 state='readonly').pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        ttk.Button(input_frame, text="Select File", 
                  command=self._select_input_file, 
                  style="Secondary.TButton").pack(side=tk.LEFT)
        
        # Output folder
        ttk.Label(file_inner, text="Output Folder:", 
                 background=COLOR_FRAME).grid(row=1, column=0, sticky=tk.W, pady=8, padx=(0, 15))
        output_frame = ttk.Frame(file_inner, style="Card.TFrame")
        output_frame.grid(row=1, column=1, sticky=tk.EW, pady=8, padx=(0, 10))
        ttk.Entry(output_frame, textvariable=self.output_dir_var, 
                 state='readonly').pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        ttk.Button(output_frame, text="Chose Folder", 
                  command=self._select_output_dir, 
                  style="Secondary.TButton").pack(side=tk.LEFT)
        
        # Process button
        button_frame = ttk.Frame(top_section, style="Card.TFrame")
        button_frame.pack(fill=tk.X, pady=10)
        
        self.process_button = ttk.Button(button_frame, text="🚀 Start Extraction", 
                                        command=self.start_extraction, 
                                        style="Accent.TButton")
        self.process_button.pack(expand=True, ipadx=30)

        # Main content - 2 kolom
        main_content = ttk.Frame(content_wrapper)
        main_content.pack(fill=tk.BOTH, expand=True)
        main_content.columnconfigure(0, weight=0, minsize=200)
        main_content.columnconfigure(1, weight=1)
        main_content.rowconfigure(0, weight=1)

        # Left sidebar - Lead selection
        sidebar_card = ttk.LabelFrame(main_content, text="📊 Select Lead", 
                                     padding=15, style="Card.TLabelframe")
        sidebar_card.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        
        # Scrollable lead selection
        lead_canvas = tk.Canvas(sidebar_card, bg=COLOR_FRAME, highlightthickness=0, 
                               width=180)
        lead_scrollbar = ttk.Scrollbar(sidebar_card, orient="vertical", 
                                      command=lead_canvas.yview)
        lead_frame = ttk.Frame(lead_canvas, style="Card.TFrame")
        
        lead_frame.bind("<Configure>", 
                       lambda e: lead_canvas.configure(scrollregion=lead_canvas.bbox("all")))
        
        lead_canvas.create_window((0, 0), window=lead_frame, anchor="nw")
        lead_canvas.configure(yscrollcommand=lead_scrollbar.set)
        
        lead_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        lead_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.lead_radio_buttons = []
        leads = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        
        for i, lead in enumerate(leads):
            rb = ttk.Radiobutton(lead_frame, text=f"Lead {lead}", 
                                variable=self.selected_lead_var, value=lead, 
                                command=self.show_plot, state="disabled",
                                style="TRadiobutton")
            rb.grid(row=i, column=0, sticky="w", pady=5, padx=10)
            self.lead_radio_buttons.append(rb)

        # Right panel - Plot dan Log dengan ukuran lebih besar
        right_panel = ttk.Frame(main_content)
        right_panel.grid(row=0, column=1, sticky="nsew")
        right_panel.rowconfigure(0, weight=2)
        right_panel.rowconfigure(1, weight=1)
        right_panel.columnconfigure(0, weight=1)

        # Plot area dengan scrollbar
        plot_card = ttk.LabelFrame(right_panel, text="📈 Chart View", 
                                   padding=10, style="Card.TLabelframe")
        plot_card.grid(row=0, column=0, sticky="nsew", pady=(0, 10))
        plot_card.rowconfigure(0, weight=1)
        plot_card.columnconfigure(0, weight=1)

        # Frame dengan canvas untuk scroll grafik
        plot_outer_frame = ttk.Frame(plot_card, style="Card.TFrame")
        plot_outer_frame.grid(row=0, column=0, sticky="nsew")
        plot_outer_frame.rowconfigure(0, weight=1)
        plot_outer_frame.columnconfigure(0, weight=1)
        
        # Canvas untuk plot dengan scrollbar
        self.plot_canvas_scroll = tk.Canvas(plot_outer_frame, bg=COLOR_FRAME, highlightthickness=0)
        plot_scrollbar_v = ttk.Scrollbar(plot_outer_frame, orient="vertical", 
                                        command=self.plot_canvas_scroll.yview)
        plot_scrollbar_h = ttk.Scrollbar(plot_outer_frame, orient="horizontal", 
                                        command=self.plot_canvas_scroll.xview)
        
        self.plot_canvas_frame = ttk.Frame(self.plot_canvas_scroll, style="Card.TFrame")
        
        self.plot_canvas_frame.bind(
            "<Configure>",
            lambda e: self.plot_canvas_scroll.configure(scrollregion=self.plot_canvas_scroll.bbox("all"))
        )
        
        self.plot_canvas_window = self.plot_canvas_scroll.create_window((0, 0), 
                                                                         window=self.plot_canvas_frame, 
                                                                         anchor="nw")
        self.plot_canvas_scroll.configure(yscrollcommand=plot_scrollbar_v.set,
                                         xscrollcommand=plot_scrollbar_h.set)
        
        # Grid layout untuk plot canvas
        self.plot_canvas_scroll.grid(row=0, column=0, sticky="nsew")
        plot_scrollbar_v.grid(row=0, column=1, sticky="ns")
        plot_scrollbar_h.grid(row=1, column=0, sticky="ew")
        
        self.plot_placeholder = ttk.Label(self.plot_canvas_frame, 
                                         text="The chart will show here after extraction completes", 
                                         font=("Segoe UI", 11), 
                                         foreground=COLOR_TEXT_SECONDARY,
                                         background=COLOR_FRAME)
        self.plot_placeholder.pack(expand=True, pady=100)
        
        # Fullscreen button
        button_container = ttk.Frame(plot_card, style="Card.TFrame")
        button_container.grid(row=2, column=0, sticky="se", pady=(5, 0), padx=5)
        
        self.fullscreen_button = ttk.Button(button_container, text="🔍 Fullscreen", 
                                           command=self.open_fullscreen_plot, 
                                           style="Secondary.TButton",
                                           state="disabled")
        self.fullscreen_button.pack()
        
        # Log area dengan ukuran lebih besar
        log_card = ttk.LabelFrame(right_panel, text="📋 Activity Log", 
                                 padding=15, style="Card.TLabelframe")
        log_card.grid(row=1, column=0, sticky="nsew")
        
        log_inner = ttk.Frame(log_card, style="Card.TFrame")
        log_inner.pack(fill=tk.BOTH, expand=True)
        
        # Tinggi log diperbesar
        self.log_text = tk.Text(log_inner, height=15, state="disabled", wrap=tk.WORD, 
                               font=FONT_LOG, bg="#1E1E1E", fg="#D4D4D4", 
                               insertbackground="white", relief="flat",
                               padx=10, pady=10, borderwidth=0)
        log_scrollbar = ttk.Scrollbar(log_inner, orient="vertical", command=self.log_text.yview)
        self.log_text.config(yscrollcommand=log_scrollbar.set)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Status bar
        status_frame = ttk.Frame(self, style="Card.TFrame", height=40)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=15, pady=(0, 15))
        status_frame.pack_propagate(False)
        
        status_indicator = ttk.Label(status_frame, text="●", font=("Segoe UI", 16), 
                                    foreground=COLOR_ACCENT, background=COLOR_FRAME)
        status_indicator.pack(side=tk.LEFT, padx=(15, 5))
        
        self.status_label = ttk.Label(status_frame, textvariable=self.status_var, 
                                     font=FONT_PRIMARY, background=COLOR_FRAME, 
                                     foreground=COLOR_TEXT)
        self.status_label.pack(side=tk.LEFT, padx=(0, 15), pady=10)

    def _bind_mousewheel(self, event):
        """Bind mousewheel saat mouse masuk ke canvas"""
        self.main_canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.main_canvas.bind_all("<Button-4>", self._on_mousewheel)
        self.main_canvas.bind_all("<Button-5>", self._on_mousewheel)
    
    def _unbind_mousewheel(self, event):
        """Unbind mousewheel saat mouse keluar dari canvas"""
        self.main_canvas.unbind_all("<MouseWheel>")
        self.main_canvas.unbind_all("<Button-4>")
        self.main_canvas.unbind_all("<Button-5>")
    
    def _on_mousewheel(self, event):
        """Handle mousewheel scrolling"""
        if event.num == 5 or event.delta < 0:
            self.main_canvas.yview_scroll(1, "units")
        elif event.num == 4 or event.delta > 0:
            self.main_canvas.yview_scroll(-1, "units")

    def _select_input_file(self):
        filepath = filedialog.askopenfilename(
            title="Select CSV Input File", 
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filepath:
            self.input_file_path_var.set(filepath)
    
    def _select_output_dir(self):
        path = filedialog.askdirectory(title="Select Excel Output Folder")
        if path:
            self.output_dir_var.set(path)

    def start_extraction(self):
        input_file, output_dir = self.input_file_path_var.get(), self.output_dir_var.get()
        if not all([input_file, output_dir]):
            messagebox.showerror("Incomplete Input", 
                               "Choose input file & output folder",
                               parent=self)
            return
        if not os.path.isdir(self.fixed_model_path):
            messagebox.showwarning("Model folder not found", 
                                 f"Model Folder '{os.path.basename(self.fixed_model_path)}' is not found",
                                 parent=self)
            return

        base_name = os.path.basename(input_file)
        file_name_no_ext = os.path.splitext(base_name)[0]
        output_path = os.path.join(output_dir, f"Hasil_{file_name_no_ext}.xlsx")
        
        self.process_button.config(state="disabled")
        self.log_text.config(state="normal")
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state="disabled")

        threading.Thread(
            target=self.run_extraction_thread, 
            args=(input_file, self.fixed_model_path, output_path), 
            daemon=True
        ).start()
    
    def run_extraction_thread(self, input_path, model_path, output_path):
        _, figures = self.extractor.process_file(input_path, model_path, output_path)
        self.after(0, self.on_extraction_complete, figures)

    def on_extraction_complete(self, figures):
        self.process_button.config(state="normal")
        if figures:
            self.plot_figures = figures
            for rb in self.lead_radio_buttons:
                rb.config(state="normal")
            
            self.fullscreen_button.config(state="normal")
            
            leads = list(figures.keys())
            if leads:
                self.selected_lead_var.set(leads[0])
                self.show_plot()
            
            messagebox.showinfo("Extraction completed", 
                              "Extraction completed successfully!",
                              parent=self)
        else:
            self.log_message("❌ Extraction finished with errors, no chart was produced.")
            messagebox.showerror("Exstraction failed", 
                               "Extraction process failed. Check the log for details.",
                               parent=self)

    def show_plot(self, event=None):
        lead = self.selected_lead_var.get()
        if lead in self.plot_figures:
            self.plot_placeholder.pack_forget()
            if self.canvas_widget:
                self.canvas_widget.get_tk_widget().destroy()
            
            fig = self.plot_figures[lead]
            
            # Set ukuran figure lebih besar untuk detail lebih jelas
            fig.set_size_inches(12, 8)
            
            self.canvas_widget = FigureCanvasTkAgg(fig, master=self.plot_canvas_frame)
            self.canvas_widget.draw()
            self.canvas_widget.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Update scroll region setelah plot ditampilkan
            self.plot_canvas_frame.update_idletasks()
            self.plot_canvas_scroll.configure(scrollregion=self.plot_canvas_scroll.bbox("all"))

    def open_fullscreen_plot(self):
        lead = self.selected_lead_var.get()
        if not lead or lead not in self.plot_figures:
            messagebox.showinfo("No chart available", 
                              "Select a valid lead after the extraction is complete.",
                              parent=self)
            return

        win = tk.Toplevel(self)
        win.title(f"Fullscreen Chart - Lead {lead}")
        win.geometry("1400x800")
        win.configure(bg=COLOR_FRAME)
        
        # Header
        header = ttk.Frame(win, style="Card.TFrame", height=60)
        header.pack(fill=tk.X, padx=15, pady=15)
        header.pack_propagate(False)
        
        ttk.Label(header, text=f"Lead {lead}", font=FONT_SUBTITLE, 
                 background=COLOR_FRAME, foreground=COLOR_PRIMARY).pack(side=tk.LEFT, padx=15)
        
        ttk.Button(header, text="✕ Close", command=win.destroy, 
                  style="Secondary.TButton").pack(side=tk.RIGHT, padx=15)

        # Main content
        content = ttk.Frame(win, style="Card.TFrame")
        content.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 15))
        
        fig = self.plot_figures[lead]
        canvas = FigureCanvasTkAgg(fig, master=content)
        canvas.draw()
        
        # Toolbar di atas
        toolbar_frame = ttk.Frame(content, style="Card.TFrame")
        toolbar_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))
        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
        toolbar.update()
        
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def update_status(self, message):
        self.status_var.set(message)
        
    def log_message(self, message):
        self.after(0, self._update_log_widget, message)
        
    def _update_log_widget(self, message):
        self.log_text.config(state="normal")
        self.log_text.insert(tk.END, str(message) + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state="disabled")

# ===================================================================
# --- Halaman Info Pengembang ---
# ===================================================================
class InfoFrame(ttk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        
        # Header dengan tombol navigasi
        header_frame = ttk.Frame(self, style="Card.TFrame", height=70)
        header_frame.pack(fill=tk.X, padx=15, pady=15)
        header_frame.pack_propagate(False)
        
        # Tombol kembali
        back_button = ttk.Button(header_frame, text="← Kembali", 
                                command=lambda: controller.show_frame("MainMenuFrame"), 
                                style="Link.TButton")
        back_button.pack(side=tk.LEFT, padx=15, pady=15)
        
        # Judul
        title_label = ttk.Label(header_frame, text="About the Developers", 
                               font=FONT_SUBTITLE, background=COLOR_FRAME, 
                               foreground=COLOR_PRIMARY)
        title_label.pack(side=tk.LEFT, padx=10, pady=15)
        
        # Tombol keluar
        exit_button = ttk.Button(header_frame, text="✕ Exit", 
                                command=controller.quit_app, 
                                style="DangerSmall.TButton")
        exit_button.pack(side=tk.RIGHT, padx=15, pady=15)
        
        # Canvas untuk scrolling
        self.canvas = tk.Canvas(self, bg=COLOR_BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        
        # Container dengan scroll
        scroll_container = ttk.Frame(self.canvas)
        
        scroll_container.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        canvas_window = self.canvas.create_window((0, 0), window=scroll_container, anchor="n")
        self.canvas.configure(yscrollcommand=scrollbar.set)
        
        # Bind untuk auto-resize lebar canvas
        def _configure_canvas(event):
            self.canvas.itemconfig(canvas_window, width=event.width)
        self.canvas.bind("<Configure>", _configure_canvas)
        
        # Pack canvas dan scrollbar
        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Bind mousewheel untuk scrolling
        self.canvas.bind("<Enter>", self._bind_mousewheel)
        self.canvas.bind("<Leave>", self._unbind_mousewheel)
        
        # Content container
        container = ttk.Frame(scroll_container)
        container.pack(expand=True, pady=40)
        
        # Info card
        info_card = ttk.Frame(container, style="Card.TFrame", relief="solid", borderwidth=1)
        info_card.pack(padx=40, pady=20)
        
        content_frame = ttk.Frame(info_card, style="Card.TFrame")
        content_frame.pack(padx=60, pady=40)
        
        # Institution
        institution = ttk.Label(content_frame, 
                               text="Departement of Electrical Engineering\nPadjadjaran University",
                               font=("Segoe UI", 16, "bold"), 
                               background=COLOR_FRAME,
                               foreground=COLOR_PRIMARY,
                               justify=tk.CENTER)
        institution.pack(pady=(0, 10))
        
        lab = ttk.Label(content_frame, 
                       text="Laboratorium Cogno-Technology & Artificial Intelligence",
                       font=("Segoe UI", 12), 
                       background=COLOR_FRAME,
                       foreground=COLOR_TEXT_SECONDARY,
                       justify=tk.CENTER)
        lab.pack(pady=(0, 30))
        
        # Separator
        separator1 = ttk.Separator(content_frame, orient="horizontal")
        separator1.pack(fill=tk.X, pady=20)
        
        # Hardware Developer
        hw_label = ttk.Label(content_frame, 
                            text="👨‍💻 Hardware Developer",
                            font=("Segoe UI", 12, "bold"), 
                            background=COLOR_FRAME,
                            foreground=COLOR_TEXT)
        hw_label.pack(pady=(10, 5))
        
        hw_name = ttk.Label(content_frame, 
                           text="Jeffry Fane",
                           font=("Segoe UI", 14), 
                           background=COLOR_FRAME,
                           foreground=COLOR_TEXT)
        hw_name.pack(pady=(0, 20))
        
        # Software Developer
        sw_label = ttk.Label(content_frame, 
                            text="💻 Software Developer",
                            font=("Segoe UI", 12, "bold"), 
                            background=COLOR_FRAME,
                            foreground=COLOR_TEXT)
        sw_label.pack(pady=(10, 5))
        
        sw_name = ttk.Label(content_frame, 
                           text="Willy Juliansyah",
                           font=("Segoe UI", 14), 
                           background=COLOR_FRAME,
                           foreground=COLOR_TEXT)
        sw_name.pack(pady=(0, 20))
        
        # Separator
        separator2 = ttk.Separator(content_frame, orient="horizontal")
        separator2.pack(fill=tk.X, pady=20)
        
        # Supervisors
        supervisor_label = ttk.Label(content_frame, 
                                     text="🎓 Supervisor",
                                     font=("Segoe UI", 12, "bold"), 
                                     background=COLOR_FRAME,
                                     foreground=COLOR_TEXT)
        supervisor_label.pack(pady=(10, 10))
        
        supervisors = [
            "Arjon Turnip, Ph.D.",
            "Fikri Rida Fadillah, S.T."
        ]
        
        for supervisor in supervisors:
            sup_name = ttk.Label(content_frame, 
                                text=supervisor,
                                font=("Segoe UI", 13), 
                                background=COLOR_FRAME,
                                foreground=COLOR_TEXT)
            sup_name.pack(pady=5)
        
        # Footer
        footer_frame = ttk.Frame(scroll_container)
        footer_frame.pack(pady=30)
        
        footer = ttk.Label(footer_frame, 
                          text="© 2025 Lab Cogno-Technology & Artificial Intelligence\nUniversitas Padjadjaran",
                          font=("Segoe UI", 9), 
                          foreground=COLOR_TEXT_SECONDARY, 
                          background=COLOR_BG,
                          justify=tk.CENTER)
        footer.pack()
    
    def _bind_mousewheel(self, event):
        """Bind mousewheel saat mouse masuk ke canvas"""
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>", self._on_mousewheel)
        self.canvas.bind_all("<Button-5>", self._on_mousewheel)
    
    def _unbind_mousewheel(self, event):
        """Unbind mousewheel saat mouse keluar dari canvas"""
        self.canvas.unbind_all("<MouseWheel>")
        self.canvas.unbind_all("<Button-4>")
        self.canvas.unbind_all("<Button-5>")
    
    def _on_mousewheel(self, event):
        """Handle mousewheel scrolling"""
        if event.num == 5 or event.delta < 0:
            self.canvas.yview_scroll(1, "units")
        elif event.num == 4 or event.delta > 0:
            self.canvas.yview_scroll(-1, "units")

if __name__ == "__main__":
    app = App()
    app.mainloop()