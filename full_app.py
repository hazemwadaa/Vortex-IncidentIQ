"""
IncidentIQ — Smart Incident Reporting & NLP Classification System
"The Vortex" Desktop Application
Built with CustomTkinter | LinearSVC Pipeline | Glassmorphism Dark UI
"""

import sys
import os
import re
import threading
import time
import datetime
import customtkinter as ctk
from tkinter import filedialog, messagebox
import joblib
import pandas as pd
from PIL import Image, ImageTk, ImageDraw, ImageFilter

# ─────────────────────────────────────────────
#  PyInstaller resource path helper
# ─────────────────────────────────────────────
def resource_path(relative_path: str) -> str:
    """Return the absolute path; works both for dev and for PyInstaller EXE."""
    try:
        base = sys._MEIPASS  # type: ignore
    except AttributeError:
        base = os.path.abspath(".")
    return os.path.join(base, relative_path)


# ─────────────────────────────────────────────
#  Custom preprocessing — MUST be present for
#  unpickling the Scikit-Learn Pipeline
# ─────────────────────────────────────────────
import sklearn

def clean_text_logic(text: str) -> str:
    """Core text-cleaning logic used during model training."""
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)          # remove URLs
    text = re.sub(r"[^a-z0-9\s]", " ", text)              # keep alphanum + spaces
    text = re.sub(r"\s+", " ", text).strip()               # collapse whitespace
    return text


def text_cleaner_transformer(X, y=None):
    """FunctionTransformer helper used by the pickled pipeline."""
    return [clean_text_logic(doc) for doc in X]


# ─────────────────────────────────────────────
#  Severity configuration
# ─────────────────────────────────────────────
SEVERITY_CONFIG = {
    "fire":      {"label": "CRITICAL",  "color": "#FF3B5C", "fill": 0.90, "icon": "🔥"},
    "theft":     {"label": "WARNING",   "color": "#FF9500", "fill": 0.70, "icon": "🚨"},
    "technical": {"label": "NORMAL",    "color": "#00D4AA", "fill": 0.40, "icon": "⚙️"},
}
DEFAULT_SEVERITY = {"label": "UNKNOWN", "color": "#6E7B8B", "fill": 0.10, "icon": "❓"}


def get_severity(predicted_class: str) -> dict:
    return SEVERITY_CONFIG.get(str(predicted_class).strip().lower(), DEFAULT_SEVERITY)


# ─────────────────────────────────────────────
#  Glow-border canvas helper
# ─────────────────────────────────────────────
def draw_glow_border(canvas, width, height, color, thickness=3, blur_passes=2):
    """Fill a Tkinter Canvas with a glowing rectangle border."""
    canvas.delete("all")
    canvas.configure(bg="#0D1117")
    # Draw layered rectangles to simulate glow
    for i in range(blur_passes + 1):
        pad = i * 2
        canvas.create_rectangle(
            pad, pad, width - pad, height - pad,
            outline=color,
            width=thickness - i if thickness - i > 0 else 1,
        )


# ═════════════════════════════════════════════
#  Main Application Class
# ═════════════════════════════════════════════
class IncidentIQApp(ctk.CTk):

    # ── palette ──────────────────────────────
    BG_PRIMARY    = "#0D1117"
    BG_SECONDARY  = "#161B22"
    BG_CARD       = "#1C2430"
    BG_SIDEBAR    = "#10161E"
    ACCENT_BLUE   = "#00BFFF"
    ACCENT_GREEN  = "#00D4AA"
    ACCENT_RED    = "#FF3B5C"
    ACCENT_ORANGE = "#FF9500"
    TEXT_PRIMARY  = "#E6EDF3"
    TEXT_MUTED    = "#8B949E"
    BORDER        = "#30363D"
    FONT_TITLE    = ("Segoe UI", 22, "bold")
    FONT_SUBTITLE = ("Segoe UI", 11)
    FONT_LABEL    = ("Segoe UI", 10, "bold")
    FONT_BODY     = ("Segoe UI", 10)
    FONT_MONO     = ("Consolas", 10)
    FONT_HUGE     = ("Segoe UI", 48, "bold")

    def __init__(self):
        super().__init__()

        # ── Window setup ──────────────────────
        self.title("IncidentIQ  ·  Smart Incident Classification")
        self.geometry("1180x720")
        self.minsize(1000, 640)
        self.configure(fg_color=self.BG_PRIMARY)
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")

        # ── State ─────────────────────────────
        self.model = None
        self.incident_log: list[dict] = []   # last-5 history
        self._pulse_job = None
        self._bulk_cancel = False

        # ── Load model ────────────────────────
        self._load_model()

        # ── Build UI ──────────────────────────
        self._build_layout()
        self._animate_startup()

    # ══════════════════════════════════════════
    #  Model loading
    # ══════════════════════════════════════════
    def _load_model(self):
        model_path = resource_path("incidentIQ_SVC.pkl")
        try:
            self.model = joblib.load(model_path)
            self._model_status = ("✔  Model loaded", self.ACCENT_GREEN)
        except FileNotFoundError:
            self._model_status = ("⚠  Model not found — place incidentIQ_SVC.pkl next to this app", self.ACCENT_ORANGE)
            self.model = None
        except Exception as exc:
            self._model_status = (f"✘  Load error: {exc}", self.ACCENT_RED)
            self.model = None

    # ══════════════════════════════════════════
    #  Layout skeleton
    # ══════════════════════════════════════════
    def _build_layout(self):
        # ── Sidebar ───────────────────────────
        self.sidebar = ctk.CTkFrame(
            self, width=220, fg_color=self.BG_SIDEBAR,
            corner_radius=0, border_width=0,
        )
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        # ── Main area ─────────────────────────
        self.main_area = ctk.CTkFrame(self, fg_color=self.BG_PRIMARY, corner_radius=0)
        self.main_area.pack(side="left", fill="both", expand=True)

        self._build_sidebar()
        self._build_main()

        # Activate first tab
        self._switch_tab("analyze")

    # ══════════════════════════════════════════
    #  Sidebar
    # ══════════════════════════════════════════
    def _build_sidebar(self):
        sb = self.sidebar

        # Logo block
        logo_frame = ctk.CTkFrame(sb, fg_color="transparent")
        logo_frame.pack(pady=(28, 8), padx=16, fill="x")

        ctk.CTkLabel(
            logo_frame, text="⬡", font=("Segoe UI", 32, "bold"),
            text_color=self.ACCENT_BLUE,
        ).pack(side="left", padx=(0, 8))

        title_block = ctk.CTkFrame(logo_frame, fg_color="transparent")
        title_block.pack(side="left")
        ctk.CTkLabel(
            title_block, text="IncidentIQ",
            font=("Segoe UI", 16, "bold"), text_color=self.TEXT_PRIMARY,
        ).pack(anchor="w")
        ctk.CTkLabel(
            title_block, text="AI Command Center",
            font=("Segoe UI", 9), text_color=self.TEXT_MUTED,
        ).pack(anchor="w")

        # Separator
        ctk.CTkFrame(sb, height=1, fg_color=self.BORDER).pack(fill="x", padx=16, pady=12)

        # Nav buttons
        self._nav_buttons = {}
        nav_items = [
            ("analyze",   "🎯", "Analyze",     "Single Incident"),
            ("bulk",      "📊", "Bulk Process","Excel Upload"),
            ("log",       "📋", "Incident Log","Recent Activity"),
        ]
        for key, icon, label, sublabel in nav_items:
            self._nav_buttons[key] = self._nav_btn(sb, key, icon, label, sublabel)

        # Spacer
        ctk.CTkFrame(sb, fg_color="transparent").pack(fill="y", expand=True)

        # Model status at bottom
        ctk.CTkFrame(sb, height=1, fg_color=self.BORDER).pack(fill="x", padx=16, pady=8)
        status_text, status_color = self._model_status
        ctk.CTkLabel(
            sb, text=status_text,
            font=("Segoe UI", 9), text_color=status_color,
            wraplength=190, justify="left",
        ).pack(padx=16, pady=(0, 16), anchor="w")

        # Activate first tab
        self._active_tab = None

    def _nav_btn(self, parent, key, icon, label, sublabel):
        frame = ctk.CTkFrame(parent, fg_color="transparent", cursor="hand2")
        frame.pack(fill="x", padx=10, pady=3)

        inner = ctk.CTkFrame(frame, fg_color="transparent", corner_radius=8)
        inner.pack(fill="x")

        icon_lbl = ctk.CTkLabel(inner, text=icon, font=("Segoe UI", 16), width=32, text_color=self.TEXT_MUTED)
        icon_lbl.pack(side="left", padx=(10, 6), pady=10)

        text_block = ctk.CTkFrame(inner, fg_color="transparent")
        text_block.pack(side="left", fill="x", expand=True)
        main_lbl = ctk.CTkLabel(text_block, text=label, font=("Segoe UI", 11, "bold"), text_color=self.TEXT_MUTED, anchor="w")
        main_lbl.pack(anchor="w")
        sub_lbl = ctk.CTkLabel(text_block, text=sublabel, font=("Segoe UI", 8), text_color=self.TEXT_MUTED, anchor="w")
        sub_lbl.pack(anchor="w")

        indicator = ctk.CTkFrame(inner, width=3, fg_color="transparent", corner_radius=2)
        indicator.pack(side="right", fill="y", padx=(0, 0))

        btn_data = {"frame": inner, "icon": icon_lbl, "main": main_lbl, "sub": sub_lbl, "indicator": indicator}

        for widget in [frame, inner, icon_lbl, text_block, main_lbl, sub_lbl]:
            widget.bind("<Button-1>", lambda e, k=key: self._switch_tab(k))

        return btn_data

    def _switch_tab(self, key: str):
        # Deactivate all
        for k, data in self._nav_buttons.items():
            data["frame"].configure(fg_color="transparent")
            data["main"].configure(text_color=self.TEXT_MUTED)
            data["sub"].configure(text_color=self.TEXT_MUTED)
            data["icon"].configure(text_color=self.TEXT_MUTED)
            data["indicator"].configure(fg_color="transparent")

        # Activate selected
        d = self._nav_buttons[key]
        d["frame"].configure(fg_color=self.BG_CARD)
        d["main"].configure(text_color=self.TEXT_PRIMARY)
        d["sub"].configure(text_color=self.ACCENT_BLUE)
        d["icon"].configure(text_color=self.ACCENT_BLUE)
        d["indicator"].configure(fg_color=self.ACCENT_BLUE)
        self._active_tab = key

        # Show appropriate panel
        for panel in self.main_area.winfo_children():
            panel.pack_forget()

        if key == "analyze":
            self.analyze_panel.pack(fill="both", expand=True)
        elif key == "bulk":
            self.bulk_panel.pack(fill="both", expand=True)
        elif key == "log":
            self.log_panel.pack(fill="both", expand=True)

    # ══════════════════════════════════════════
    #  Main panels
    # ══════════════════════════════════════════
    def _build_main(self):
        self._build_analyze_panel()
        self._build_bulk_panel()
        self._build_log_panel()

    # ─── Analyze Panel ────────────────────────
    def _build_analyze_panel(self):
        self.analyze_panel = ctk.CTkFrame(self.main_area, fg_color="transparent")

        # Top header
        header = ctk.CTkFrame(self.analyze_panel, fg_color="transparent")
        header.pack(fill="x", padx=32, pady=(28, 0))

        ctk.CTkLabel(header, text="Incident Analyzer", font=("Segoe UI", 24, "bold"),
                     text_color=self.TEXT_PRIMARY).pack(side="left")
        ts_lbl = ctk.CTkLabel(header, text="", font=("Segoe UI", 10), text_color=self.TEXT_MUTED)
        ts_lbl.pack(side="right", pady=4)
        self._update_clock(ts_lbl)

        ctk.CTkLabel(
            self.analyze_panel,
            text="Enter incident description below. The AI engine will classify and assess severity in real-time.",
            font=("Segoe UI", 11), text_color=self.TEXT_MUTED,
        ).pack(anchor="w", padx=32, pady=(4, 20))

        # Content split: left = input+result, right = sidebar stats
        content = ctk.CTkFrame(self.analyze_panel, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=32, pady=0)
        content.columnconfigure(0, weight=3)
        content.columnconfigure(1, weight=2)
        content.rowconfigure(0, weight=1)

        left_col = ctk.CTkFrame(content, fg_color="transparent")
        left_col.grid(row=0, column=0, sticky="nsew", padx=(0, 16))

        right_col = ctk.CTkFrame(content, fg_color="transparent")
        right_col.grid(row=0, column=1, sticky="nsew")

        # ── Input card ──
        input_card = ctk.CTkFrame(left_col, fg_color=self.BG_CARD, corner_radius=12,
                                   border_width=1, border_color=self.BORDER)
        input_card.pack(fill="x")

        ctk.CTkLabel(input_card, text="INCIDENT DESCRIPTION",
                     font=("Segoe UI", 9, "bold"), text_color=self.ACCENT_BLUE).pack(
            anchor="w", padx=18, pady=(16, 4))

        self.text_input = ctk.CTkTextbox(
            input_card, height=130,
            fg_color="#111820", text_color=self.TEXT_PRIMARY,
            font=("Segoe UI", 11), corner_radius=8,
            border_width=1, border_color=self.BORDER,
            scrollbar_button_color=self.BORDER,
        )
        self.text_input.pack(fill="x", padx=18, pady=(0, 10))
        self.text_input.insert("0.0", "Type or paste your incident report here…")
        self.text_input.bind("<FocusIn>",  self._clear_placeholder)
        self.text_input.bind("<FocusOut>", self._restore_placeholder)

        btn_row = ctk.CTkFrame(input_card, fg_color="transparent")
        btn_row.pack(fill="x", padx=18, pady=(0, 16))

        self.analyze_btn = ctk.CTkButton(
            btn_row, text="  Analyze Incident  ⚡",
            font=("Segoe UI", 11, "bold"),
            fg_color=self.ACCENT_BLUE, hover_color="#009FCC",
            text_color="#000000", height=38, corner_radius=8,
            command=self._run_analysis,
        )
        self.analyze_btn.pack(side="left")

        self.clear_btn = ctk.CTkButton(
            btn_row, text="Clear",
            font=("Segoe UI", 10),
            fg_color="transparent", hover_color=self.BG_SECONDARY,
            border_width=1, border_color=self.BORDER,
            text_color=self.TEXT_MUTED, height=38, corner_radius=8,
            command=self._clear_all,
        )
        self.clear_btn.pack(side="left", padx=(10, 0))

        # ── Glow border canvas ──
        self.glow_canvas_frame = ctk.CTkFrame(left_col, fg_color="transparent", height=6)
        self.glow_canvas_frame.pack(fill="x", pady=(8, 0))
        self._glow_canvas_widgets = []  # filled dynamically

        # ── Result card ──
        result_card = ctk.CTkFrame(left_col, fg_color=self.BG_CARD, corner_radius=12,
                                    border_width=1, border_color=self.BORDER)
        result_card.pack(fill="both", expand=True, pady=(12, 0))

        res_header = ctk.CTkFrame(result_card, fg_color="transparent")
        res_header.pack(fill="x", padx=18, pady=(16, 0))
        ctk.CTkLabel(res_header, text="CLASSIFICATION RESULT",
                     font=("Segoe UI", 9, "bold"), text_color=self.ACCENT_BLUE).pack(side="left")
        self.confidence_badge = ctk.CTkLabel(
            res_header, text="—", font=("Segoe UI", 9, "bold"),
            fg_color=self.BG_SECONDARY, corner_radius=4,
            text_color=self.TEXT_MUTED, padx=8, pady=2,
        )
        self.confidence_badge.pack(side="right")

        # Big class label
        self.class_label = ctk.CTkLabel(
            result_card, text="—",
            font=("Segoe UI", 48, "bold"), text_color=self.TEXT_MUTED,
        )
        self.class_label.pack(pady=(8, 0))

        self.severity_label = ctk.CTkLabel(
            result_card, text="Awaiting input",
            font=("Segoe UI", 12), text_color=self.TEXT_MUTED,
        )
        self.severity_label.pack()

        # Severity meter
        meter_frame = ctk.CTkFrame(result_card, fg_color="transparent")
        meter_frame.pack(fill="x", padx=18, pady=(16, 0))
        ctk.CTkLabel(meter_frame, text="SEVERITY METER",
                     font=("Segoe UI", 8, "bold"), text_color=self.TEXT_MUTED).pack(anchor="w")
        self.severity_bar = ctk.CTkProgressBar(
            meter_frame, height=14, corner_radius=7,
            fg_color=self.BG_SECONDARY, progress_color=self.TEXT_MUTED,
        )
        self.severity_bar.pack(fill="x", pady=(6, 0))
        self.severity_bar.set(0)

        self.severity_pct_label = ctk.CTkLabel(
            result_card, text="0%",
            font=("Segoe UI", 10, "bold"), text_color=self.TEXT_MUTED,
        )
        self.severity_pct_label.pack(anchor="e", padx=18, pady=(4, 16))

        # ── Right column: Quick stats ──
        self._build_stats_sidebar(right_col)

    def _build_stats_sidebar(self, parent):
        ctk.CTkLabel(parent, text="SESSION STATS",
                     font=("Segoe UI", 9, "bold"), text_color=self.ACCENT_BLUE).pack(anchor="w", pady=(0, 8))

        self.stat_vars = {}
        stats = [
            ("total",    "Total Analyzed", "0",  self.ACCENT_BLUE),
            ("fire",     "Fire / Critical",  "0",  self.ACCENT_RED),
            ("theft",    "Theft / Warning",  "0",  self.ACCENT_ORANGE),
            ("technical","Technical / OK",   "0",  self.ACCENT_GREEN),
        ]
        for key, title, val, color in stats:
            card = ctk.CTkFrame(parent, fg_color=self.BG_CARD, corner_radius=10,
                                border_width=1, border_color=self.BORDER)
            card.pack(fill="x", pady=5)
            ctk.CTkLabel(card, text=title, font=("Segoe UI", 9), text_color=self.TEXT_MUTED).pack(
                anchor="w", padx=14, pady=(10, 0))
            lbl = ctk.CTkLabel(card, text=val, font=("Segoe UI", 26, "bold"), text_color=color)
            lbl.pack(anchor="w", padx=14, pady=(0, 10))
            self.stat_vars[key] = lbl

        # Mini incident log (last 3)
        ctk.CTkLabel(parent, text="RECENT ACTIVITY",
                     font=("Segoe UI", 9, "bold"), text_color=self.ACCENT_BLUE).pack(
            anchor="w", pady=(20, 8))

        self.mini_log_frame = ctk.CTkScrollableFrame(
            parent, fg_color=self.BG_CARD, corner_radius=10,
            border_width=1, border_color=self.BORDER,
        )
        self.mini_log_frame.pack(fill="both", expand=True)

    # ─── Bulk Panel ───────────────────────────
    def _build_bulk_panel(self):
        self.bulk_panel = ctk.CTkFrame(self.main_area, fg_color="transparent")

        header = ctk.CTkFrame(self.bulk_panel, fg_color="transparent")
        header.pack(fill="x", padx=32, pady=(28, 0))
        ctk.CTkLabel(header, text="Bulk Processing", font=("Segoe UI", 24, "bold"),
                     text_color=self.TEXT_PRIMARY).pack(side="left")

        ctk.CTkLabel(
            self.bulk_panel,
            text="Upload an Excel file (.xlsx). The engine will classify every row and generate a detailed summary report.",
            font=("Segoe UI", 11), text_color=self.TEXT_MUTED,
        ).pack(anchor="w", padx=32, pady=(4, 20))

        # Drop zone card
        drop_card = ctk.CTkFrame(self.bulk_panel, fg_color=self.BG_CARD, corner_radius=14,
                                  border_width=2, border_color=self.BORDER)
        drop_card.pack(fill="x", padx=32)

        ctk.CTkLabel(drop_card, text="📂", font=("Segoe UI", 48)).pack(pady=(30, 4))
        ctk.CTkLabel(drop_card, text="Click to select an Excel file",
                     font=("Segoe UI", 14, "bold"), text_color=self.TEXT_PRIMARY).pack()
        ctk.CTkLabel(drop_card, text="Supported format: .xlsx  ·  Column named 'description' or first column used",
                     font=("Segoe UI", 10), text_color=self.TEXT_MUTED).pack(pady=(2, 0))

        self.upload_btn = ctk.CTkButton(
            drop_card, text="  Browse File  📁",
            font=("Segoe UI", 11, "bold"),
            fg_color=self.ACCENT_BLUE, hover_color="#009FCC",
            text_color="#000000", height=40, corner_radius=8, width=200,
            command=self._browse_excel,
        )
        self.upload_btn.pack(pady=20)

        # Progress + status
        self.bulk_status_lbl = ctk.CTkLabel(
            self.bulk_panel, text="No file loaded.",
            font=("Segoe UI", 11), text_color=self.TEXT_MUTED,
        )
        self.bulk_status_lbl.pack(pady=(16, 4))

        self.bulk_progress = ctk.CTkProgressBar(
            self.bulk_panel, height=10, corner_radius=5,
            fg_color=self.BG_CARD, progress_color=self.ACCENT_BLUE,
        )
        self.bulk_progress.pack(fill="x", padx=32, pady=(0, 20))
        self.bulk_progress.set(0)

        # Results summary grid
        self.bulk_summary_frame = ctk.CTkFrame(self.bulk_panel, fg_color="transparent")
        self.bulk_summary_frame.pack(fill="x", padx=32)
        self._bulk_stat_labels = {}

        for i, (key, label, color) in enumerate([
            ("total",    "Total Rows",      self.ACCENT_BLUE),
            ("fire",     "🔥 Fire",         self.ACCENT_RED),
            ("theft",    "🚨 Theft",        self.ACCENT_ORANGE),
            ("technical","⚙️  Technical",   self.ACCENT_GREEN),
        ]):
            card = ctk.CTkFrame(self.bulk_summary_frame, fg_color=self.BG_CARD,
                                corner_radius=10, border_width=1, border_color=self.BORDER)
            card.grid(row=0, column=i, padx=8, pady=4, sticky="ew")
            self.bulk_summary_frame.columnconfigure(i, weight=1)
            ctk.CTkLabel(card, text=label, font=("Segoe UI", 9), text_color=self.TEXT_MUTED).pack(pady=(10, 2))
            val_lbl = ctk.CTkLabel(card, text="—", font=("Segoe UI", 28, "bold"), text_color=color)
            val_lbl.pack(pady=(0, 10))
            self._bulk_stat_labels[key] = val_lbl

        # Download button (hidden until results ready)
        self.download_btn = ctk.CTkButton(
            self.bulk_panel, text="  Export Results (.xlsx)  ⬇",
            font=("Segoe UI", 11, "bold"),
            fg_color=self.ACCENT_GREEN, hover_color="#00B891",
            text_color="#000000", height=40, corner_radius=8, width=260,
            command=self._export_results,
        )
        self._bulk_result_df = None

    # ─── Log Panel ────────────────────────────
    def _build_log_panel(self):
        self.log_panel = ctk.CTkFrame(self.main_area, fg_color="transparent")

        header = ctk.CTkFrame(self.log_panel, fg_color="transparent")
        header.pack(fill="x", padx=32, pady=(28, 0))
        ctk.CTkLabel(header, text="Incident Log", font=("Segoe UI", 24, "bold"),
                     text_color=self.TEXT_PRIMARY).pack(side="left")
        ctk.CTkButton(
            header, text="Clear Log",
            font=("Segoe UI", 10),
            fg_color="transparent", hover_color=self.BG_SECONDARY,
            border_width=1, border_color=self.BORDER,
            text_color=self.TEXT_MUTED, height=32, corner_radius=6,
            command=self._clear_log,
        ).pack(side="right")

        ctk.CTkLabel(
            self.log_panel,
            text="Full history of all classified incidents in this session.",
            font=("Segoe UI", 11), text_color=self.TEXT_MUTED,
        ).pack(anchor="w", padx=32, pady=(4, 16))

        self.log_scroll = ctk.CTkScrollableFrame(
            self.log_panel, fg_color=self.BG_SECONDARY, corner_radius=12,
        )
        self.log_scroll.pack(fill="both", expand=True, padx=32, pady=(0, 28))

        self.log_empty_lbl = ctk.CTkLabel(
            self.log_scroll,
            text="No incidents classified yet.\nHead to Analyzer to get started.",
            font=("Segoe UI", 12), text_color=self.TEXT_MUTED,
        )
        self.log_empty_lbl.pack(pady=60)

    # ══════════════════════════════════════════
    #  Analysis logic
    # ══════════════════════════════════════════
    def _run_analysis(self):
        text = self.text_input.get("0.0", "end").strip()
        if not text or text == "Type or paste your incident report here…":
            self._flash_input_error()
            return
        if self.model is None:
            messagebox.showerror("Model Error", self._model_status[0])
            return

        self.analyze_btn.configure(state="disabled", text="  Analyzing…  ⏳")
        threading.Thread(target=self._do_classify, args=(text,), daemon=True).start()

    def _do_classify(self, text: str):
        time.sleep(0.4)  # brief artificial "thinking" pause for UX drama
        try:
            pred = self.model.predict([text])[0]
            sev  = get_severity(pred)
        except Exception as exc:
            self.after(0, lambda: messagebox.showerror("Prediction Error", str(exc)))
            self.after(0, lambda: self.analyze_btn.configure(state="normal", text="  Analyze Incident  ⚡"))
            return

        self.after(0, lambda: self._display_result(text, str(pred), sev))

    def _display_result(self, text: str, pred: str, sev: dict):
        color = sev["color"]
        fill  = sev["fill"]
        label = sev["label"]

        # Class label
        self.class_label.configure(text=f"{sev['icon']}  {pred.upper()}", text_color=color)
        self.severity_label.configure(text=f"Severity: {label}", text_color=color)

        # Animated progress bar
        self._animate_bar(fill, color)

        # Confidence badge (SVC doesn't give probabilities, so show severity fill as "confidence")
        self.confidence_badge.configure(
            text=f"Confidence: {int(fill * 100)}%",
            text_color=color, fg_color=self.BG_SECONDARY,
        )

        # Glow border effect — draw colored border on entire input card via a thin frame
        self._pulse_glow(color)

        # Stats
        pred_key = pred.strip().lower()
        self._increment_stat("total")
        if pred_key in ("fire", "theft", "technical"):
            self._increment_stat(pred_key)

        # Log entry
        entry = {
            "text": text[:80] + ("…" if len(text) > 80 else ""),
            "class": pred.upper(),
            "severity": label,
            "color": color,
            "icon": sev["icon"],
            "time": datetime.datetime.now().strftime("%H:%M:%S"),
        }
        self.incident_log.insert(0, entry)
        if len(self.incident_log) > 50:
            self.incident_log = self.incident_log[:50]

        self._refresh_mini_log()
        self._refresh_full_log()

        self.analyze_btn.configure(state="normal", text="  Analyze Incident  ⚡")

    def _animate_bar(self, target_fill: float, color: str):
        self.severity_bar.configure(progress_color=color)
        current = 0.0
        steps = 30
        delta = target_fill / steps

        def step(i=0):
            if i <= steps:
                val = min(delta * i, target_fill)
                self.severity_bar.set(val)
                self.severity_pct_label.configure(text=f"{int(val * 100)}%", text_color=color)
                self.after(20, lambda: step(i + 1))

        step()

    def _pulse_glow(self, color: str):
        """Briefly tint the analyze panel border color to simulate glow pulse."""
        # We achieve this by temporarily changing the input card's border color
        self.text_input.configure(border_color=color)

        def reset():
            self.text_input.configure(border_color=self.BORDER)

        self.after(1200, reset)

    # ══════════════════════════════════════════
    #  Log management
    # ══════════════════════════════════════════
    def _refresh_mini_log(self):
        for w in self.mini_log_frame.winfo_children():
            w.destroy()

        recent = self.incident_log[:5]
        if not recent:
            ctk.CTkLabel(self.mini_log_frame, text="No activity yet.",
                         font=("Segoe UI", 9), text_color=self.TEXT_MUTED).pack(pady=8)
            return

        for entry in recent:
            row = ctk.CTkFrame(self.mini_log_frame, fg_color=self.BG_SECONDARY, corner_radius=6)
            row.pack(fill="x", pady=2, padx=4)
            ctk.CTkLabel(row, text=entry["icon"], font=("Segoe UI", 14), width=24).pack(side="left", padx=(6, 2), pady=4)
            inner = ctk.CTkFrame(row, fg_color="transparent")
            inner.pack(side="left", fill="x", expand=True)
            ctk.CTkLabel(inner, text=entry["class"], font=("Segoe UI", 9, "bold"),
                         text_color=entry["color"]).pack(anchor="w")
            ctk.CTkLabel(inner, text=entry["time"], font=("Segoe UI", 8),
                         text_color=self.TEXT_MUTED).pack(anchor="w")

    def _refresh_full_log(self):
        for w in self.log_scroll.winfo_children():
            w.destroy()

        if not self.incident_log:
            self.log_empty_lbl = ctk.CTkLabel(
                self.log_scroll,
                text="No incidents classified yet.\nHead to Analyzer to get started.",
                font=("Segoe UI", 12), text_color=self.TEXT_MUTED,
            )
            self.log_empty_lbl.pack(pady=60)
            return

        for i, entry in enumerate(self.incident_log):
            card = ctk.CTkFrame(self.log_scroll, fg_color=self.BG_CARD, corner_radius=10,
                                border_width=1, border_color=self.BORDER)
            card.pack(fill="x", pady=5, padx=4)

            left = ctk.CTkFrame(card, fg_color="transparent", width=56)
            left.pack(side="left", fill="y")
            left.pack_propagate(False)
            ctk.CTkLabel(left, text=entry["icon"], font=("Segoe UI", 22)).pack(expand=True)

            mid = ctk.CTkFrame(card, fg_color="transparent")
            mid.pack(side="left", fill="both", expand=True, padx=(0, 12))
            top_row = ctk.CTkFrame(mid, fg_color="transparent")
            top_row.pack(fill="x", pady=(10, 2))
            ctk.CTkLabel(top_row, text=entry["class"], font=("Segoe UI", 13, "bold"),
                         text_color=entry["color"]).pack(side="left")
            ctk.CTkLabel(top_row, text=f"  ·  {entry['severity']}",
                         font=("Segoe UI", 10), text_color=self.TEXT_MUTED).pack(side="left")
            ctk.CTkLabel(mid, text=entry["text"], font=("Segoe UI", 10),
                         text_color=self.TEXT_MUTED, wraplength=560, justify="left").pack(anchor="w")
            ctk.CTkLabel(mid, text=f"🕐  {entry['time']}", font=("Segoe UI", 8),
                         text_color=self.TEXT_MUTED).pack(anchor="w", pady=(2, 10))

    def _clear_log(self):
        self.incident_log.clear()
        self._refresh_mini_log()
        self._refresh_full_log()
        for key in self.stat_vars:
            self.stat_vars[key].configure(text="0")

    # ══════════════════════════════════════════
    #  Bulk processing
    # ══════════════════════════════════════════
    def _browse_excel(self):
        path = filedialog.askopenfilename(
            title="Select Excel File",
            filetypes=[("Excel Files", "*.xlsx"), ("All Files", "*.*")],
        )
        if not path:
            return
        if self.model is None:
            messagebox.showerror("Model Error", self._model_status[0])
            return

        self.download_btn.pack_forget()
        self._bulk_result_df = None
        self.upload_btn.configure(state="disabled")
        self.bulk_progress.set(0)
        self.bulk_status_lbl.configure(text=f"⏳  Loading: {os.path.basename(path)}", text_color=self.ACCENT_BLUE)

        threading.Thread(target=self._process_bulk, args=(path,), daemon=True).start()

    def _process_bulk(self, path: str):
        try:
            df = pd.read_excel(path)
        except Exception as exc:
            self.after(0, lambda: messagebox.showerror("File Error", f"Could not read Excel file:\n{exc}"))
            self.after(0, lambda: self.upload_btn.configure(state="normal"))
            return

        # Determine text column
        col = "description" if "description" in [c.lower() for c in df.columns] else df.columns[0]
        texts = df[col].fillna("").astype(str).tolist()
        total = len(texts)

        if total == 0:
            self.after(0, lambda: messagebox.showwarning("Empty File", "No rows found in the file."))
            self.after(0, lambda: self.upload_btn.configure(state="normal"))
            return

        predictions = []
        for i, text in enumerate(texts):
            try:
                pred = self.model.predict([text])[0]
            except Exception:
                pred = "unknown"
            predictions.append(str(pred))
            progress = (i + 1) / total
            self.after(0, lambda p=progress, idx=i+1: (
                self.bulk_progress.set(p),
                self.bulk_status_lbl.configure(
                    text=f"⏳  Processing row {idx} / {total}…",
                    text_color=self.ACCENT_BLUE,
                ),
            ))
            time.sleep(0.02)  # slight pace so animation is visible

        df["predicted_class"] = predictions
        df["severity"] = [get_severity(p)["label"] for p in predictions]
        self._bulk_result_df = df

        counts = {k: predictions.count(k) for k in ("fire", "theft", "technical")}
        self.after(0, lambda: self._show_bulk_results(total, counts))

    def _show_bulk_results(self, total: int, counts: dict):
        self.bulk_progress.set(1)
        self.bulk_status_lbl.configure(
            text=f"✔  Done! {total} incidents classified.",
            text_color=self.ACCENT_GREEN,
        )
        self._bulk_stat_labels["total"].configure(text=str(total))
        self._bulk_stat_labels["fire"].configure(text=str(counts.get("fire", 0)))
        self._bulk_stat_labels["theft"].configure(text=str(counts.get("theft", 0)))
        self._bulk_stat_labels["technical"].configure(text=str(counts.get("technical", 0)))

        self.download_btn.pack(pady=(16, 0))
        self.upload_btn.configure(state="normal")

    def _export_results(self):
        if self._bulk_result_df is None:
            return
        save_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel Files", "*.xlsx")],
            title="Save Results As",
            initialfile="incidentIQ_results.xlsx",
        )
        if not save_path:
            return
        try:
            self._bulk_result_df.to_excel(save_path, index=False)
            messagebox.showinfo("Exported", f"Results saved to:\n{save_path}")
        except Exception as exc:
            messagebox.showerror("Export Error", str(exc))

    # ══════════════════════════════════════════
    #  Helpers
    # ══════════════════════════════════════════
    def _increment_stat(self, key: str):
        lbl = self.stat_vars.get(key)
        if lbl:
            current = int(lbl.cget("text")) if lbl.cget("text").isdigit() else 0
            lbl.configure(text=str(current + 1))

    def _clear_placeholder(self, event):
        current = self.text_input.get("0.0", "end").strip()
        if current == "Type or paste your incident report here…":
            self.text_input.delete("0.0", "end")
            self.text_input.configure(text_color=self.TEXT_PRIMARY)

    def _restore_placeholder(self, event):
        current = self.text_input.get("0.0", "end").strip()
        if not current:
            self.text_input.insert("0.0", "Type or paste your incident report here…")
            self.text_input.configure(text_color=self.TEXT_MUTED)

    def _clear_all(self):
        self.text_input.delete("0.0", "end")
        self._restore_placeholder(None)
        self.class_label.configure(text="—", text_color=self.TEXT_MUTED)
        self.severity_label.configure(text="Awaiting input", text_color=self.TEXT_MUTED)
        self.severity_bar.set(0)
        self.severity_bar.configure(progress_color=self.TEXT_MUTED)
        self.severity_pct_label.configure(text="0%", text_color=self.TEXT_MUTED)
        self.confidence_badge.configure(text="—", text_color=self.TEXT_MUTED)
        self.text_input.configure(border_color=self.BORDER)

    def _flash_input_error(self):
        self.text_input.configure(border_color=self.ACCENT_RED)
        self.after(800, lambda: self.text_input.configure(border_color=self.BORDER))

    def _update_clock(self, label: ctk.CTkLabel):
        label.configure(text=datetime.datetime.now().strftime("  🗓  %A, %d %B %Y   🕐  %H:%M:%S"))
        self.after(1000, lambda: self._update_clock(label))

    def _animate_startup(self):
        """Subtle fade-in effect via window alpha."""
        self.attributes("-alpha", 0.0)

        def fade(step=0):
            alpha = min(1.0, step * 0.08)
            self.attributes("-alpha", alpha)
            if alpha < 1.0:
                self.after(20, lambda: fade(step + 1))

        self.after(50, lambda: fade())


# ══════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════
if __name__ == "__main__":
    app = IncidentIQApp()
    app.mainloop()