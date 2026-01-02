# ============================================================
# HackClubAI — PySide6 Full Single-File App (CLEAN)
# Text + Image + Web Search + Settings + Galaxy Theme + App Icon
#
# ✅ Windows-friendly modern UI
# ✅ Saves keys/config next to EXE
# ✅ Text + Image generation via Hack Club AI proxy
# ✅ Web/News search grounding via Hack Club Search (optional)
# ✅ App icon (window + taskbar)
# ✅ No indentation traps / no partial-paste required
#
# REQUIREMENTS (install once):
#   python -m pip install pyside6 requests openrouter pyinstaller
#
# RUN:
#   python OpenRouter.py
#
# ASSETS:
#   assets/app.ico                 (recommended)
#   assets/app.png                 (optional fallback)
#   assets/icons/default.png       (required)
#   assets/icons/openai.png, qwen.png, deepseek.png, mistral.png, llama.png,
#   assets/icons/moonshot.png, gemini.png, grok.png, nvidia.png, zai.png (optional)
#
# BUILD EXE (PowerShell):
#   python -m PyInstaller --noconfirm --clean --onefile --windowed --name HackClubAI --icon assets\app.ico --add-data "assets;assets" OpenRouter.py
# ============================================================

import os
import sys
import json
import base64
import datetime
from typing import Optional, Dict, Tuple

import requests
from openrouter import OpenRouter

from PySide6.QtCore import Qt, QThread, Signal, QSize
from PySide6.QtGui import QIcon, QPixmap, QFont, QAction
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton,
    QPlainTextEdit, QTabWidget, QListWidget, QListWidgetItem, QSplitter,
    QFormLayout, QLineEdit, QDialog, QDialogButtonBox, QComboBox, QSpinBox, QDoubleSpinBox,
    QCheckBox, QFileDialog, QMessageBox, QGroupBox
)

# -------------------- Paths / Storage --------------------


def app_dir() -> str:
    """Store config/keys next to EXE (frozen) or script (dev)."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def resource_path(relative: str) -> str:
    """PyInstaller resource helper."""
    base = getattr(sys, "_MEIPASS", os.path.abspath("."))
    return os.path.join(base, relative)


KEYS_FILE = os.path.join(app_dir(), "keys.json")
CONFIG_FILE = os.path.join(app_dir(), "config.json")

AI_BASE_URL = "https://ai.hackclub.com/proxy/v1"
CHAT_COMPLETIONS_URL = f"{AI_BASE_URL}/chat/completions"

ICON_DIR = resource_path(os.path.join("assets", "icons"))
DEFAULT_ICON = "default.png"

APP_ICON_ICO = resource_path(os.path.join("assets", "app.ico"))
APP_ICON_PNG = resource_path(os.path.join("assets", "app.png"))


def set_app_icon(window: QMainWindow) -> None:
    if os.path.exists(APP_ICON_ICO):
        window.setWindowIcon(QIcon(APP_ICON_ICO))
        return
    if os.path.exists(APP_ICON_PNG):
        window.setWindowIcon(QIcon(APP_ICON_PNG))


# -------------------- Models --------------------

TEXT_MODEL_CATALOG = {
    "qwen/qwen3-32b": {"name": "Qwen 3 32B", "ctx": "41K", "best_for": "Fast daily driver for chat + coding."},
    "moonshotai/kimi-k2-thinking": {"name": "Kimi K2 Thinking", "ctx": "262K", "best_for": "Long-context reasoning + big documents."},
    "openai/gpt-oss-120b": {"name": "gpt-oss-120b", "ctx": "131K", "best_for": "Heavy duty hard prompts (slower)."},
    "moonshotai/kimi-k2-0905": {"name": "Kimi K2 0905", "ctx": "262K", "best_for": "Long-context general model."},
    "google/gemini-3-flash-preview": {"name": "Gemini 3 Flash Preview", "ctx": "1.0M", "best_for": "Huge context + fast."},
    "qwen/qwen3-next-80b-a3b-instruct": {"name": "Qwen 3 Next 80B", "ctx": "262K", "best_for": "Stronger reasoning / complex coding."},
    "qwen/qwen3-vl-235b-a22b-instruct": {"name": "Qwen 3 VL 235B", "ctx": "262K", "best_for": "Vision-language (if supported)."},
    "z-ai/glm-4.7": {"name": "GLM 4.7", "ctx": "203K", "best_for": "Strong general reasoning."},
    "deepseek/deepseek-v3.2-speciale": {"name": "DeepSeek V3.2 Speciale", "ctx": "164K", "best_for": "Strong technical/code tasks."},
    "deepseek/deepseek-v3.2": {"name": "DeepSeek V3.2", "ctx": "164K", "best_for": "Good balance."},
    "x-ai/grok-4.1-fast": {"name": "Grok 4.1 Fast", "ctx": "2.0M", "best_for": "Massive context, fast."},
    "nvidia/nemotron-nano-12b-v2-vl": {"name": "Nemotron Nano 12B VL", "ctx": "131K", "best_for": "Smaller/faster; decent."},
    "google/gemini-2.5-flash": {"name": "Gemini 2.5 Flash", "ctx": "1.0M", "best_for": "Huge context + speed."},
    "openai/gpt-5.1": {"name": "GPT-5.1", "ctx": "400K", "best_for": "Higher quality reasoning/writing."},
    "openai/gpt-5-mini": {"name": "GPT-5 Mini", "ctx": "400K", "best_for": "Fast general Q&A."},
    "deepseek/deepseek-v3.2-exp": {"name": "DeepSeek V3.2 Exp", "ctx": "164K", "best_for": "Experimental."},
    "deepseek/deepseek-r1-0528": {"name": "DeepSeek R1 0528", "ctx": "164K", "best_for": "Reasoning-focused."},
    "z-ai/glm-4.6": {"name": "GLM 4.6", "ctx": "203K", "best_for": "Older but solid."},
    "qwen/qwen3-235b-a22b": {"name": "Qwen 3 235B", "ctx": "41K", "best_for": "Very large; tough prompts."},
    "deepseek/deepseek-r1-distill-qwen-32b": {"name": "DeepSeek R1 Distill Qwen 32B", "ctx": "131K", "best_for": "Reasoning distill."},
    "google/gemini-3-pro-preview": {"name": "Gemini 3 Pro Preview", "ctx": "1.0M", "best_for": "Huge context + high quality."},
    "google/gemini-2.5-flash-lite-preview-09-2025": {"name": "Gemini 2.5 Flash Lite Preview", "ctx": "1.0M", "best_for": "Cheaper/faster long context."},
}

IMAGE_MODEL_CATALOG = {
    "google/gemini-2.5-flash-image": {"name": "Gemini 2.5 Flash Image", "ctx": "33K", "best_for": "Fast image iterations."},
    "google/gemini-3-pro-image-preview": {"name": "Gemini 3 Pro Image Preview", "ctx": "66K", "best_for": "Higher quality, slower."},
}

ASPECT_RATIOS = ["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"]
NEWS_FRESHNESS = ["pd", "pw", "pm", "py"]


def auto_icon_for_model_id(model_id: str) -> str:
    mid = model_id.lower()
    if "openai/" in mid:
        return "openai.png"
    if "qwen/" in mid:
        return "qwen.png"
    if "deepseek/" in mid:
        return "deepseek.png"
    if "mistral" in mid:
        return "mistral.png"
    if "llama" in mid or "meta-llama" in mid:
        return "llama.png"
    if "moonshot" in mid or "kimi" in mid:
        return "moonshot.png"
    if "google/" in mid or "gemini" in mid:
        return "gemini.png"
    if "x-ai/" in mid or "grok" in mid:
        return "grok.png"
    if "nvidia/" in mid or "nemotron" in mid:
        return "nvidia.png"
    if "z-ai/" in mid or "glm" in mid:
        return "zai.png"
    return DEFAULT_ICON


def icon_for(model_id: str) -> QIcon:
    path = os.path.join(ICON_DIR, auto_icon_for_model_id(model_id))
    if not os.path.exists(path):
        path = os.path.join(ICON_DIR, DEFAULT_ICON)
    return QIcon(path)


# -------------------- JSON helpers --------------------


def load_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def save_json(path: str, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_keys() -> Tuple[Optional[str], Optional[str]]:
    d = load_json(KEYS_FILE)
    return d.get("ai_key"), d.get("search_key")


def save_keys(ai_key: str, search_key: Optional[str]) -> None:
    save_json(KEYS_FILE, {"ai_key": ai_key, "search_key": search_key})


def default_config() -> dict:
    return {
        "theme": "dark",  # "dark" or "light"
        "system_prompt": "You are a helpful assistant. Be accurate. If unsure, say so.",
        "default_text_model": list(TEXT_MODEL_CATALOG.keys())[0],
        "default_temp": 0.7,
        "default_image_model": list(IMAGE_MODEL_CATALOG.keys())[0],
        "default_aspect": "1:1",
        "default_n": 1,
        "use_search": False,
        "search_type": "web",        # "web" or "news"
        "search_count": 5,
        "news_freshness": "pw",      # pd/pw/pm/py
    }


def load_config() -> dict:
    cfg = default_config()
    cfg.update(load_json(CONFIG_FILE))

    if cfg.get("theme") not in ("dark", "light"):
        cfg["theme"] = "dark"

    if cfg.get("default_text_model") not in TEXT_MODEL_CATALOG:
        cfg["default_text_model"] = list(TEXT_MODEL_CATALOG.keys())[0]

    if cfg.get("default_image_model") not in IMAGE_MODEL_CATALOG:
        cfg["default_image_model"] = list(IMAGE_MODEL_CATALOG.keys())[0]

    if cfg.get("default_aspect") not in ASPECT_RATIOS:
        cfg["default_aspect"] = "1:1"

    try:
        cfg["default_temp"] = float(cfg.get("default_temp", 0.7))
    except Exception:
        cfg["default_temp"] = 0.7
    cfg["default_temp"] = max(0.0, min(2.0, cfg["default_temp"]))

    try:
        cfg["default_n"] = int(cfg.get("default_n", 1))
    except Exception:
        cfg["default_n"] = 1
    cfg["default_n"] = max(1, min(4, cfg["default_n"]))

    cfg["system_prompt"] = str(cfg.get("system_prompt", "")).strip() or default_config()["system_prompt"]

    cfg["use_search"] = bool(cfg.get("use_search", False))
    if cfg.get("search_type") not in ("web", "news"):
        cfg["search_type"] = "web"
    try:
        cfg["search_count"] = int(cfg.get("search_count", 5))
    except Exception:
        cfg["search_count"] = 5
    cfg["search_count"] = max(1, min(10, cfg["search_count"]))
    if cfg.get("news_freshness") not in NEWS_FRESHNESS:
        cfg["news_freshness"] = "pw"

    return cfg


def save_config(cfg: dict) -> None:
    save_json(CONFIG_FILE, cfg)


# -------------------- Search API --------------------


def search_hackclub(query: str, search_key: str, kind: str = "web", count: int = 5, freshness: str = "pw") -> dict:
    base = "https://search.hackclub.com/res/v1"
    if kind == "news":
        url = f"{base}/news/search"
        params = {"q": query, "count": count, "freshness": freshness}
    else:
        url = f"{base}/web/search"
        params = {"q": query, "count": count}

    r = requests.get(url, params=params, headers={"Authorization": f"Bearer {search_key}"}, timeout=30)
    r.raise_for_status()
    return r.json()


def format_search_context(results: dict, kind: str = "web") -> str:
    bucket = results.get("news", {}) if kind == "news" else results.get("web", {})
    items = bucket.get("results", []) or []
    lines = []
    for i, r in enumerate(items, start=1):
        title = r.get("title", "Untitled")
        url = r.get("url", "")
        desc = r.get("description", "") or r.get("snippet", "") or ""
        lines.append(f"{i}. {title}\n{url}\n{desc}".strip())
    return "\n\n".join(lines).strip()


# -------------------- Worker Thread --------------------

class RunWorker(QThread):
    done_text = Signal(str, str, str)          # header, text, search_context
    done_image = Signal(str, str, str)         # header, assistant_text, image_path
    error = Signal(str)
    status = Signal(str)

    def __init__(
        self,
        ai_key: str,
        search_key: Optional[str],
        mode: str,
        prompt: str,
        cfg: dict,
        text_model: str,
        temp: float,
        image_model: str,
        aspect: str,
        n: int,
    ):
        super().__init__()
        self.ai_key = ai_key
        self.search_key = search_key
        self.mode = mode
        self.prompt = prompt
        self.cfg = cfg
        self.text_model = text_model
        self.temp = temp
        self.image_model = image_model
        self.aspect = aspect
        self.n = n

    def run(self):
        try:
            if self.mode == "text":
                self.status.emit("Searching (optional)…")
                search_context = ""
                if self.cfg.get("use_search") and self.search_key:
                    try:
                        kind = self.cfg.get("search_type", "web")
                        count = int(self.cfg.get("search_count", 5))
                        fresh = self.cfg.get("news_freshness", "pw")
                        results = search_hackclub(self.prompt, self.search_key, kind=kind, count=count, freshness=fresh)
                        search_context = format_search_context(results, kind=kind)
                    except Exception:
                        search_context = ""

                self.status.emit("Calling text model…")
                client = OpenRouter(api_key=self.ai_key, server_url=AI_BASE_URL)

                messages = [{"role": "system", "content": self.cfg["system_prompt"]}]
                if search_context:
                    messages.append({
                        "role": "system",
                        "content": (
                            "Use these search results to answer accurately. "
                            "If you use a fact, cite the URL.\n\nSEARCH RESULTS:\n" + search_context
                        )
                    })
                messages.append({"role": "user", "content": self.prompt})

                resp = client.chat.send(model=self.text_model, messages=messages, temperature=self.temp, stream=False)
                out = resp.choices[0].message.content
                header = f"[{TEXT_MODEL_CATALOG[self.text_model]['name']} | temp={self.temp:.2f}]"
                self.done_text.emit(header, out, search_context)

            else:
                self.status.emit("Generating image…")
                headers = {"Authorization": f"Bearer {self.ai_key}", "Content-Type": "application/json"}
                payload = {
                    "model": self.image_model,
                    "messages": [{"role": "user", "content": self.prompt}],
                    "modalities": ["image", "text"],
                    "image_config": {"aspect_ratio": self.aspect},
                    "stream": False,
                    "n": self.n,
                }
                r = requests.post(CHAT_COMPLETIONS_URL, headers=headers, json=payload, timeout=180)
                r.raise_for_status()
                data = r.json()

                if not data.get("choices"):
                    raise RuntimeError("No choices returned")

                msg = data["choices"][0].get("message", {})
                assistant_text = (msg.get("content") or "").strip()

                imgs = msg.get("images") or []
                if not imgs:
                    raise RuntimeError("No images returned")

                url = (imgs[0].get("image_url") or {}).get("url", "")
                if not url:
                    raise RuntimeError("Image URL missing")

                b64 = url.split(",", 1)[1] if "," in url else url
                image_bytes = base64.b64decode(b64)

                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                out_path = os.path.join(app_dir(), f"generated_{ts}.png")
                with open(out_path, "wb") as f:
                    f.write(image_bytes)

                header = f"[{IMAGE_MODEL_CATALOG[self.image_model]['name']} | aspect={self.aspect} | n={self.n}]"
                self.done_image.emit(header, assistant_text, out_path)

        except Exception as e:
            self.error.emit(str(e))


# -------------------- Settings Dialog --------------------

class SettingsDialog(QDialog):
    def __init__(self, parent: QWidget, cfg: dict, ai_key: Optional[str], search_key: Optional[str]):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setModal(True)
        self.cfg = dict(cfg)

        layout = QVBoxLayout(self)

        form = QFormLayout()

        self.ai_edit = QLineEdit(ai_key or "")
        self.ai_edit.setEchoMode(QLineEdit.Password)
        form.addRow("Hack Club AI Key:", self.ai_edit)

        self.search_edit = QLineEdit(search_key or "")
        self.search_edit.setEchoMode(QLineEdit.Password)
        form.addRow("Search Key (optional):", self.search_edit)

        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["dark", "light"])
        self.theme_combo.setCurrentText(self.cfg.get("theme", "dark"))
        form.addRow("Theme:", self.theme_combo)

        self.sys_prompt = QPlainTextEdit(self.cfg.get("system_prompt", ""))
        self.sys_prompt.setMinimumHeight(110)
        form.addRow("System Prompt:", self.sys_prompt)

        layout.addLayout(form)

        # --- Web Search Settings ---
        self.use_search = QCheckBox("Use web/news search to ground TEXT answers")
        self.use_search.setChecked(bool(self.cfg.get("use_search", False)))
        layout.addWidget(self.use_search)

        row = QHBoxLayout()

        self.search_type = QComboBox()
        self.search_type.addItems(["web", "news"])
        self.search_type.setCurrentText(self.cfg.get("search_type", "web"))
        row.addWidget(QLabel("Type:"))
        row.addWidget(self.search_type)

        self.search_count = QSpinBox()
        self.search_count.setRange(1, 10)
        self.search_count.setValue(int(self.cfg.get("search_count", 5)))
        row.addWidget(QLabel("Count:"))
        row.addWidget(self.search_count)

        self.news_fresh = QComboBox()
        self.news_fresh.addItems(NEWS_FRESHNESS)
        self.news_fresh.setCurrentText(self.cfg.get("news_freshness", "pw"))
        row.addWidget(QLabel("Freshness:"))
        row.addWidget(self.news_fresh)

        layout.addLayout(row)

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_values(self) -> Tuple[dict, str, Optional[str]]:
        cfg = dict(self.cfg)
        cfg["theme"] = self.theme_combo.currentText()
        cfg["system_prompt"] = self.sys_prompt.toPlainText().strip() or default_config()["system_prompt"]

        cfg["use_search"] = self.use_search.isChecked()
        cfg["search_type"] = self.search_type.currentText()
        cfg["search_count"] = int(self.search_count.value())
        cfg["news_freshness"] = self.news_fresh.currentText()

        ai_key = self.ai_edit.text().strip()
        search_key = self.search_edit.text().strip() or None
        return cfg, ai_key, search_key


# -------------------- Main Window --------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HackClubAI")
        self.resize(1320, 900)
        set_app_icon(self)

        self.cfg = load_config()
        self.ai_key, self.search_key = load_keys()

        self.worker: Optional[RunWorker] = None
        self.last_image_path: Optional[str] = None

        self._apply_theme()

        # Menu
        menu = self.menuBar()
        file_menu = menu.addMenu("&File")
        act_settings = QAction("Settings", self)
        act_settings.triggered.connect(self.open_settings)
        file_menu.addAction(act_settings)
        file_menu.addSeparator()
        act_quit = QAction("Quit", self)
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)

        # Root
        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(10)

        # Top bar
        top = QHBoxLayout()
        title = QLabel("HackClubAI")
        title.setFont(QFont("Segoe UI", 18, QFont.Bold))
        top.addWidget(title)

        self.status = QLabel("Ready")
        top.addStretch(1)
        top.addWidget(self.status)

        self.btn_run = QPushButton("Run")
        self.btn_run.clicked.connect(self.on_run)
        top.addWidget(self.btn_run)

        root_layout.addLayout(top)

        # Splitter
        split = QSplitter(Qt.Horizontal)
        root_layout.addWidget(split, 1)

        # Left panel (tabs/models/settings)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.tabs = QTabWidget()
        left_layout.addWidget(self.tabs, 1)

        # Text tab
        self.text_tab = QWidget()
        t_layout = QVBoxLayout(self.text_tab)

        self.text_list = QListWidget()
        self.text_list.currentItemChanged.connect(self.on_text_model_change)
        t_layout.addWidget(self.text_list, 1)

        self.text_detail = QLabel("Select a model")
        self.text_detail.setWordWrap(True)
        self.text_detail.setMinimumHeight(90)
        t_layout.addWidget(self.text_detail)

        temp_box = QGroupBox("Temperature")
        tbx = QVBoxLayout(temp_box)

        self.temp_spin = QDoubleSpinBox()
        self.temp_spin.setRange(0.0, 2.0)
        self.temp_spin.setSingleStep(0.1)
        self.temp_spin.setValue(self.cfg["default_temp"])
        tbx.addWidget(self.temp_spin)

        self.temp_help = QLabel("Low = consistent/factual. High = creative/varied (more mistakes).")
        self.temp_help.setWordWrap(True)
        tbx.addWidget(self.temp_help)
        t_layout.addWidget(temp_box)

        self.tabs.addTab(self.text_tab, "Text")

        # Image tab
        self.image_tab = QWidget()
        i_layout = QVBoxLayout(self.image_tab)

        self.image_list = QListWidget()
        self.image_list.currentItemChanged.connect(self.on_image_model_change)
        i_layout.addWidget(self.image_list, 1)

        self.image_detail = QLabel("Select an image model")
        self.image_detail.setWordWrap(True)
        self.image_detail.setMinimumHeight(90)
        i_layout.addWidget(self.image_detail)

        img_settings = QGroupBox("Image Settings")
        il = QFormLayout(img_settings)

        self.aspect_combo = QComboBox()
        self.aspect_combo.addItems(ASPECT_RATIOS)
        self.aspect_combo.setCurrentText(self.cfg["default_aspect"])
        il.addRow("Aspect:", self.aspect_combo)

        self.n_spin = QSpinBox()
        self.n_spin.setRange(1, 4)
        self.n_spin.setValue(self.cfg["default_n"])
        il.addRow("Count:", self.n_spin)

        i_layout.addWidget(img_settings)
        self.tabs.addTab(self.image_tab, "Image")

        split.addWidget(left)

        # Right panel (prompt/output/search/preview)
        right = QWidget()
        r_layout = QVBoxLayout(right)
        r_layout.setContentsMargins(0, 0, 0, 0)
        r_layout.setSpacing(10)

        r_layout.addWidget(QLabel("Prompt"))
        self.prompt = QPlainTextEdit()
        self.prompt.setPlaceholderText("Type your prompt here…")
        self.prompt.setMinimumHeight(120)
        r_layout.addWidget(self.prompt)

        r_layout.addWidget(QLabel("Output"))
        self.output = QPlainTextEdit()
        self.output.setReadOnly(True)
        r_layout.addWidget(self.output, 1)

        r_layout.addWidget(QLabel("Search results (if enabled)"))
        self.search_out = QPlainTextEdit()
        self.search_out.setReadOnly(True)
        self.search_out.setMaximumHeight(160)
        self.search_out.setPlainText("")
        r_layout.addWidget(self.search_out)

        self.preview = QLabel("(no image yet)")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumHeight(220)
        r_layout.addWidget(self.preview)

        btn_row = QHBoxLayout()
        self.btn_open_image = QPushButton("Open last image")
        self.btn_open_image.clicked.connect(self.open_last_image)
        btn_row.addWidget(self.btn_open_image)

        self.btn_save_output = QPushButton("Save output…")
        self.btn_save_output.clicked.connect(self.save_output)
        btn_row.addWidget(self.btn_save_output)

        btn_row.addStretch(1)
        r_layout.addLayout(btn_row)

        split.addWidget(right)
        split.setSizes([440, 880])

        self._populate_lists()

        # Ensure key on first run
        if not self.ai_key:
            QMessageBox.information(self, "API Key Required", "Enter your Hack Club AI key. It will be saved locally.")
            self.open_settings(force=True)

    # ---------- Theme ----------
    def _apply_theme(self):
        if self.cfg.get("theme") == "light":
            self.setStyleSheet("""
                QWidget {
                    font-family: Segoe UI;
                    font-size: 11px;
                    color: #1f2937;
                }
                QMainWindow {
                    background: #e9ecf3;
                }
                QPlainTextEdit {
                    background: #ffffff;
                    border: 1px solid #c7cde0;
                    border-radius: 14px;
                    padding: 12px;
                }
                QListWidget {
                    background: #ffffff;
                    border: 1px solid #c7cde0;
                    border-radius: 14px;
                    padding: 8px;
                }
                QLabel {
                    color: #1f2937;
                }
                QPushButton {
                    background: qlineargradient(
                        x1:0, y1:0, x2:0, y2:1,
                        stop:0 #3b82f6,
                        stop:1 #2563eb
                    );
                    color: white;
                    border: none;
                    border-radius: 14px;
                    padding: 10px 16px;
                    font-weight: 600;
                }
                QPushButton:hover { background: #1d4ed8; }
                QPushButton:disabled { background: #a5b4fc; color: #eef2ff; }
                QTabWidget::pane { border: none; }
                QTabBar::tab {
                    padding: 10px 16px;
                    border-radius: 14px;
                    margin-right: 6px;
                    background: #dde3f3;
                    color: #1f2937;
                }
                QTabBar::tab:selected { background: #c7d2fe; }
            """)
        else:
            # Galaxy-ish background + luminous UI
            self.setStyleSheet("""
                QWidget {
                    font-family: Segoe UI;
                    font-size: 11px;
                    color: #eaeaf0;
                }
                QMainWindow {
                    background: qlineargradient(
                        x1:0, y1:0, x2:1, y2:1,
                        stop:0 #070a12,
                        stop:0.35 #101633,
                        stop:0.7 #1c1f4a,
                        stop:1 #070a12
                    );
                }
                QPlainTextEdit {
                    background: rgba(16, 20, 35, 0.88);
                    color: #eaeaf0;
                    border: 1px solid rgba(99, 102, 241, 0.40);
                    border-radius: 14px;
                    padding: 12px;
                }
                QListWidget {
                    background: rgba(16, 20, 35, 0.88);
                    color: #eaeaf0;
                    border: 1px solid rgba(99, 102, 241, 0.40);
                    border-radius: 14px;
                    padding: 8px;
                }
                QLabel { color: #eaeaf0; }
                QPushButton {
                    background: qlineargradient(
                        x1:0, y1:0, x2:1, y2:0,
                        stop:0 #7c3aed,
                        stop:0.5 #6366f1,
                        stop:1 #3b82f6
                    );
                    color: white;
                    border: none;
                    border-radius: 14px;
                    padding: 10px 16px;
                    font-weight: 700;
                }
                QPushButton:hover { background: #4f46e5; }
                QPushButton:disabled { background: #1e293b; color: #94a3b8; }
                QTabWidget::pane { border: none; }
                QTabBar::tab {
                    padding: 10px 16px;
                    border-radius: 14px;
                    margin-right: 6px;
                    background: rgba(30, 41, 59, 0.78);
                    color: #dbeafe;
                }
                QTabBar::tab:selected { background: rgba(99, 102, 241, 0.62); }
            """)

    # ---------- Populate ----------
    def _populate_lists(self):
        self.text_list.clear()
        self.image_list.clear()

        for mid, info in TEXT_MODEL_CATALOG.items():
            item = QListWidgetItem(icon_for(mid), f"{info['name']}  •  {info['ctx']}")
            item.setData(Qt.UserRole, mid)
            self.text_list.addItem(item)

        for mid, info in IMAGE_MODEL_CATALOG.items():
            item = QListWidgetItem(icon_for(mid), f"{info['name']}  •  {info['ctx']}")
            item.setData(Qt.UserRole, mid)
            self.image_list.addItem(item)

        self._select_list_by_model(self.text_list, self.cfg["default_text_model"])
        self._select_list_by_model(self.image_list, self.cfg["default_image_model"])

    def _select_list_by_model(self, lst: QListWidget, model_id: str):
        for i in range(lst.count()):
            it = lst.item(i)
            if it.data(Qt.UserRole) == model_id:
                lst.setCurrentRow(i)
                return
        if lst.count() > 0:
            lst.setCurrentRow(0)

    def on_text_model_change(self, current: QListWidgetItem, _prev: QListWidgetItem):
        if not current:
            return
        mid = current.data(Qt.UserRole)
        info = TEXT_MODEL_CATALOG[mid]
        self.text_detail.setText(
            f"{info['name']}\nModel: {mid}\nContext: {info['ctx']}\n\nBest for: {info['best_for']}"
        )

    def on_image_model_change(self, current: QListWidgetItem, _prev: QListWidgetItem):
        if not current:
            return
        mid = current.data(Qt.UserRole)
        info = IMAGE_MODEL_CATALOG[mid]
        self.image_detail.setText(
            f"{info['name']}\nModel: {mid}\nContext: {info['ctx']}\n\nBest for: {info['best_for']}"
        )

    # ---------- Settings ----------
    def open_settings(self, force: bool = False):
        dlg = SettingsDialog(self, self.cfg, self.ai_key, self.search_key)
        res = dlg.exec()
        if res != QDialog.Accepted:
            if force and not self.ai_key:
                QMessageBox.critical(self, "Missing Key", "You must set an AI key to use the app.")
            return

        cfg, ai_key, search_key = dlg.get_values()
        if not ai_key:
            QMessageBox.critical(self, "Missing Key", "AI key is required.")
            return

        self.cfg = cfg
        self.ai_key = ai_key
        self.search_key = search_key

        save_config(self.cfg)
        save_keys(self.ai_key, self.search_key)

        self._apply_theme()
        self.status.setText("Settings saved")

    # ---------- Run ----------
    def on_run(self):
        prompt = self.prompt.toPlainText().strip()
        if not prompt:
            QMessageBox.information(self, "Prompt required", "Type a prompt first.")
            return
        if not self.ai_key:
            QMessageBox.critical(self, "Missing key", "Set your AI key in Settings.")
            return
        if self.worker and self.worker.isRunning():
            return

        mode = "text" if self.tabs.currentIndex() == 0 else "image"

        text_item = self.text_list.currentItem()
        text_model = text_item.data(Qt.UserRole) if text_item else self.cfg["default_text_model"]
        temp = float(self.temp_spin.value())

        img_item = self.image_list.currentItem()
        image_model = img_item.data(Qt.UserRole) if img_item else self.cfg["default_image_model"]
        aspect = self.aspect_combo.currentText()
        n = int(self.n_spin.value())

        self.btn_run.setEnabled(False)
        self.status.setText("Working…")

        self.worker = RunWorker(
            ai_key=self.ai_key,
            search_key=self.search_key,
            mode=mode,
            prompt=prompt,
            cfg=self.cfg,
            text_model=text_model,
            temp=temp,
            image_model=image_model,
            aspect=aspect,
            n=n,
        )
        self.worker.status.connect(self.status.setText)
        self.worker.done_text.connect(self._on_done_text)
        self.worker.done_image.connect(self._on_done_image)
        self.worker.error.connect(self._on_error)
        self.worker.finished.connect(lambda: self.btn_run.setEnabled(True))
        self.worker.start()

    def _on_done_text(self, header: str, text: str, search_context: str):
        self.output.appendPlainText(header)
        self.output.appendPlainText(text.strip())
        self.output.appendPlainText("")
        self.search_out.setPlainText(search_context or "(no search context)")
        self.status.setText("Ready")

    def _on_done_image(self, header: str, assistant_text: str, image_path: str):
        self.output.appendPlainText(header)
        if assistant_text:
            self.output.appendPlainText(assistant_text)
        self.output.appendPlainText(f"Saved: {image_path}")
        self.output.appendPlainText("")
        self.search_out.setPlainText("(search is only used for TEXT mode)")
        self.last_image_path = image_path
        self._set_preview(image_path)
        self.status.setText("Ready")

    def _on_error(self, msg: str):
        self.output.appendPlainText(f"Error: {msg}\n")
        self.status.setText("Error")
        QMessageBox.warning(self, "Request failed", msg)

    # ---------- Preview / Save ----------
    def _set_preview(self, path: str):
        if not os.path.exists(path):
            self.preview.setText("(image missing)")
            return
        pix = QPixmap(path)
        if pix.isNull():
            self.preview.setText("(preview failed)")
            return
        scaled = pix.scaled(QSize(560, 360), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.preview.setPixmap(scaled)

    def open_last_image(self):
        if not self.last_image_path:
            QMessageBox.information(self, "No image", "No image generated yet.")
            return
        if sys.platform.startswith("win"):
            os.startfile(self.last_image_path)  # type: ignore[attr-defined]
        else:
            os.system(f'xdg-open "{self.last_image_path}"')

    def save_output(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save output", app_dir(), "Text Files (*.txt)")
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.output.toPlainText())
        self.status.setText(f"Saved: {path}")


# -------------------- Main --------------------

def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
