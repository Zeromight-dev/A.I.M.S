"""
AIMS HMI - Streamlit Single-File Human-Machine Interface

Implements the AIMS specification as a Streamlit app:
- State A: ACTIVE_RUN (live monitoring, metrics, tracking log, alarm)
- State B: SESSION_REPORT (executive metrics, defect charts, export tools)

Hardware integration is simulated via mock serial (TRIG/ERR_MOTOR and
PC->Arduino commands 0/1/STOP/START) and a synthetic OpenCV camera feed
so the HMI is fully runnable without a real camera or Arduino.

Run:
    .venv/bin/streamlit run aims_hmi/hmi_streamlit.py --server.headless true
"""

from __future__ import annotations

import time
import io
import os
import threading
import random
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Optional, Dict, List, Tuple

import numpy as np
import cv2
import pandas as pd

import streamlit as st

# =============================================================================
# Project-local imports (classic package layout in this workspace)
# =============================================================================
from aims_hmi.config import (
    COLORS,
    FONTS,
    LAYOUT,
    SERIAL,
    VISION,
    ALARM,
    SESSION,
)
from aims_hmi.exporters import CSVExporter, PDFExporter


# =============================================================================
# App constants (mirrors spec where useful)
# =============================================================================
COMPONENT_LABEL = "Bottle Caps"
ABOUT_URL = "https://your-project-link.com"

METRIC_CSS = """
<style>
    /* dark theme + metric cards + alarm banner */
    .aims-main { background-color: #1E1E2E; color: #ffffff; }
    .metric-card {
        background-color: #252538;
        border: 1px solid #3D3D56;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
        height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .metric-label { font-size: 11px; color: #A0A0B0; text-transform: uppercase; letter-spacing: 1px; }
    .metric-value { font-size: 26px; font-weight: bold; color: #ffffff; }
    .metric-value.pass { color: #2ECC71; }
    .metric-value.fail { color: #E74C3C; }
    .metric-value.warn { color: #F1C40F; }
    .alarm-banner {
        background-color: #2A0A0A;
        color: #FF4444;
        border: 2px solid #FF4444;
        border-radius: 6px;
        padding: 10px 14px;
        font-weight: bold;
        font-size: 13px;
        margin-bottom: 12px;
    }
    .status-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: bold;
        font-size: 12px;
    }
    .status-connected { background-color: #2ECC71; color: #000000; }
    .status-disconnected { background-color: #E74C3C; color: #ffffff; }
    .btn-start { background-color: #2ECC71; color: #000000; border: 1px solid #2ECC71; }
    .btn-stop { background-color: #E74C3C; color: #ffffff; border: 1px solid #E74C3C; }
    .section-title { color: #A0A0B0; font-size: 12px; letter-spacing: 1px; text-transform: uppercase; }
    /* table row tints */
    .pass-row { background-color: #1A3A2A !important; }
    .fail-row { background-color: #3A1A1A !important; }
</style>
"""

# =============================================================================
# Helpers
# =============================================================================

def rgb_to_qphoto_bytes(img_rgb: np.ndarray) -> bytes:
    """Encode an RGB numpy array as PNG bytes for st.image."""
    ok, buf = cv2.imencode(".png", img_rgb)
    if not ok:
        return b""
    return buf.tobytes()


def bgr_to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def color_for_status(status: str) -> str:
    return "#2ECC71" if status == "PASS" else "#E74C3C"


# =============================================================================
# Mock hardware (lightweight in-process simulation)
# =============================================================================

class MockSerialSim:
    """Simulates Arduino serial link with the protocol from spec Section 5."""

    def __init__(self, port: str, baud: int):
        self.port = port
        self.baud = baud
        self.connected = False
        self.mock_mode = True
        self.trigs = 0
        self.errors = 0
        self.commands = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.on_trig = None
        self.on_error = None

    def start(self):
        self.connected = True
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        self.connected = False

    def _run(self):
        while not self._stop.is_set():
            # random TRIGs simulating caps passing IR sensor
            time.sleep(random.uniform(0.6, 2.6))
            if self._stop.is_set():
                break
            self.trigs += 1
            if self.on_trig:
                try:
                    self.on_trig()
                except Exception:
                    pass

    def send(self, payload: bytes):
        self.commands += 1
        # In a real impl this would write to pyserial; here we just log.
        if payload in (b"STOP", b"START", b"0", b"1"):
            pass


class MockCameraSim:
    """Generates synthetic frames with a bottle cap and occasional defects."""

    def __init__(self, width: int = 640, height: int = 480):
        self.width = width
        self.height = height
        self._running = False
        self._frame = None
        self._lock = threading.Lock()
        self._defect_mode = False

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._gen, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if hasattr(self, "_thread"):
            self._thread.join(timeout=1.0)

    def _gen(self):
        cap_x = 0
        while self._running:
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            # conveyor background
            cv2.rectangle(frame, (0, self.height - 70), (self.width, self.height), (60, 60, 60), -1)
            for x in range(0, self.width, 30):
                cv2.line(frame, (x, self.height - 70), (x + 15, self.height), (70, 70, 70), 1)

            cap_x += 4
            if cap_x > self.width + 40:
                cap_x = -40
                self._defect_mode = random.random() < 0.22

            cx = int(cap_x)
            cy = self.height // 2
            r = min(self.width, self.height) // 7

            if self._defect_mode:
                kind = random.choice(["dent", "scratch", "liner", "misalign"])
                cv2.circle(frame, (cx, cy), r, (170, 170, 170), -1)
                if kind == "dent":
                    cv2.circle(frame, (cx + r//3, cy - r//4), r//5, (90, 90, 90), -1)
                elif kind == "scratch":
                    cv2.line(frame, (cx - r//2, cy), (cx + r//3, cy + r//4), (40, 40, 40), 2)
                elif kind == "liner":
                    cv2.circle(frame, (cx, cy), r - 14, (150, 50, 50), -1)
                else:
                    cv2.circle(frame, (cx + r//4, cy), r - 14, (40, 40, 40), -1)
            else:
                cv2.circle(frame, (cx, cy), r, (192, 192, 192), -1)
                cv2.circle(frame, (cx, cy), r - 5, (220, 220, 220), -1)
                cv2.circle(frame, (cx, cy), r - 8, (180, 180, 180), 3)
                cv2.circle(frame, (cx, cy), r - 15, (40, 40, 40), -1)

            noise = np.random.randint(0, 18, (self.height, self.width, 3), dtype=np.uint8)
            frame = cv2.add(frame, noise)

            with self._lock:
                self._frame = frame
            time.sleep(0.016)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self._lock:
            if self._frame is None:
                return False, None
            return True, self._frame.copy()

    def get_golden_template(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._frame is None:
                return None
            # Return a clean perfect cap render from the current frame size
            return self._perfect_cap()

    def _perfect_cap(self) -> np.ndarray:
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        cv2.rectangle(frame, (0, self.height - 70), (self.width, self.height), (60, 60, 60), -1)
        cx, cy = self.width // 2, self.height // 2 - 10
        r = min(self.width, self.height) // 6
        cv2.circle(frame, (cx, cy), r, (192, 192, 192), -1)
        cv2.circle(frame, (cx, cy), r - 5, (220, 220, 220), -1)
        cv2.circle(frame, (cx, cy), r - 8, (180, 180, 180), 3)
        cv2.circle(frame, (cx, cy), r - 15, (40, 40, 40), -1)
        cv2.circle(frame, (cx - r//3, cy - r//3), r//4, (255, 255, 255), -1, cv2.LINE_AA)
        return frame


# =============================================================================
# Mock YOLO inference (mirrors VISION + defect list from config)
# =============================================================================

def mock_inference(frame: np.ndarray, threshold: int) -> Dict:
    """Returns a dict mimicking an ultralytics-style result for a single item."""
    h, w = frame.shape[:2]
    cx = random.randint(w // 4, 3 * w // 4)
    cy = random.randint(h // 4, 3 * h // 4)
    box = (cx - 40, cy - 40, cx + 40, cy + 40)

    # bias toward pass
    is_pass = random.random() > 0.20
    if is_pass:
        conf = random.uniform(0.85, 0.99)
        defect = None
        label = f"BOTTLE_CAP: PASS - {conf:.0%}"
        color = (0, 255, 0)
    else:
        conf = random.uniform(0.60, 0.94)
        defect = random.choice(VISION["defect_types"])
        label = f"BOTTLE_CAP: FAIL [{defect}] - {conf:.0%}"
        color = (0, 0, 255)

    # annotate a copy
    annotated = frame.copy()
    cv2.rectangle(annotated, (box[0], box[1]), (box[2], box[3]), color, 3)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    tx, ty = box[0], max(20, box[1] - 10)
    cv2.rectangle(annotated, (tx, ty - th - 5), (tx + tw + 10, ty + 5), (0, 0, 0), -1)
    cv2.putText(annotated, label, (tx + 5, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    ts = datetime.now().strftime("%H:%M:%S")
    cv2.putText(annotated, f"TIME: {ts}", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    return {
        "is_pass": is_pass,
        "confidence": conf,
        "defect_type": defect,
        "bounding_box": box,
        "label": label,
        "annotation_rgb": bgr_to_rgb(annotated),
        "frame_ts": ts,
    }


# =============================================================================
# Session state initializer
# =============================================================================

def init_session_state():
    defaults: dict = {
        "state": "active",  # "active" | "report"
        "session_running": False,
        "session_start": None,
        "item_counter": 0,
        "pass_count": 0,
        "fail_count": 0,
        "consecutive_fails": 0,
        "alarm_active": False,
        "alarm_msg": "",
        "threshold": VISION["threshold_default"],
        "serial": None,
        "camera": None,
        "log": [],
        "defect_counts": {},
        "defect_timeline": [],
        "pending_trigger_event": threading.Event(),
        "last_inference_result": None,
        "live_frame_rgb": None,
        "fps": 0.0,
        "batch_id": "BATCH-2026-001",
        "operator_id": "Op_Rajan",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# =============================================================================
# UI sections
# =============================================================================

def render_header():
    """Global header bar (spec Section 3.1)."""
    col_branding, col_meta, col_hardware, col_actions = st.columns([2.2, 2.2, 1.6, 2.0])
    with col_branding:
        st.markdown(
            f"<div style='font-family:Arial; font-size:18px; font-weight:bold; color:#ffffff;'>"
            f"AIMS - AI-driven Industrial Manufacturing Sorter</div>",
            unsafe_allow_html=True,
        )
    with col_meta:
        b1, b2, b3 = st.columns([1.1, 1.1, 1.3])
        with b1:
            st.text_input("Batch ID", key="batch_id", value=st.session_state["batch_id"],
                          disabled=st.session_state["session_running"])
        with b2:
            st.text_input("Operator ID", key="operator_id", value=st.session_state["operator_id"],
                          disabled=st.session_state["session_running"])
        with b3:
            st.markdown(
                f"<div class='section-title'>Component</div><div style='color:#ffffff;'>{COMPONENT_LABEL}</div>",
                unsafe_allow_html=True,
            )
    with col_hardware:
        ser = st.session_state.get("serial")
        if ser is None:
            pill = "<div class='status-pill status-disconnected'>SERIAL: DISCONNECTED</div>"
        else:
            if ser.connected:
                info = f"{ser.port} @ {ser.baud}"
                pill = f"<div class='status-pill status-connected'>SERIAL: CONNECTED ({info})</div>"
            else:
                pill = "<div class='status-pill status-disconnected'>SERIAL: DISCONNECTED</div>"
        st.markdown(pill, unsafe_allow_html=True)
    with col_actions:
        st.session_state["operator_id"] = st.session_state.get("operator_id", "Op_Rajan")
        st.session_state["batch_id"] = st.session_state.get("batch_id", "BATCH-2026-001")

        if st.session_state["session_running"]:
            btn = st.button("STOP SESSION", key="btn_toggle", use_container_width=True, type="primary")
        else:
            btn = st.button("START SESSION", key="btn_toggle", use_container_width=True)
        if btn:
            if st.session_state["session_running"]:
                stop_session()
            else:
                start_session()
        if st.button("About", key="btn_about", use_container_width=False):
            import webbrowser
            webbrowser.open_new_tab(ABOUT_URL)


def render_active_run_view():
    """State A: left panel video + AI controls, right panel metrics + log + alarm."""
    st.markdown(METRIC_CSS, unsafe_allow_html=True)
    st.markdown('<div class="aims-main">', unsafe_allow_html=True)

    left, right = st.columns([1.25, 1.0])

    # ---------- LEFT PANEL ----------
    with left:
        st.markdown("<div class='section-title'>Live Vision & AI Controls</div>", unsafe_allow_html=True)

        # Video feed
        st.markdown("<div class='section-title'>Video Feed (OpenCV Stream)</div>", unsafe_allow_html=True)
        frame = st.session_state.get("live_frame_rgb")
        if frame is None:
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            blank[:] = (30, 30, 30)
            cv2.putText(blank, "Awaiting video feed...", (40, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (120, 120, 120), 2)
            frame = bgr_to_rgb(blank)
        st.image(frame, channels="RGB", use_container_width=True)

        # FPS
        fps = st.session_state.get("fps", 0)
        st.markdown(
            f"<div style='text-align:right; color:#2ECC71; font-family:Courier; font-weight:bold;'>"
            f"FPS: {int(fps)}</div>",
            unsafe_allow_html=True,
        )

        # AI threshold slider
        st.markdown("<div class='section-title'>AI Detection Threshold (%)</div>", unsafe_allow_html=True)
        threshold = st.slider(
            "AI Detection Threshold (%)",
            min_value=VISION["threshold_min"],
            max_value=VISION["threshold_max"],
            value=st.session_state["threshold"],
            step=VISION["threshold_step"],
            key="thr_slider",
            help="Minimum confidence passed to the YOLO inference engine.",
        )
        st.session_state["threshold"] = threshold
        st.markdown(f"Selected: **{threshold}%**")

        st.info(
            "Threshold: Minimum confidence required for YOLO classification. "
            "Lower values = more detections (higher false positive rate).",
            icon=":material/info:",
        )

    # ---------- RIGHT PANEL ----------
    with right:
        st.markdown("<div class='section-title'>Telemetry & Tracking Log</div>", unsafe_allow_html=True)

        # Metric cards
        total = st.session_state["item_counter"]
        passed = st.session_state["pass_count"]
        failed = st.session_state["fail_count"]
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(
                f"<div class='metric-card'><div class='metric-label'>Total</div>"
                f"<div class='metric-value'>{total:,}</div></div>",
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f"<div class='metric-card'><div class='metric-label'>Pass</div>"
                f"<div class='metric-value pass'>{passed:,}</div></div>",
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                f"<div class='metric-card'><div class='metric-label'>Fail</div>"
                f"<div class='metric-value fail'>{failed:,}</div></div>",
                unsafe_allow_html=True,
            )

        # Alarm banner + acknowledge
        alarm_active = st.session_state.get("alarm_active", False)
        alarm_msg = st.session_state.get("alarm_msg", "")
        if alarm_active and alarm_msg:
            st.markdown(f"<div class='alarm-banner'>{alarm_msg}</div>", unsafe_allow_html=True)
            if st.button("Acknowledge & Silence Alarm", key="btn_ack", use_container_width=True):
                st.session_state["alarm_active"] = False
                st.session_state["alarm_msg"] = ""
                ack_alarm()
        else:
            st.markdown(
                "<div style='height:32px;'></div>",
                unsafe_allow_html=True,
            )

        # Tracking log
        st.markdown("<div class='section-title'>Tracking Log</div>", unsafe_allow_html=True)
        log = st.session_state.get("log", [])
        if log:
            df = pd.DataFrame(log)
            df_display = df.copy()
            df_display["Timestamp"] = df_display["Timestamp"].dt.strftime("%H:%M:%S.%f").str[:-3]
            # Build row-color mapping
            color_map = {i: "pass-row" if row["Status"] == "PASS" else "fail-row"
                         for i, row in df.iterrows()}
            st.dataframe(
                df_display,
                column_config={
                    "Status": st.column_config.TextColumn(label="Status"),
                    "Confidence": st.column_config.TextColumn(label="Confidence"),
                },
                use_container_width=True,
                hide_index=True,
                selection_mode="single-row",
            )
            # Row click handling note + manual trigger
            st.caption("Click a FAIL row to open Snapshot Review (defective cap vs golden template).")
            if st.button("Review Last FAIL Snapshot", key="btn_review", use_container_width=False):
                open_snapshot_review()
        else:
            st.caption("No items processed yet. Start a session to begin logging.")

    st.markdown("</div>", unsafe_allow_html=True)


def render_report_view():
    """State B: executive metrics, defect pie + timeline line, export actions."""
    st.markdown(METRIC_CSS, unsafe_allow_html=True)
    st.markdown('<div class="aims-main">', unsafe_allow_html=True)

    total = st.session_state["item_counter"]
    passed = st.session_state["pass_count"]
    failed = st.session_state["fail_count"]
    session_start = st.session_state.get("session_start")
    if session_start:
        duration = datetime.now() - session_start
        total_seconds = int(duration.total_seconds())
        h, m, s = total_seconds // 3600, (total_seconds % 3600) // 60, total_seconds % 60
        session_time = f"{h:02d}:{m:02d}:{s:02d}"
    else:
        session_time = "00:00:00"

    yield_rate = (passed / total * 100) if total else 0.0

    st.markdown("<div class='section-title'>End of Session Report</div>", unsafe_allow_html=True)

    # Executive metric cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"<div class='metric-card'><div class='metric-label'>Total Session Time</div>"
            f"<div class='metric-value'>{session_time}</div></div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"<div class='metric-card'><div class='metric-label'>Total Units Processed</div>"
            f"<div class='metric-value'>{total:,} units</div></div>",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"<div class='metric-card'><div class='metric-label'>Line Yield Rate</div>"
            f"<div class='metric-value {"warn" if yield_rate < 90 else ""}'>{yield_rate:.1f}% PASS</div></div>",
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"<div class='metric-card'><div class='metric-label'>Total Rejections</div>"
            f"<div class='metric-value fail'>{failed:,} units</div></div>",
            unsafe_allow_html=True,
        )

    # Charts
    defect_counts = st.session_state.get("defect_counts", {})
    timeline = st.session_state.get("defect_timeline", [])

    c_plot1, c_plot2 = st.columns([1.0, 1.0])

    with c_plot1:
        st.markdown("<div class='section-title'>Defect Distribution Breakdown</div>", unsafe_allow_html=True)
        render_defect_pie(defect_counts)

    with c_plot2:
        st.markdown("<div class='section-title'>Defect Timeline Spike Analysis (5-min intervals)</div>", unsafe_allow_html=True)
        render_defect_timeline(timeline)

    # Action toolbar
    st.markdown("<div class='section-title'>Post-Session Actions</div>", unsafe_allow_html=True)
    a1, a2, a3 = st.columns([1.0, 1.0, 1.4])
    with a1:
        if st.button("Export PDF", key="btn_export_pdf", use_container_width=True):
            export_pdf()
    with a2:
        if st.button("Export CSV", key="btn_export_csv", use_container_width=True):
            export_csv()
    with a3:
        if st.button("Start New Session", key="btn_new_session", use_container_width=True):
            new_session()

    st.caption("Session data retained for export. Click 'Start New Session' to clear buffers and return to live monitoring.")

    st.markdown("</div>", unsafe_allow_html=True)


def render_defect_pie(defect_counts: Dict[str, int]):
    """Draw a matplotlib pie chart of defect breakdown."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not defect_counts or all(v == 0 for v in defect_counts.values()):
        fig, ax = plt.subplots(figsize=(4.2, 3.2), dpi=110)
        ax.pie([1], labels=["No Defects Recorded"], colors=["#6B6B80"], autopct="%1.1f%%")
        ax.set_facecolor("#252538")
        fig.patch.set_facecolor("#252538")
        ax.tick_params(colors="#A0A0B0")
        for sp in ax.spines.values():
            sp.set_color("#3D3D56")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        return

    filtered = {k: v for k, v in defect_counts.items() if v > 0}
    if not filtered:
        fig, ax = plt.subplots(figsize=(4.2, 3.2), dpi=110)
        ax.pie([1], labels=["No Defects Recorded"], colors=["#6B6B80"], autopct="%1.1f%%")
        ax.set_facecolor("#252538")
        fig.patch.set_facecolor("#252538")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        return

    labels = list(filtered.keys())
    values = list(filtered.values())
    colors = ["#E74C3C", "#C0392B", "#E67E22", "#F39C12", "#3498DB", "#2ECC71", "#9B59B6", "#1ABC9C"][:len(values)]

    fig, ax = plt.subplots(figsize=(4.2, 3.2), dpi=110)
    wedges, texts, autotexts = ax.pie(
        values,
        labels=labels,
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
        pctdistance=0.8,
        wedgeprops=dict(edgecolor="#252538"),
    )
    for t in texts:
        t.set_color("#FFFFFF")
        t.set_fontsize(8)
    for at in autotexts:
        at.set_color("#FFFFFF")
        at.set_fontsize(8)
        at.set_weight("bold")
    ax.set_facecolor("#252538")
    fig.patch.set_facecolor("#252538")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render_defect_timeline(timeline: List[Tuple[float, int]]):
    """Draw a matplotlib line chart of defects per 5-minute interval."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not timeline:
        fig, ax = plt.subplots(figsize=(4.2, 3.2), dpi=110)
        ax.plot([], [])
        ax.set_facecolor("#252538")
        fig.patch.set_facecolor("#252538")
        ax.set_title("Defect Spike Timeline", color="#FFFFFF", fontweight="bold")
        ax.set_xlabel("Session Time (minutes)", color="#A0A0B0")
        ax.set_ylabel("Defect Count", color="#A0A0B0")
        ax.grid(True, alpha=0.3, color="#3D3D56")
        for sp in ax.spines.values():
            sp.set_color("#3D3D56")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        return

    aggregated = defaultdict(int)
    for minute, count in timeline:
        interval = int(minute // 5) * 5
        aggregated[interval] += count
    xs = sorted(aggregated.keys())
    ys = [aggregated[x] for x in xs]

    fig, ax = plt.subplots(figsize=(4.2, 3.2), dpi=110)
    ax.plot(xs, ys, color="#E74C3C", linewidth=2, marker="o", markersize=6,
            markerfacecolor="#E74C3C", markeredgecolor="#252538", markeredgewidth=1)
    ax.fill_between(xs, ys, alpha=0.2, color="#E74C3C")
    ax.set_facecolor("#252538")
    fig.patch.set_facecolor("#252538")
    ax.set_title("Defect Spike Timeline", color="#FFFFFF", fontweight="bold")
    ax.set_xlabel("Session Time (minutes)", color="#A0A0B0")
    ax.set_ylabel("Defect Count", color="#A0A0B0")
    ax.grid(True, alpha=0.3, color="#3D3D56")
    for sp in ax.spines.values():
        sp.set_color("#3D3D56")
    if xs:
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{x:.0f}" for x in xs])
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# =============================================================================
# Session control
# =============================================================================

def start_session():
    st.session_state["session_running"] = True
    st.session_state["state"] = "active"
    st.session_state["session_start"] = datetime.now()
    st.session_state["item_counter"] = 0
    st.session_state["pass_count"] = 0
    st.session_state["fail_count"] = 0
    st.session_state["consecutive_fails"] = 0
    st.session_state["alarm_active"] = False
    st.session_state["alarm_msg"] = ""
    st.session_state["log"] = []
    st.session_state["defect_counts"] = {}
    st.session_state["defect_timeline"] = []
    st.session_state["pending_trigger_event"] = threading.Event()
    st.session_state["last_inference_result"] = None

    # start hardware sim if not present
    if st.session_state.get("serial") is None:
        st.session_state["serial"] = MockSerialSim(SERIAL["default_port"], SERIAL["default_baudrate"])
    if st.session_state.get("camera") is None:
        st.session_state["camera"] = MockCameraSim(640, 480)

    serial = st.session_state["serial"]
    serial.start()
    camera = st.session_state["camera"]
    camera.start()

    # wire TRIG callback
    serial.on_trig = on_trig

    # send START to hardware (simulated)
    serial.send(SERIAL["CMD_START"])

    st.toast("Session started", icon=":material/play_circle:")


def stop_session():
    st.session_state["session_running"] = False
    st.session_state["state"] = "report"

    serial = st.session_state.get("serial")
    if serial:
        serial.send(SERIAL["CMD_STOP"])
        serial.stop()
    camera = st.session_state.get("camera")
    if camera:
        camera.stop()

    st.toast("Session stopped - generating report", icon=":material/assignment:")


def new_session():
    st.session_state["session_running"] = False
    st.session_state["state"] = "active"
    st.session_state["session_start"] = None
    st.session_state["item_counter"] = 0
    st.session_state["pass_count"] = 0
    st.session_state["fail_count"] = 0
    st.session_state["consecutive_fails"] = 0
    st.session_state["alarm_active"] = False
    st.session_state["alarm_msg"] = ""
    st.session_state["log"] = []
    st.session_state["defect_counts"] = {}
    st.session_state["defect_timeline"] = []
    st.session_state["pending_trigger_event"] = threading.Event()
    st.session_state["last_inference_result"] = None

    serial = st.session_state.get("serial")
    if serial:
        serial.stop()
        serial.trigs = 0
        serial.errors = 0
        serial.commands = 0
        serial.on_trig = on_trig
        serial.start()
    camera = st.session_state.get("camera")
    if camera:
        camera.stop()
        camera._defect_mode = False
        camera.start()

    st.rerun()


# =============================================================================
# TRIG / inference / alarm logic
# =============================================================================

def on_trig():
    """Callback fired by mock serial on TRIG."""
    if not st.session_state.get("session_running"):
        return
    st.session_state["pending_trigger_event"].set()


def refresh_hardware_frames():
    """Pull latest camera frames for the UI (called each rerun)."""
    camera = st.session_state.get("camera")
    if camera:
        ok, frame = camera.read()
        if ok:
            res = mock_inference(frame, st.session_state["threshold"])
            st.session_state["live_frame_rgb"] = res["annotation_rgb"]
            st.session_state["last_inference_result"] = res
            # crude FPS estimate
            st.session_state["fps"] = st.session_state.get("fps", 0) * 0.9 + 58 * 0.1


def flush_pending_triggers():
    """Process queued TRIGs (without blocking UI) using a short background job."""
    ev = st.session_state.get("pending_trigger_event")
    if ev is None:
        return
    processed = 0
    while ev.is_set():
        ev.clear()
        processed += 1
        if not st.session_state.get("session_running"):
            break
        # increment item counter
        st.session_state["item_counter"] += 1
        item_id = f"ITEM-{st.session_state['item_counter']:05d}"
        frame = None
        camera = st.session_state.get("camera")
        if camera:
            ok, frame = camera.read()
        if not ok:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # run inference now
        res = mock_inference(frame, st.session_state["threshold"])
        st.session_state["last_inference_result"] = res
        record_result(item_id, res)


def record_result(item_id: str, res: Dict):
    ts = datetime.now()
    is_pass = res["is_pass"]
    conf = res["confidence"]
    defect = res.get("defect_type")

    if is_pass:
        status = "PASS"
        action = "ACCEPT"
        st.session_state["pass_count"] += 1
        st.session_state["consecutive_fails"] = 0
        if st.session_state.get("serial"):
            st.session_state["serial"].send(SERIAL["CMD_PASS"])
    else:
        status = "FAIL"
        action = "REJECT"
        st.session_state["fail_count"] += 1
        st.session_state["consecutive_fails"] += 1
        if st.session_state.get("serial"):
            st.session_state["serial"].send(SERIAL["CMD_FAIL"])
        if defect:
            st.session_state["defect_counts"][defect] = st.session_state["defect_counts"].get(defect, 0) + 1

    st.session_state["log"].append({
        "Timestamp": ts,
        "Item_ID": item_id,
        "Status": status,
        "Defect_Type": defect or "-",
        "Confidence": conf,
        "Action": action,
    })

    # timeline for failures
    if not is_pass:
        minute_mark = ts.timestamp() / 60.0
        st.session_state["defect_timeline"].append((minute_mark, 1))

    check_alarm()


def check_alarm():
    consecutive = st.session_state.get("consecutive_fails", 0)
    if consecutive >= ALARM["consecutive_fail_threshold"]:
        st.session_state["alarm_active"] = True
        st.session_state["alarm_msg"] = f"CRITICAL: {consecutive} consecutive failures detected!"
        return
    total = st.session_state.get("item_counter", 0)
    if total >= ALARM["min_items_for_rate_check"]:
        failed = st.session_state.get("fail_count", 0)
        rate = failed / total
        if rate > ALARM["defect_rate_threshold"]:
            st.session_state["alarm_active"] = True
            st.session_state["alarm_msg"] = f"CRITICAL: Defect rate at {rate:.1%} exceeds {ALARM['defect_rate_threshold']:.0%} threshold!"
            return
    # if neither condition persists, keep alarm off once acknowledged
    if st.session_state.get("alarm_active"):
        # only clear if user hasn't acknowledged or conditions gone away
        pass


def ack_alarm():
    st.session_state["alarm_active"] = False
    st.session_state["alarm_msg"] = ""
    if st.session_state.get("serial"):
        # In a real system this silences a buzzer; here we just log.
        st.session_state["serial"].send(b"ACK\n")
    st.toast("Alarm acknowledged", icon=":material/check_circle:")


def open_snapshot_review():
    """Render a simple two-image comparison view for the last FAIL result."""
    res = st.session_state.get("last_inference_result")
    if res is None or res["is_pass"]:
        st.warning("No defective item available for review.")
        return

    camera = st.session_state.get("camera")
    frame = None
    if camera:
        ok, frame = camera.read()
    if not ok or frame is None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

    golden = camera.get_golden_template() if camera else None
    if golden is None:
        golden = np.zeros_like(frame)

    c1, c2 = st.columns(2)
    with c1:
        st.image(bgr_to_rgb(frame), channels="RGB", caption="Captured Snapshot (Defective)", use_container_width=True)
    with c2:
        st.image(bgr_to_rgb(golden), channels="RGB", caption="Golden Template (Perfect Cap)", use_container_width=True)

    defect = res.get("defect_type") or "Unspecified"
    conf = res.get("confidence", 0)
    st.markdown(
        f"<div class='metric-card'><div class='metric-label'>Defect Classification</div>"
        f"<div class='metric-value fail'>{defect}</div></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div class='metric-card'><div class='metric-label'>Confidence Score</div>"
        f"<div class='metric-value'>{conf:.1%}</div></div>",
        unsafe_allow_html=True,
    )


# =============================================================================
# Export
# =============================================================================

def export_pdf():
    total = st.session_state["item_counter"]
    passed = st.session_state["pass_count"]
    failed = st.session_state["fail_count"]
    session_start = st.session_state.get("session_start")
    if session_start:
        duration = datetime.now() - session_start
        total_seconds = int(duration.total_seconds())
        h, m, s = total_seconds // 3600, (total_seconds % 3600) // 60, total_seconds % 60
        session_time = f"{h:02d}:{m:02d}:{s:02d}"
    else:
        session_time = "00:00:00"

    defect_counts = dict(st.session_state.get("defect_counts", {}))
    timeline = list(st.session_state.get("defect_timeline", []))
    batch_id = st.session_state.get("batch_id", "BATCH-2026-001")
    operator_id = st.session_state.get("operator_id", "Op_Rajan")

    exporter = PDFExporter()
    path = os.path.abspath(f"AIMS_Report_{batch_id}.txt")
    exporter.export(
        file_path=path,
        session_time=session_time,
        total_units=total,
        pass_count=passed,
        fail_count=failed,
        defect_counts=defect_counts,
        timeline=timeline,
        batch_id=batch_id,
        operator_id=operator_id,
        component=COMPONENT_LABEL,
    )
    st.success(f"Report exported to: `{path}`")


def export_csv():
    log = st.session_state.get("log", [])
    if not log:
        st.warning("No session data to export.")
        return
    df = pd.DataFrame(log)
    df["Timestamp"] = df["Timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S.%f").str[:-3]
    df["Confidence"] = df["Confidence"].apply(lambda x: f"{x:.1%}")
    path = os.path.abspath(f"AIMS_Log_{st.session_state.get('batch_id', 'BATCH')}.csv")
    df.to_csv(path, index=False)
    st.success(f"Log exported to: `{path}`")


# =============================================================================
# Background trigger processor (non-blocking, UI-friendly)
# =============================================================================

def start_trigger_processor():
    """One-shot background job to drain queued TRIGs quickly."""
    def _drain():
        flush_pending_triggers()

    t = threading.Thread(target=_drain, daemon=True)
    t.start()


# =============================================================================
# Main app
# =============================================================================

def main():
    st.set_page_config(
        page_title="AIMS HMI",
        page_icon=":material/settings_suggest:",
        layout="wide",
        initial_sidebar_state="collapsed",
        menu_items={
            "About": f"AIMS - AI-driven Industrial Manufacturing Sorter. More info: {ABOUT_URL}",
        },
    )

    init_session_state()

    # Always pull the latest camera frame so the feed appears "live"
    refresh_hardware_frames()

    # Drain any queued TRIGs from the mock serial thread
    start_trigger_processor()

    # Header (common to both states)
    render_header()

    # State router
    if st.session_state["state"] == "report":
        render_report_view()
    else:
        render_active_run_view()


if __name__ == "__main__":
    main()
