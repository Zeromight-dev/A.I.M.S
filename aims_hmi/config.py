"""
AIMS HMI Configuration Constants

Defines all UI styling, hardware protocol parameters, thresholds,
and application settings in one centralized location.
"""

from dataclasses import dataclass
from typing import Optional


# =============================================================================
# UI THEME CONSTANTS (from spec Section 3.1)
# =============================================================================

# Color palette - Dark theme
COLORS = {
    "bg_dark": "#1E1E2E",          # Main background (Dark Slate)
    "bg_card": "#252538",          # Card background
    "bg_panel": "#2D2D42",         # Panel background
    "bg_input": "#363650",         # Input field background
    "text_primary": "#FFFFFF",     # Primary text
    "text_secondary": "#A0A0B0",   # Secondary/muted text
    "text_dim": "#6B6B80",         # Dim text
    "accent_green": "#2ECC71",     # Connected/Pass/Start
    "accent_red": "#E74C3C",       # Disconnected/Fail/Stop
    "accent_yellow": "#F1C40F",    # Warning/Attention
    "accent_blue": "#3498DB",      # Info/Links
    "border": "#3D3D56",           # Border color
    "border_focus": "#4ECDC4",     # Focus border
    "row_pass_bg": "#1A3A2A",      # Light green tint for PASS rows
    "row_fail_bg": "#3A1A1A",      # Light red tint for FAIL rows
    "alarm_flash": "#FF4444",      # Flashing alarm color
    "alarm_bg": "#2A0A0A",         # Alarm banner background
}

# Font specifications
FONTS = {
    "title": ("Arial", 18, True),          # Header title
    "subtitle": ("Arial", 12, True),       # Panel titles
    "body": ("Arial", 10, False),          # Body text
    "metric_value": ("Arial", 24, True),   # Metric card values
    "metric_label": ("Arial", 9, False),   # Metric card labels
    "table_header": ("Arial", 10, True),   # Table headers
    "table_cell": ("Arial", 9, False),     # Table cells
    "fps": ("Courier", 10, True),          # FPS counter
    "overlay_text": ("Arial", 11, True),   # YOLO overlay text
    "button": ("Arial", 10, True),         # Buttons
    "input": ("Arial", 10, False),         # Input fields
}


# =============================================================================
# LAYOUT CONSTANTS
# =============================================================================

LAYOUT = {
    "window_width": 1600,
    "window_height": 900,
    "min_width": 1200,
    "min_height": 700,
    "header_height": 70,
    "left_panel_width_ratio": 0.55,   # Video panel takes ~55% width
    "right_panel_width_ratio": 0.45,  # Metrics panel takes ~45% width
    "video_aspect_ratio": 16 / 9,
    "metric_card_width": 180,
    "metric_card_height": 100,
    "table_max_height": 350,
    "slider_height": 40,
    "button_height": 36,
    "input_height": 32,
}


# =============================================================================
# HARDWARE PROTOCOL CONSTANTS (from spec Section 4 & 5)
# =============================================================================

SERIAL = {
    "default_baudrate": 115200,
    "default_port": "COM3",           # Windows default
    "timeout": 0.1,                   # Serial read timeout (seconds)
    "write_timeout": 0.1,
    # Incoming messages (Arduino -> PC)
    "MSG_TRIG": b"TRIG\n",
    "MSG_ERR_MOTOR": b"ERR_MOTOR\n",
    # Outgoing commands (PC -> Arduino)
    "CMD_PASS": b"0",                 # PASS - servo stays at 0°
    "CMD_FAIL": b"1",                 # FAIL - servo sweeps to 90°
    "CMD_STOP": b"STOP",              # Emergency stop / pause
    "CMD_START": b"START",            # Resume conveyor
    # Servo timing
    "servo_hold_ms": 500,             # How long to hold servo at 90°
}


# =============================================================================
# YOLO / VISION CONSTANTS
# =============================================================================

VISION = {
    "target_fps": 60,
    "capture_width": 1920,
    "capture_height": 1080,
    "display_scale_factor": 0.5,      # Scale down for display (50%)
    # Confidence threshold range (from spec Section 3.2.A.2)
    "threshold_min": 50,
    "threshold_max": 99,
    "threshold_default": 85,
    "threshold_step": 1,
    # Defect types for classification
    "defect_types": [
        "Rim Deformity / Dent",
        "Missing Inner Liner",
        "Print / Logo Misalignment",
        "Surface Scratch",
    ],
    # Annotation colors
    "color_pass": (0, 255, 0),        # Green BGR
    "color_fail": (0, 0, 255),        # Red BGR
    "color_text_bg": (0, 0, 0),       # Black background for text
}


# =============================================================================
# ALARM SYSTEM CONSTANTS (from spec Section 3.2.B.2)
# =============================================================================

ALARM = {
    "consecutive_fail_threshold": 3,  # Trigger after 3 consecutive FAILs
    "defect_rate_threshold": 0.10,    # Trigger when defect rate > 10%
    "min_items_for_rate_check": 20,   # Minimum items before rate check
}


# =============================================================================
# SESSION / DATA CONSTANTS
# =============================================================================

SESSION = {
    "snapshot_dir": "snapshots",      # Directory for captured snapshots
    "export_dir": "exports",          # Directory for exports
    "golden_template_path": None,     # Path to golden template image
    "auto_capture_on_trig": True,     # Auto-capture frame on TRIG
    "capture_quality": 95,            # JPEG quality for snapshots
}


# =============================================================================
# Application Settings Dataclass
# =============================================================================

@dataclass
class AppSettings:
    """Runtime application settings that can be modified."""
    
    # Serial
    serial_port: str = "COM3"
    serial_baudrate: int = 115200
    serial_connected: bool = False
    use_mock_serial: bool = True      # Fallback to mock if port unavailable
    
    # Vision
    camera_index: int = 0
    confidence_threshold: int = VISION["threshold_default"]
    use_mock_camera: bool = True      # Generate synthetic frames if no camera
    
    # YOLO
    model_path: Optional[str] = None  # None = use mock inference
    mock_model_loaded: bool = False
    
    # Session
    batch_id: str = "BATCH-2026-001"
    operator_id: str = "Op_Rajan"
    component_label: str = "Bottle Caps"
    
    # About
    about_url: str = "https://your-project-link.com"


# =============================================================================
# Export format templates
# =============================================================================

CSV_EXPORT_COLUMNS = [
    "Timestamp",
    "Item_ID",
    "Status",
    "Defect_Type",
    "Confidence",
    "Action",
]

PDF_TEMPLATE_TITLE = "AIMS - End of Session Report"
PDF_TEMPLATE_COMPANY = "AI-driven Industrial Manufacturing Sorter"
