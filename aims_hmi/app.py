"""
AIMS HMI - Main Application Module

PyQt6-based Human-Machine Interface implementing the complete
AIMS specification with dual-state machine architecture.

State Machine:
- STATE_A: ACTIVE_RUN (Live Monitoring, Snapshot Inspection, Tracking Log, Hardware Telemetry)
- STATE_B: SESSION_REPORT (Post-Run Telemetry, Defect Pattern Analysis, Export Tools)
"""

import sys
import os
import time
import threading
from datetime import datetime
from typing import Optional, Dict, List, Tuple
from collections import defaultdict

import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QTableWidget, QTableWidgetItem,
    QHeaderView, QFileDialog, QTabWidget, QGroupBox, QFrame,
    QMessageBox, QDialog, QGraphicsOpacityEffect, QStackedLayout,
    QLineEdit, QComboBox, QSplitter, QSizePolicy, QMessageBox
)
from PyQt6.QtCore import (
    Qt, QTimer, QThread, pyqtSignal, QSize, QRect, QPoint,
    QMutex, QDateTime
)
from PyQt6.QtGui import (
    QPixmap, QImage, QPainter, QColor, QFont, QIcon,
    QCursor, QPalette, QBrush, QLinearGradient
)

from aims_hmi.config import (
    COLORS, FONTS, LAYOUT, SERIAL, VISION, ALARM, SESSION,
    AppSettings, CSV_EXPORT_COLUMNS, PDF_TEMPLATE_TITLE
)
from aims_hmi.hardware import SerialManager, MockCamera
from aims_hmi.workers import VisionWorker, InferenceResult
from aims_hmi.views import (
    ActiveRunView, SessionReportView, SnapshotReviewDialog
)
from aims_hmi.exporters import PDFExporter, CSVExporter


# =============================================================================
# Style Sheet Helpers
# =============================================================================

def get_dark_theme_stylesheet() -> str:
    """Generate the dark theme stylesheet for the entire application."""
    return f"""
    /* Global Application Styles */
    QMainWindow {{
        background-color: {COLORS['bg_dark']};
        color: {COLORS['text_primary']};
    }}
    
    QWidget {{
        background-color: {COLORS['bg_dark']};
        color: {COLORS['text_primary']};
        font-family: Arial;
        font-size: 10pt;
    }}
    
    /* Panel/Card Styling */
    QFrame {{
        background-color: {COLORS['bg_card']};
        border: 1px solid {COLORS['border']};
        border-radius: 4px;
    }}
    
    QGroupBox {{
        background-color: {COLORS['bg_card']};
        border: 1px solid {COLORS['border']};
        border-radius: 4px;
        margin-top: 12px;
        padding-top: 10px;
        color: {COLORS['text_primary']};
        font-weight: bold;
        font-size: 11pt;
    }}
    
    QGroupBox ::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        padding: 0 4px;
        color: {COLORS['text_secondary']};
    }}
    
    /* Input Fields */
    QLineEdit {{
        background-color: {COLORS['bg_input']};
        border: 1px solid {COLORS['border']};
        border-radius: 3px;
        color: {COLORS['text_primary']};
        padding: 6px 8px;
        font-size: 10pt;
    }}
    
    QLineEdit:focus {{
        border: 2px solid {COLORS['border_focus']};
    }}
    
    QLineEdit:read-only {{
        background-color: {COLORS['bg_dark']};
        color: {COLORS['text_secondary']};
    }}
    
    /* Buttons */
    QPushButton {{
        background-color: {COLORS['bg_card']};
        border: 1px solid {COLORS['border']};
        border-radius: 4px;
        color: {COLORS['text_primary']};
        padding: 8px 16px;
        font-weight: bold;
        font-size: 10pt;
        min-height: {LAYOUT['button_height']}px;
    }}
    
    QPushButton:hover {{
        background-color: {COLORS['bg_panel']};
        border-color: {COLORS['border_focus']};
    }}
    
    QPushButton:pressed {{
        background-color: {COLORS['bg_dark']};
    }}
    
    /* Specialized Button Colors */
    QPushButton#btn_start {{
        background-color: {COLORS['accent_green']};
        color: #FFFFFF;
        border-color: {COLORS['accent_green']};
    }}
    
    QPushButton#btn_start:hover {{
        background-color: #27AE60;
        border-color: #27AE60;
    }}
    
    QPushButton#btn_stop {{
        background-color: {COLORS['accent_red']};
        color: #FFFFFF;
        border-color: {COLORS['accent_red']};
    }}
    
    QPushButton#btn_stop:hover {{
        background-color: #C0392B;
        border-color: #C0392B;
    }}
    
    QPushButton#btn_ack {{
        background-color: {COLORS['accent_yellow']};
        color: #000000;
        border-color: {COLORS['accent_yellow']};
    }}
    
    QPushButton#btn_export {{
        background-color: {COLORS['accent_blue']};
        color: #FFFFFF;
        border-color: {COLORS['accent_blue']};
    }}
    
    /* Sliders */
    QSlider::groove:horizontal {{
        background-color: {COLORS['bg_input']};
        height: 8px;
        border-radius: 4px;
    }}
    
    QSlider::handle:horizontal {{
        background-color: {COLORS['accent_blue']};
        width: 18px;
        margin: -5px 0;
        border-radius: 9px;
    }}
    
    QSlider::handle:horizontal:hover {{
        background-color: {COLORS['border_focus']};
    }}
    
    QSlider::sub-page:horizontal {{
        background-color: {COLORS['accent_green']};
        border-radius: 4px;
    }}
    
    /* Tables */
    QTableWidget {{
        background-color: {COLORS['bg_card']};
        alternate-background-color: {COLORS['bg_panel']};
        gridline-color: {COLORS['border']};
        border: 1px solid {COLORS['border']};
        color: {COLORS['text_primary']};
    }}
    
    QTableWidget::item:selected {{
        background-color: {COLORS['accent_blue']};
        color: #FFFFFF;
    }}
    
    QHeaderView::section {{
        background-color: {COLORS['bg_panel']};
        color: {COLORS['text_secondary']};
        padding: 6px;
        border: none;
        border-bottom: 2px solid {COLORS['border']};
        font-weight: bold;
        font-size: 9pt;
    }}
    
    /* Scrollbars */
    QScrollBar:vertical {{
        background-color: {COLORS['bg_dark']};
        width: 12px;
        border: none;
    }}
    
    QScrollBar::handle:vertical {{
        background-color: {COLORS['bg_panel']};
        border-radius: 6px;
        min-height: 20px;
    }}
    
    QScrollBar::handle:vertical:hover {{
        background-color: {COLORS['border']};
    }}
    
    QScrollBar::add-line:vertical,
    QScrollBar::sub-line:vertical {{
        height: 0px;
    }}
    
    /* Status Indicator */
    QLabel#status_pill {{
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: bold;
        font-size: 9pt;
    }}
    
    QLabel#status_pill_connected {{
        background-color: {COLORS['accent_green']};
        color: #FFFFFF;
    }}
    
    QLabel#status_pill_disconnected {{
        background-color: {COLORS['accent_red']};
        color: #FFFFFF;
    }}
    
    /* Alarm Banner */
    QLabel#alarm_banner {{
        background-color: {COLORS['alarm_bg']};
        color: {COLORS['alarm_flash']};
        font-weight: bold;
        font-size: 12pt;
        border: 2px solid {COLORS['alarm_flash']};
        border-radius: 4px;
        padding: 8px 16px;
    }}
    
    /* Metric Cards */
    QLabel#metric_value {{
        font-size: 24pt;
        font-weight: bold;
        color: {COLORS['text_primary']};
    }}
    
    QLabel#metric_label {{
        font-size: 9pt;
        color: {COLORS['text_secondary']};
        text-transform: uppercase;
    }}
    
    /* FPS Counter */
    QLabel#fps_counter {{
        background-color: rgba(0, 0, 0, 0.7);
        color: {COLORS['accent_green']};
        font-family: Courier;
        font-size: 10pt;
        font-weight: bold;
        padding: 4px 8px;
        border-radius: 3px;
    }}
    
    /* Tab Widget */
    QTabWidget::pane {{
        background-color: {COLORS['bg_card']};
        border: 1px solid {COLORS['border']};
        border-radius: 4px;
    }}
    
    QTabBar::tab {{
        background-color: {COLORS['bg_panel']};
        color: {COLORS['text_secondary']};
        padding: 8px 16px;
        border: 1px solid {COLORS['border']};
        border-bottom: none;
        border-top-left-radius: 4px;
        border-top-right-radius: 4px;
        margin-right: 2px;
    }}
    
    QTabBar::tab:selected {{
        background-color: {COLORS['bg_dark']};
        color: {COLORS['text_primary']};
        border-bottom-color: {COLORS['bg_dark']};
    }}
    
    QTabBar::tab:hover:!selected {{
        background-color: {COLORS['bg_input']};
    }}
    
    /* Combo box */
    QComboBox {{
        background-color: {COLORS['bg_input']};
        border: 1px solid {COLORS['border']};
        border-radius: 3px;
        color: {COLORS['text_primary']};
        padding: 6px 8px;
    }}
    
    QComboBox::drop-down {{
        border: none;
        width: 20px;
    }}
    
    QComboBox:hover {{
        border-color: {COLORS['border_focus']};
    }}
    
    /* Dialog */
    QDialog {{
        background-color: {COLORS['bg_dark']};
    }}
    
    /* Menu */
    QMenu {{
        background-color: {COLORS['bg_card']};
        border: 1px solid {COLORS['border']};
        color: {COLORS['text_primary']};
    }}
    
    QMenu::item {{
        padding: 6px 20px;
    }}
    
    QMenu::item:selected {{
        background-color: {COLORS['accent_blue']};
    }}
    
    /* Tooltip */
    QToolTip {{
        background-color: {COLORS['bg_panel']};
        color: {COLORS['text_primary']};
        border: 1px solid {COLORS['border']};
        border-radius: 4px;
        padding: 4px 8px;
    }}
    """


# =============================================================================
# Custom Widgets
# =============================================================================

class MetricCard(QFrame):
    """A metric display card with label and value."""
    
    def __init__(self, label: str, value: str = "0", 
                 value_color: str = COLORS['text_primary'],
                 parent=None):
        super().__init__(parent)
        self.setFixedSize(LAYOUT['metric_card_width'], LAYOUT['metric_card_height'])
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['bg_card']};
                border: 1px solid {COLORS['border']};
                border-radius: 6px;
            }}
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(4)
        
        self.label = QLabel(label.upper())
        self.label.setObjectName("metric_label")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = QFont(FONTS['metric_label'][0], FONTS['metric_label'][1], 
                     QFont.Weight.Bold if FONTS['metric_label'][2] else QFont.Weight.Normal)
        self.label.setFont(font)
        
        self.value_label = QLabel(value)
        self.value_label.setObjectName("metric_value")
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = QFont(FONTS['metric_value'][0], FONTS['metric_value'][1],
                     QFont.Weight.Bold if FONTS['metric_value'][2] else QFont.Weight.Normal)
        self.value_label.setFont(font)
        self.value_label.setStyleSheet(f"color: {value_color};")
        
        layout.addWidget(self.label)
        layout.addWidget(self.value_label)
    
    def update_value(self, value: str, color: str = None):
        """Update the displayed value."""
        self.value_label.setText(value)
        if color:
            self.value_label.setStyleSheet(f"color: {color};")


class StatusPill(QLabel):
    """Hardware status indicator pill."""
    
    def __init__(self, parent=None):
        super().__init__("SERIAL: DISCONNECTED", parent)
        self.setObjectName("status_pill")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setFixedHeight(24)
        self.update_status(False)
    
    def update_status(self, connected: bool, port_info: str = ""):
        """Update the status display."""
        if connected:
            self.setText(f"SERIAL: CONNECTED ({port_info})")
            self.setObjectName("status_pill status_pill_connected")
        else:
            self.setText("SERIAL: DISCONNECTED")
            self.setObjectName("status_pill status_pill_disconnected")


class AlarmBanner(QLabel):
    """Flashing alarm banner for critical alerts."""
    
    def __init__(self, parent=None):
        super().__init__("CRITICAL DEFECT RATE DETECTED - AUDIT LINE", parent)
        self.setObjectName("alarm_banner")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setWordWrap(True)
        self.setVisible(False)
        
        # Set up flashing effect
        self.opacity_effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self.opacity_effect)
        self._flash_timer = QTimer(self)
        self._flash_timer.timeout.connect(self._toggle_flash)
        self._flash_state = False
    
    def _toggle_flash(self):
        """Toggle the flashing effect."""
        self._flash_state = not self._flash_state
        opacity = 0.7 if self._flash_state else 1.0
        self.opacity_effect.setOpacity(opacity)
    
    def show_alarm(self):
        """Display the alarm with flashing effect."""
        self.setVisible(True)
        self._flash_state = False
        self.opacity_effect.setOpacity(1.0)
        self._flash_timer.start(500)  # Flash every 500ms
    
    def hide_alarm(self):
        """Hide the alarm and stop flashing."""
        self._flash_timer.stop()
        self.setVisible(False)
        self.opacity_effect.setOpacity(1.0)


class AnnotatedVideoCanvas(QWidget):
    """Widget for displaying video frames with YOLO annotations."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(320, 240)
        self.setStyleSheet(f"background-color: {COLORS['bg_dark']};")
        
        self._frame: Optional[np.ndarray] = None
        self._fps_counter = QLabel(self)
        self._fps_counter.setObjectName("fps_counter")
        self._fps_counter.setText("FPS: 0")
        self._fps_counter.setAlignment(Qt.AlignmentFlag.AlignRight | 
                                        Qt.AlignmentFlag.AlignTop)
        
        # Scale factor for display
        self._display_scale = VISION['display_scale_factor']
    
    def update_frame(self, frame: np.ndarray, fps: float = 0):
        """
        Update the canvas with a new frame.
        
        Args:
            frame: numpy array (BGR format from OpenCV)
            fps: current frames per second
        """
        self._frame = frame
        self._fps_counter.setText(f"FPS: {int(fps)}")
        
        # Trigger repaint
        self.update()
    
    def paintEvent(self, event):
        """Custom paint event to draw the frame."""
        if self._frame is None:
            return
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Convert OpenCV BGR frame to QImage
        frame = self._frame
        
        # Ensure contiguous array
        frame = np.ascontiguousarray(frame)
        
        # Calculate display size maintaining aspect ratio
        h, w = frame.shape[:2]
        target_w = int(self.width() * self._display_scale)
        target_h = int(target_w * h / w)
        
        # Scale the frame
        import cv2
        scaled = cv2.resize(frame, (target_w, target_h), 
                           interpolation=cv2.INTER_LINEAR)
        
        # Convert to QImage
        rgb_frame = cv2.cvtColor(scaled, cv2.COLOR_BGR2RGB)
        bytes_per_line = 3 * target_w
        q_image = QImage(rgb_frame.data, target_w, target_h, 
                        bytes_per_line, QImage.Format.Format_RGB888)
        
        # Draw the image centered
        x = (self.width() - target_w) // 2
        y = (self.height() - target_h) // 2
        painter.drawImage(QPoint(x, y), q_image)
        
        # Draw FPS counter
        painter.end()


# =============================================================================
# Main Application Window
# =============================================================================

class AIMSMainWindow(QMainWindow):
    """
    Main AIMS HMI Application Window.
    
    Implements the dual-state machine architecture:
    - STATE_A: ACTIVE_RUN
    - STATE_B: SESSION_REPORT
    """
    
    def __init__(self):
        super().__init__()
        
        # Initialize settings
        self.settings = AppSettings()
        
        # Thread management
        self._shutdown_flag = threading.Event()
        
        # Serial manager
        self.serial_manager: Optional[SerialManager] = None
        
        # Camera (mock or real)
        self.camera: Optional[MockCamera] = None
        
        # Vision worker thread
        self.vision_worker: Optional[VisionWorker] = None
        self._vision_thread: Optional[QThread] = None
        
        # Session data
        self._session_active = False
        self._session_start_time: Optional[datetime] = None
        self._item_counter = 0
        self._pass_count = 0
        self._fail_count = 0
        self._consecutive_fails = 0
        self._tracking_log: List[Dict] = []
        self._defect_counts: Dict[str, int] = defaultdict(int)
        self._defect_timeline: List[Tuple[float, int]] = []  # (timestamp, count)
        
        # Alarm state
        self._alarm_active = False
        self._alarm_acknowledged = False
        
        # Callbacks storage for worker signals
        self._on_inference_result = None
        
        # Initialize UI
        self._init_ui()
        self._init_workers()
        self._init_connections()
        
        # Start in Active Run state
        self._state = "ACTIVE_RUN"
        self._show_active_run_view()
    
    def _init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("AIMS - AI-driven Industrial Manufacturing Sorter")
        self.setMinimumSize(LAYOUT['min_width'], LAYOUT['min_height'])
        self.resize(LAYOUT['window_width'], LAYOUT['window_height'])
        
        # Apply dark theme
        self.setStyleSheet(get_dark_theme_stylesheet())
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Global Header Bar
        self._header_bar = self._create_header_bar()
        main_layout.addWidget(self._header_bar)
        
        # Stacked layout for state views
        self._view_stack = QStackedLayout()
        main_layout.addLayout(self._view_stack)
        
        # Initialize views
        self._active_run_view = ActiveRunView(self)
        self._session_report_view = SessionReportView(self)
        
        self._view_stack.addWidget(self._active_run_view)
        self._view_stack.addWidget(self._session_report_view)
    
    def _create_header_bar(self) -> QFrame:
        """Create the global header bar."""
        header = QFrame()
        header.setFixedHeight(LAYOUT['header_height'])
        header.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['bg_dark']};
                border-bottom: 2px solid {COLORS['border']};
            }}
        """)
        
        layout = QHBoxLayout(header)
        layout.setContentsMargins(16, 8, 16, 8)
        layout.setSpacing(16)
        
        # Title
        title_label = QLabel("AIMS - AI-driven Industrial Manufacturing Sorter")
        title_label.setFont(QFont(FONTS['title'][0], FONTS['title'][1],
                                  QFont.Weight.Bold))
        title_label.setStyleSheet(f"color: {COLORS['text_primary']};")
        layout.addWidget(title_label)
        
        layout.addStretch()
        
        # Batch Metadata Group
        batch_group = QGroupBox("Batch Information")
        batch_group.setFixedWidth(320)
        batch_layout = QVBoxLayout(batch_group)
        batch_layout.setSpacing(6)
        
        # Batch ID Input
        batch_id_layout = QHBoxLayout()
        batch_id_label = QLabel("Batch ID:")
        batch_id_label.setFont(QFont(FONTS['body'][0], FONTS['body'][1],
                                      QFont.Weight.Bold))
        self._batch_id_input = QLineEdit(self.settings.batch_id)
        self._batch_id_input.setFixedHeight(LAYOUT['input_height'])
        batch_id_layout.addWidget(batch_id_label)
        batch_id_layout.addWidget(self._batch_id_input)
        batch_layout.addLayout(batch_id_layout)
        
        # Operator ID Input
        operator_layout = QHBoxLayout()
        operator_label = QLabel("Operator ID:")
        operator_label.setFont(QFont(FONTS['body'][0], FONTS['body'][1],
                                     QFont.Weight.Bold))
        self._operator_id_input = QLineEdit(self.settings.operator_id)
        self._operator_id_input.setFixedHeight(LAYOUT['input_height'])
        operator_layout.addWidget(operator_label)
        operator_layout.addWidget(self._operator_id_input)
        batch_layout.addLayout(operator_layout)
        
        # Component Label (Read-only)
        component_layout = QHBoxLayout()
        component_label = QLabel("Component:")
        component_label.setFont(QFont(FONTS['body'][0], FONTS['body'][1],
                                      QFont.Weight.Bold))
        self._component_label = QLabel(self.settings.component_label)
        self._component_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        component_layout.addWidget(component_label)
        component_layout.addWidget(self._component_label)
        batch_layout.addLayout(component_layout)
        
        layout.addWidget(batch_group)
        
        # Hardware Status Indicator
        self._status_pill = StatusPill()
        layout.addWidget(self._status_pill)
        
        layout.addStretch()
        
        # About Button
        self._btn_about = QPushButton("About")
        self._btn_about.setFixedHeight(LAYOUT['button_height'])
        layout.addWidget(self._btn_about)
        
        # Session Toggle Button
        self._btn_session_toggle = QPushButton("START SESSION")
        self._btn_session_toggle.setObjectName("btn_start")
        self._btn_session_toggle.setFixedHeight(LAYOUT['button_height'])
        self._btn_session_toggle.setFixedWidth(140)
        layout.addWidget(self._btn_session_toggle)
        
        return header
    
    def _init_workers(self):
        """Initialize background workers."""
        # Create serial manager
        self.serial_manager = SerialManager(self.settings)
        
        # Set up serial callbacks
        self.serial_manager.on_trig = self._handle_serial_trig
        
        # Create mock camera
        cam_width = int(LAYOUT['window_width'] * LAYOUT['left_panel_width_ratio'] 
                       * VISION['display_scale_factor'])
        cam_height = int(cam_width / VISION['video_aspect_ratio'])
        self.camera = MockCamera(cam_width, cam_height)
        
        # Create vision worker
        self.vision_worker = VisionWorker(
            camera=self.camera,
            serial_manager=self.serial_manager,
            confidence_threshold=self.settings.confidence_threshold,
            mock_mode=True
        )
        
        # Set up vision worker callbacks
        self.vision_worker.inference_result.connect(self._handle_inference_result)
        self.vision_worker.frame_ready.connect(self._active_run_view.update_video_frame)
        self.vision_worker.fps_updated.connect(self._active_run_view.update_fps)
    
    def _init_connections(self):
        """Connect UI signals to slots."""
        # Session toggle
        self._btn_session_toggle.clicked.connect(self._toggle_session)
        
        # About button
        self._btn_about.clicked.connect(self._show_about)
        
        # Acknowledge alarm button (in active run view)
        self._active_run_view.btn_ack_alarm.clicked.connect(self._acknowledge_alarm)
        
        # Export buttons (in session report view)
        self._session_report_view.btn_export_pdf.clicked.connect(self._export_pdf)
        self._session_report_view.btn_export_csv.clicked.connect(self._export_csv)
        self._session_report_view.btn_new_session.clicked.connect(self._new_session)
        
        # AI threshold slider
        self._active_run_view.threshold_slider.valueChanged.connect(
            self._on_threshold_changed
        )
        
        # Serial connection
        self.serial_manager.on_connect = self._on_serial_connected
        self.serial_manager.on_disconnect = self._on_serial_disconnected
    
    # =========================================================================
    # Session State Management
    # =========================================================================
    
    def _toggle_session(self):
        """Toggle between session start and stop."""
        if not self._session_active:
            self._start_session()
        else:
            self._stop_session()
    
    def _start_session(self):
        """Start a new sorting session."""
        # Update button appearance
        self._btn_session_toggle.setText("STOP SESSION")
        self._btn_session_toggle.setObjectName("btn_stop")
        
        # Record session start
        self._session_active = True
        self._session_start_time = datetime.now()
        self._item_counter = 0
        self._pass_count = 0
        self._fail_count = 0
        self._consecutive_fails = 0
        self._tracking_log = []
        self._defect_counts = defaultdict(int)
        self._defect_timeline = []
        self._alarm_active = False
        self._alarm_acknowledged = False
        
        # Clear tracking table
        self._active_run_view.clear_tracking_table()
        
        # Reset metrics
        self._active_run_view.update_metrics(
            total=0, 
            passed=0, 
            failed=0
        )
        
        # Start camera and vision worker
        if self.camera:
            self.camera.start()
        
        if self.vision_worker:
            self.vision_worker.start()
        
        # Send START command to hardware
        if self.serial_manager:
            self.serial_manager.send_start()
        
        # Show active run view
        self._show_active_run_view()
    
    def _stop_session(self):
        """Stop the current session and transition to report view."""
        # Update button appearance
        self._btn_session_toggle.setText("START SESSION")
        self._btn_session_toggle.setObjectName("btn_start")
        
        # Stop session
        self._session_active = False
        
        # Stop camera and vision worker
        if self.camera:
            self.camera.stop()
        
        if self.vision_worker:
            self.vision_worker.stop()
        
        # Send STOP command to hardware
        if self.serial_manager:
            self.serial_manager.send_stop()
        
        # Transition to session report
        self._show_session_report_view()
    
    def _show_active_run_view(self):
        """Display the active run (State A) view."""
        self._view_stack.setCurrentIndex(0)
        self._state = "ACTIVE_RUN"
        
        # Clear alarm
        self._active_run_view.hide_alarm()
    
    def _show_session_report_view(self):
        """Display the session report (State B) view."""
        self._view_stack.setCurrentIndex(1)
        self._state = "SESSION_REPORT"
        
        # Update report view with session data
        self._session_report_view.update_report(
            session_time=self._get_session_duration(),
            total_units=self._item_counter,
            pass_count=self._pass_count,
            fail_count=self._fail_count,
            defect_counts=dict(self._defect_counts),
            timeline=self._defect_timeline,
            batch_id=self._batch_id_input.text(),
            operator_id=self._operator_id_input.text()
        )
    
    def _get_session_duration(self) -> str:
        """Get formatted session duration."""
        if self._session_start_time:
            delta = datetime.now() - self._session_start_time
            total_seconds = int(delta.total_seconds())
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            seconds = total_seconds % 60
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        return "00:00:00"
    
    def _new_session(self):
        """Reset everything and prepare for a new session."""
        # Clear all data
        self._session_start_time = None
        self._item_counter = 0
        self._pass_count = 0
        self._fail_count = 0
        self._consecutive_fails = 0
        self._tracking_log = []
        self._defect_counts = defaultdict(int)
        self._defect_timeline = []
        self._alarm_active = False
        self._alarm_acknowledged = False
        
        # Clear tracking table
        self._active_run_view.clear_tracking_table()
        
        # Reset metrics
        self._active_run_view.update_metrics(0, 0, 0)
        
        # Clear report view
        self._session_report_view.clear()
        
        # Return to active run view
        self._show_active_run_view()
        
        # Update button
        self._btn_session_toggle.setText("START SESSION")
        self._btn_session_toggle.setObjectName("btn_start")
    
    # =========================================================================
    # Hardware / Serial Event Handlers
    # =========================================================================
    
    def _handle_serial_trig(self):
        """
        Handle incoming TRIG message from Arduino.
        
        This is called when the IR sensor detects a bottle cap.
        """
        if not self._session_active:
            return
        
        # Increment item counter
        self._item_counter += 1
        item_id = f"ITEM-{self._item_counter:05d}"
        
        # Get latest frame from camera for analysis
        frame = None
        if self.camera:
            success, frame = self.camera.read()
        
        # Store pending item for when inference completes
        self._pending_item = {
            'item_id': item_id,
            'timestamp': datetime.now(),
            'frame': frame
        }
        
        # Trigger inference (worker will pick this up)
        if self.vision_worker:
            self.vision_worker.request_inference()
    
    def _handle_inference_result(self, result: InferenceResult):
        """
        Handle completed inference result.
        
        Args:
            result: InferenceResult with classification data
        """
        if not hasattr(self, '_pending_item') or self._pending_item is None:
            return
        
        pending = self._pending_item
        self._pending_item = None
        
        item_id = pending['item_id']
        timestamp = pending['timestamp']
        frame = pending['frame']
        
        # Determine status and action
        is_pass = result.is_pass
        confidence = result.confidence
        defect_type = result.defect_type
        
        if is_pass:
            status = "PASS"
            action = "ACCEPT"
            self._pass_count += 1
            self._consecutive_fails = 0
            # Send PASS command to hardware
            if self.serial_manager:
                self.serial_manager.send_pass()
        else:
            status = "FAIL"
            action = "REJECT"
            self._fail_count += 1
            self._consecutive_fails += 1
            # Send FAIL command to hardware
            if self.serial_manager:
                self.serial_manager.send_fail()
            
            # Track defect type
            if defect_type:
                self._defect_counts[defect_type] += 1
        
        # Record in timeline (for defect spike analysis)
        if not is_pass:
            minute_mark = timestamp.timestamp() / 60  # Convert to minutes
            self._defect_timeline.append((minute_mark, 1))
        
        # Add to tracking log
        log_entry = {
            'timestamp': timestamp,
            'item_id': item_id,
            'status': status,
            'defect_type': defect_type or '-',
            'confidence': confidence,
            'action': action
        }
        self._tracking_log.append(log_entry)
        
        # Update UI
        self._active_run_view.add_tracking_entry(log_entry)
        self._active_run_view.update_metrics(
            total=self._item_counter,
            passed=self._pass_count,
            failed=self._fail_count
        )
        
        # Check alarm conditions
        self._check_alarm_conditions()
    
    def _check_alarm_conditions(self):
        """Check if alarm conditions are met."""
        if not self._session_active:
            return
        
        # Condition 1: 3 consecutive fails
        if self._consecutive_fails >= ALARM['consecutive_fail_threshold']:
            if not self._alarm_active:
                self._trigger_alarm(
                    f"CRITICAL: {self._consecutive_fails} consecutive failures detected!"
                )
            return
        
        # Condition 2: Defect rate > 10%
        total = self._item_counter
        if total >= ALARM['min_items_for_rate_check']:
            defect_rate = self._fail_count / total
            if defect_rate > ALARM['defect_rate_threshold']:
                if not self._alarm_active:
                    self._trigger_alarm(
                        f"CRITICAL: Defect rate at {defect_rate:.1%} exceeds {ALARM['defect_rate_threshold']:.0%} threshold!"
                    )
    
    def _trigger_alarm(self, message: str):
        """Trigger the alarm banner."""
        self._alarm_active = True
        self._alarm_acknowledged = False
        self._active_run_view.show_alarm(message)
    
    def _acknowledge_alarm(self):
        """Acknowledge and silence the alarm."""
        self._alarm_active = False
        self._alarm_acknowledged = True
        self._active_run_view.hide_alarm()
        
        # Send silence command over serial if buzzer active
        if self.serial_manager:
            # In a real system, this would send a specific silence command
            # For now, just log it
            pass
        
        QMessageBox.information(
            self,
            "Alarm Acknowledged",
            "Alarm has been acknowledged and silenced.\n"
            "Please investigate the line for quality issues."
        )
    
    # =========================================================================
    # Threshold / Settings
    # =========================================================================
    
    def _on_threshold_changed(self, value: int):
        """Handle AI confidence threshold slider change."""
        self.settings.confidence_threshold = value
        if self.vision_worker:
            self.vision_worker.set_confidence_threshold(value)
    
    # =========================================================================
    # About Dialog
    # =========================================================================
    
    def _show_about(self):
        """Show the about dialog."""
        from webbrowser import open_new_tab
        open_new_tab(self.settings.about_url)
    
    # =========================================================================
    # Export Functions
    # =========================================================================
    
    def _export_pdf(self):
        """Export session report to PDF."""
        if not self._tracking_log:
            QMessageBox.warning(
                self,
                "Export Error",
                "No session data to export. Please run a session first."
            )
            return
        
        # Prompt for save location
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Report as PDF",
            f"AIMS_Report_{self._batch_id_input.text()}.pdf",
            "PDF Files (*.pdf);;All Files (*)"
        )
        
        if not file_path:
            return
        
        try:
            exporter = PDFExporter(self._session_report_view)
            exporter.export(
                file_path=file_path,
                session_time=self._get_session_duration(),
                total_units=self._item_counter,
                pass_count=self._pass_count,
                fail_count=self._fail_count,
                defect_counts=dict(self._defect_counts),
                timeline=self._defect_timeline,
                batch_id=self._batch_id_input.text(),
                operator_id=self._operator_id_input.text(),
                component=self._component_label.text()
            )
            
            QMessageBox.information(
                self,
                "Export Successful",
                f"Report exported to:\n{file_path}"
            )
        except Exception as e:
            QMessageBox.critical(
                self,
                "Export Error",
                f"Failed to export PDF:\n{str(e)}"
            )
    
    def _export_csv(self):
        """Export tracking log to CSV."""
        if not self._tracking_log:
            QMessageBox.warning(
                self,
                "Export Error",
                "No session data to export. Please run a session first."
            )
            return
        
        # Prompt for save location
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Log as CSV",
            f"AIMS_Log_{self._batch_id_input.text()}.csv",
            "CSV Files (*.csv);;All Files (*)"
        )
        
        if not file_path:
            return
        
        try:
            exporter = CSVExporter()
            exporter.export(
                file_path=file_path,
                data=self._tracking_log,
                columns=CSV_EXPORT_COLUMNS
            )
            
            QMessageBox.information(
                self,
                "Export Successful",
                f"Log exported to:\n{file_path}"
            )
        except Exception as e:
            QMessageBox.critical(
                self,
                "Export Error",
                f"Failed to export CSV:\n{str(e)}"
            )
    
    # =========================================================================
    # Serial Connection Status
    # =========================================================================
    
    def _on_serial_connected(self, connected: bool):
        """Handle serial connection status change."""
        port_info = f"{self.settings.serial_port} @ {self.settings.serial_baudrate}"
        if self._status_pill:
            self._status_pill.update_status(connected, port_info)
    
    def _on_serial_disconnected(self):
        """Handle serial disconnection."""
        if self._status_pill:
            self._status_pill.update_status(False)
    
    # =========================================================================
    # Cleanup
    # =========================================================================
    
    def closeEvent(self, event):
        """Handle window close event."""
        # Clean up workers
        if self.vision_worker:
            self.vision_worker.stop()
        
        if self.camera:
            self.camera.release()
        
        if self.serial_manager:
            self.serial_manager.disconnect()
        
        # Signal shutdown
        self._shutdown_flag.set()
        
        event.accept()


# =============================================================================
# Application Entry Point
# =============================================================================

def main():
    """Main entry point for the AIMS HMI application."""
    # Create application
    app = QApplication(sys.argv)
    
    # Set application style options
    app.setStyle("Fusion")
    
    # Create and show main window
    window = AIMSMainWindow()
    window.show()
    
    # Start event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
