"""
AIMS HMI Views Module

Contains the UI view classes for:
1. ActiveRunView: Live monitoring state (State A)
2. SessionReportView: Post-run analytics state (State B)
3. SnapshotReviewDialog: Modal for reviewing defective cap snapshots
"""

import time
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QSlider, QTableWidget, QTableWidgetItem, QHeaderView,
    QGroupBox, QFrame, QMessageBox, QDialog, QGraphicsOpacityEffect,
    QSplitter, QSizePolicy, QFileDialog, QGridLayout, QScrollArea
)
from PyQt6.QtCore import Qt, QTimer, QSize, QPoint, QRect, QMutex, QDateTime
from PyQt6.QtGui import QPixmap, QImage, QPainter, QColor, QFont, QBrush

from aims_hmi.config import (
    COLORS, FONTS, LAYOUT, VISION, ALARM, SESSION
)
from aims_hmi.workers import InferenceResult


# =============================================================================
# Active Run View (State A)
# =============================================================================

class ActiveRunView(QWidget):
    """
    Active Run View - Live Monitoring State (State A).
    
    Layout:
    +------------------------------------------+----------------------------------+
    | LEFT PANEL (Vision & AI Overlay)         | RIGHT PANEL (Metrics & Live Log) |
    | - OpenCV Video Feed (Canvas)             | - Metric Cards (Total, Pass, Fail)|
    | - Dynamic YOLO Bounding Boxes            | - Live Tracking Data Table        |
    | - AI Sensitivity Slider                  | - Visual Warning Banner           |
    |                                          | - [Acknowledge Alarm] Button      |
    +------------------------------------------+----------------------------------+
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background-color: {COLORS['bg_dark']};")
        
        # Layout
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)
        
        # Create panels
        self._left_panel = self._create_left_panel()
        self._right_panel = self._create_right_panel()
        
        # Add to splitter for resizable panels
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._left_panel)
        splitter.addWidget(self._right_panel)
        splitter.setStretchFactor(0, 55)  # Left panel 55%
        splitter.setStretchFactor(1, 45)  # Right panel 45%
        splitter.setMinimumWidth(600)
        
        main_layout.addWidget(splitter)
        
        # Alarm banner
        self._alarm_banner = None
    
    def _create_left_panel(self) -> QWidget:
        """Create the left panel with video feed and AI controls."""
        panel = QWidget()
        panel.setStyleSheet(f"background-color: {COLORS['bg_card']};")
        panel.setMinimumWidth(400)
        
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)
        
        # Title
        title = QLabel("LIVE VISION & AI CONTROLS")
        title.setFont(QFont(FONTS['subtitle'][0], FONTS['subtitle'][1],
                           QFont.Weight.Bold))
        title.setStyleSheet(f"color: {COLORS['text_secondary']};")
        layout.addWidget(title)
        
        # Video Canvas
        video_group = QGroupBox("Video Feed (OpenCV Stream)")
        video_layout = QVBoxLayout(video_group)
        
        self._video_canvas = AnnotatedVideoCanvas(panel)
        self._video_canvas.setMinimumSize(320, 240)
        self._video_canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                                         QSizePolicy.Policy.Expanding)
        video_layout.addWidget(self._video_canvas)
        
        layout.addWidget(video_group)
        
        # AI Threshold Slider
        slider_group = QGroupBox("AI Detection Threshold")
        slider_layout = QVBoxLayout(slider_group)
        slider_layout.setSpacing(8)
        
        slider_label = QLabel("AI Detection Threshold (%)")
        slider_label.setFont(QFont(FONTS['body'][0], FONTS['body'][1],
                                   QFont.Weight.Bold))
        
        self._threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self._threshold_slider.setMinimum(VISION['threshold_min'])
        self._threshold_slider.setMaximum(VISION['threshold_max'])
        self._threshold_slider.setValue(VISION['threshold_default'])
        self._threshold_slider.setFixedHeight(30)
        
        # Value display
        self._threshold_value = QLabel(f"{VISION['threshold_default']}%")
        self._threshold_value.setFont(QFont(FONTS['body'][0], FONTS['body'][1],
                                            QFont.Weight.Bold))
        self._threshold_value.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._threshold_value.setStyleSheet(f"color: {COLORS['accent_blue']};")
        
        slider_layout.addWidget(slider_label)
        slider_layout.addWidget(self._threshold_slider)
        slider_layout.addWidget(self._threshold_value)
        
        layout.addWidget(slider_group)
        
        # FIFO size info
        info_label = QLabel(
            "Threshold: Minimum confidence required for YOLO classification.\n"
            "Lower values = more detections (higher false positive rate)."
        )
        info_label.setFont(QFont(FONTS['body'][0], FONTS['body'][1],
                                 QFont.Weight.Normal))
        info_label.setStyleSheet(f"color: {COLORS['text_dim']};")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        return panel
    
    def _create_right_panel(self) -> QWidget:
        """Create the right panel with metrics and tracking log."""
        panel = QWidget()
        panel.setStyleSheet(f"background-color: {COLORS['bg_card']};")
        
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)
        
        # Title
        title = QLabel("TELEMETRY & TRACKING LOG")
        title.setFont(QFont(FONTS['subtitle'][0], FONTS['subtitle'][1],
                           QFont.Weight.Bold))
        title.setStyleSheet(f"color: {COLORS['text_secondary']};")
        layout.addWidget(title)
        
        # Metric Cards (3-box layout)
        metric_group = QGroupBox("Live Metrics")
        metric_layout = QHBoxLayout(metric_group)
        metric_layout.setSpacing(8)
        
        # Card 1: Total
        self._metric_total = MetricCard("TOTAL", "0", COLORS['text_primary'])
        metric_layout.addWidget(self._metric_total)
        
        # Card 2: Pass (Green)
        self._metric_pass = MetricCard("PASS", "0", COLORS['accent_green'])
        metric_layout.addWidget(self._metric_pass)
        
        # Card 3: Fail (Red)
        self._metric_fail = MetricCard("FAIL", "0", COLORS['accent_red'])
        metric_layout.addWidget(self._metric_fail)
        
        layout.addWidget(metric_group)
        
        # Warning Banner & Acknowledge Button
        alarm_group = QGroupBox("Alarm System")
        alarm_layout = QVBoxLayout(alarm_group)
        alarm_layout.setSpacing(8)
        
        self._alarm_banner = AlarmBanner(alarm_group)
        alarm_layout.addWidget(self._alarm_banner)
        
        self._btn_ack_alarm = QPushButton("Acknowledge & Silence Alarm")
        self._btn_ack_alarm.setObjectName("btn_ack")
        self._btn_ack_alarm.setFixedHeight(LAYOUT['button_height'])
        self._btn_ack_alarm.setEnabled(False)
        alarm_layout.addWidget(self._btn_ack_alarm)
        
        layout.addWidget(alarm_group)
        
        # Live Tracking Data Table
        table_group = QGroupBox("Tracking Log")
        table_layout = QVBoxLayout(table_group)
        
        self._tracking_table = QTableWidget()
        self._tracking_table.setColumnCount(6)
        self._tracking_table.setHorizontalHeaderLabels([
            "Timestamp", "Item ID", "Status", "Defect Type", 
            "Confidence", "Action"
        ])
        self._tracking_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self._tracking_table.setAlternatingRowColors(True)
        self._tracking_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self._tracking_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers
        )
        
        # Connection for row clicks
        self._tracking_table.cellClicked.connect(self._on_table_row_clicked)
        
        # Scrollbar styling handled by stylesheet
        table_layout.addWidget(self._tracking_table)
        
        layout.addWidget(table_group)
        
        return panel
    
    def update_video_frame(self, frame: np.ndarray):
        """Update the video canvas with a new frame."""
        if self._video_canvas:
            self._video_canvas.update_frame(frame)
    
    def update_fps(self, fps: float):
        """Update the FPS counter display."""
        if self._video_canvas:
            self._video_canvas.update_frame(
                self._video_canvas._frame if hasattr(self._video_canvas, '_frame') else None,
                fps
            )
    
    def update_metrics(self, total: int, passed: int, failed: int):
        """Update all metric cards."""
        self._metric_total.update_value(f"{total:,}")
        self._metric_pass.update_value(f"{passed:,}")
        self._metric_fail.update_value(f"{failed:,}")
    
    def add_tracking_entry(self, entry: Dict):
        """
        Add an entry to the tracking table.
        
        Args:
            entry: Dictionary with timestamp, item_id, status, 
                   defect_type, confidence, action
        """
        row = self._tracking_table.rowCount()
        self._tracking_table.insertRow(row)
        
        # Format timestamp
        timestamp = entry['timestamp'].strftime("%H:%M:%S.%f")[:-3]
        
        # Status-based styling
        status = entry['status']
        if status == "PASS":
            bg_color = COLORS['row_pass_bg']
            status_color = COLORS['accent_green']
        else:
            bg_color = COLORS['row_fail_bg']
            status_color = COLORS['accent_red']
        
        # Create items
        items = [
            QTableWidgetItem(timestamp),
            QTableWidgetItem(entry['item_id']),
            QTableWidgetItem(status),
            QTableWidgetItem(entry['defect_type']),
            QTableWidgetItem(f"{entry['confidence']:.1%}"),
            QTableWidgetItem(entry['action'])
        ]
        
        # Style the status column
        items[2].setForeground(QBrush(QColor(status_color)))
        items[2].setFont(QFont(FONTS['table_cell'][0], FONTS['table_cell'][1],
                               QFont.Weight.Bold))
        
        # Set items and styling
        for col, item in enumerate(items):
            self._tracking_table.setItem(row, col, item)
            
            # Apply row background for PASS/FAIL
            if status in ("PASS", "FAIL"):
                item.setBackground(QBrush(QColor(bg_color)))
        
        # Auto-scroll to bottom
        scrollbar = self._tracking_table.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def clear_tracking_table(self):
        """Clear all entries from the tracking table."""
        self._tracking_table.setRowCount(0)
    
    def show_alarm(self, message: str = None):
        """Show the alarm banner with optional custom message."""
        if message:
            self._alarm_banner.setText(message)
        self._alarm_banner.show_alarm()
        self._btn_ack_alarm.setEnabled(True)
    
    def hide_alarm(self):
        """Hide the alarm banner."""
        self._alarm_banner.hide_alarm()
        self._btn_ack_alarm.setEnabled(False)
    
    def _on_table_row_clicked(self, row: int, column: int):
        """
        Handle table row click - opens snapshot review for FAIL rows.
        
        Args:
            row: Row index that was clicked
            column: Column index that was clicked
        """
        # Only handle FAIL rows
        status_item = self._tracking_table.item(row, 2)
        if not status_item or status_item.text() != "FAIL":
            return
        
        # Get item data
        item_id = self._tracking_table.item(row, 1).text()
        defect_type = self._tracking_table.item(row, 3).text()
        confidence = self._tracking_table.item(row, 4).text()
        
        # Find the original inference result
        # (In production, we'd store the full result in the model)
        
        # Open snapshot review dialog
        dialog = SnapshotReviewDialog(
            item_id=item_id,
            defect_type=defect_type,
            confidence=confidence,
            parent=self
        )
        dialog.exec()


# =============================================================================
# Session Report View (State B)
# =============================================================================

class SessionReportView(QWidget):
    """
    Session Report View - Post-Run Analytics State (State B).
    
    Layout:
    +-----------------------------------------------------------------------------+
    | - Executive Metrics Overview Cards (Yield %, Total Time, Rejection Count)   |
    | - Defect Breakdown Chart (Pie/Bar) | Timeline Defect Spike Graph (Line)     |
    | - Action Bar: [Export PDF] | [Export CSV] | [Start New Session]            |
    +-----------------------------------------------------------------------------+
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background-color: {COLORS['bg_dark']};")
        
        # Store chart widgets
        self._pie_chart: Optional[ChartWidget] = None
        self._line_chart: Optional[ChartWidget] = None
        
        # Layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(16)
        
        # Title
        title = QLabel("END OF SESSION REPORT")
        title.setFont(QFont(FONTS['title'][0], FONTS['title'][1],
                           QFont.Weight.Bold))
        title.setStyleSheet(f"color: {COLORS['text_primary']};")
        main_layout.addWidget(title)
        
        # Executive Summary Metric Cards
        metrics_group = QGroupBox("Executive Summary")
        metrics_layout = QGridLayout(metrics_group)
        metrics_layout.setSpacing(12)
        
        # Card 1: Total Session Time
        self._card_time = MetricCard("TOTAL SESSION TIME", "00:00:00")
        metrics_layout.addWidget(self._card_time, 0, 0)
        
        # Card 2: Total Units Processed
        self._card_units = MetricCard("TOTAL UNITS PROCESSED", "0 units")
        metrics_layout.addWidget(self._card_units, 0, 1)
        
        # Card 3: Line Yield Rate
        self._card_yield = MetricCard("LINE YIELD RATE", "0% PASS")
        metrics_layout.addWidget(self._card_yield, 0, 2)
        
        # Card 4: Total Rejections
        self._card_rejects = MetricCard("TOTAL REJECTIONS", "0 units")
        metrics_layout.addWidget(self._card_rejects, 0, 3)
        
        main_layout.addWidget(metrics_group)
        
        # Charts Section
        charts_group = QGroupBox("Defect Analysis")
        charts_layout = QHBoxLayout(charts_group)
        charts_layout.setSpacing(16)
        
        # Defect Distribution Chart (Pie/Bar)
        pie_group = QGroupBox("Defect Distribution")
        pie_layout = QVBoxLayout(pie_group)
        
        self._pie_chart = ChartWidget(chart_type="pie")
        pie_layout.addWidget(self._pie_chart)
        
        charts_layout.addWidget(pie_group)
        
        # Defect Timeline Chart (Line)
        timeline_group = QGroupBox("Defect Timeline (5-min intervals)")
        timeline_layout = QVBoxLayout(timeline_group)
        
        self._line_chart = ChartWidget(chart_type="line")
        timeline_layout.addWidget(self._line_chart)
        
        charts_layout.addWidget(timeline_group)
        
        main_layout.addWidget(charts_group)
        
        # Action Toolbar
        action_group = QGroupBox("Post-Session Actions")
        action_layout = QHBoxLayout(action_group)
        action_layout.setSpacing(12)
        
        self.btn_export_pdf = QPushButton("Export PDF Report")
        self.btn_export_pdf.setObjectName("btn_export")
        self.btn_export_pdf.setFixedHeight(LAYOUT['button_height'])
        action_layout.addWidget(self.btn_export_pdf)
        
        self.btn_export_csv = QPushButton("Export CSV Log")
        self.btn_export_csv.setObjectName("btn_export")
        self.btn_export_csv.setFixedHeight(LAYOUT['button_height'])
        action_layout.addWidget(self.btn_export_csv)
        
        action_layout.addStretch()
        
        self.btn_new_session = QPushButton("Start New Session")
        self.btn_new_session.setObjectName("btn_start")
        self.btn_new_session.setFixedHeight(LAYOUT['button_height'])
        action_layout.addWidget(self.btn_new_session)
        
        main_layout.addWidget(action_group)
        
        # Footer
        footer = QLabel("Session data retained for export. Click 'Start New Session' to clear.")
        footer.setFont(QFont(FONTS['body'][0], FONTS['body'][1],
                             QFont.Weight.Normal))
        footer.setStyleSheet(f"color: {COLORS['text_dim']};")
        footer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(footer)
    
    def update_report(self, 
                      session_time: str,
                      total_units: int,
                      pass_count: int,
                      fail_count: int,
                      defect_counts: Dict[str, int],
                      timeline: List[Tuple[float, int]],
                      batch_id: str,
                      operator_id: str):
        """
        Update the report view with session data.
        
        Args:
            session_time: Formatted session duration (HH:MM:SS)
            total_units: Total items processed
            pass_count: Number of passed items
            fail_count: Number of failed items
            defect_counts: Dictionary mapping defect types to counts
            timeline: List of (minute, count) tuples for timeline chart
            batch_id: Batch identifier
            operator_id: Operator identifier
        """
        # Update metric cards
        self._card_time.update_value(session_time)
        self._card_units.update_value(f"{total_units:,} units")
        
        # Calculate yield rate
        if total_units > 0:
            yield_rate = (pass_count / total_units) * 100
            yield_text = f"{yield_rate:.1f}% PASS"
        else:
            yield_text = "0% PASS"
        self._card_yield.update_value(yield_text)
        
        self._card_rejects.update_value(f"{fail_count:,} units")
        
        # Update charts
        if defect_counts:
            self._pie_chart.update_chart(defect_counts)
        else:
            self._pie_chart.update_chart({"No Defects Recorded": 0})
        
        if timeline:
            # Aggregate timeline by 5-minute intervals
            aggregated = defaultdict(int)
            for minute, count in timeline:
                interval = int(minute // 5) * 5
                aggregated[interval] += count
            
            # Convert to sorted list for chart
            chart_data = sorted(aggregated.items())
            self._line_chart.update_chart(chart_data)
        else:
            self._line_chart.update_chart([(0, 0)])
    
    def clear(self):
        """Clear all report data."""
        self._card_time.update_value("00:00:00")
        self._card_units.update_value("0 units")
        self._card_yield.update_value("0% PASS")
        self._card_rejects.update_value("0 units")
        self._pie_chart.update_chart({"No Data": 0})
        self._line_chart.update_chart([(0, 0)])


# =============================================================================
# Snapshot Review Dialog
# =============================================================================

class SnapshotReviewDialog(QDialog):
    """
    Modal dialog for reviewing defective cap snapshots.
    
    Layout:
    +------------------------------------------+
    | SNAPSHOT REVIEW - DEFECTIVE ITEM         |
    +------------------------------------------+
    | Left: Captured Snapshot  | Right: Golden |
    | (Defective Cap)          | Template     |
    +---------------------------+--------------+
    | Defect Classification: X                  |
    | Confidence Score: Y%                      |
    +------------------------------------------+
    | [Close]                                   |
    +------------------------------------------+
    """
    
    def __init__(self, 
                 item_id: str,
                 defect_type: str,
                 confidence: str,
                 parent=None):
        super().__init__(parent)
        
        self.setWindowTitle(f"Snapshot Review - {item_id}")
        self.setMinimumSize(800, 500)
        self.setStyleSheet(get_dark_theme_stylesheet())
        
        # Layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(16)
        
        # Title
        title = QLabel(f"SNAPSHOT REVIEW - DEFECTIVE ITEM: {item_id}")
        title.setFont(QFont(FONTS['subtitle'][0], FONTS['subtitle'][1],
                           QFont.Weight.Bold))
        title.setStyleSheet(f"color: {COLORS['text_primary']};")
        main_layout.addWidget(title)
        
        # Image comparison area
        image_group = QGroupBox("Defect Comparison")
        image_layout = QHBoxLayout(image_group)
        image_layout.setSpacing(16)
        
        # Left: Captured snapshot (defective)
        captured_group = QGroupBox("Captured Snapshot (Defective)")
        captured_layout = QVBoxLayout(captured_group)
        
        self._captured_label = QLabel()
        self._captured_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._captured_label.setStyleSheet(
            f"background-color: {COLORS['bg_dark']}; "
            f"border: 2px solid {COLORS['accent_red']};"
        )
        self._captured_label.setMinimumSize(250, 250)
        self._captured_label.setScaledContents(True)
        
        # Set placeholder image
        self._set_placeholder_image(self._captured_label, COLORS['accent_red'])
        
        captured_layout.addWidget(self._captured_label)
        image_layout.addWidget(captured_group)
        
        # Right: Golden template
        golden_group = QGroupBox("Golden Template (Perfect Cap)")
        golden_layout = QVBoxLayout(golden_group)
        
        self._golden_label = QLabel()
        self._golden_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._golden_label.setStyleSheet(
            f"background-color: {COLORS['bg_dark']}; "
            f"border: 2px solid {COLORS['accent_green']};"
        )
        self._golden_label.setMinimumSize(250, 250)
        self._golden_label.setScaledContents(True)
        
        # Set placeholder image
        self._set_placeholder_image(self._golden_label, COLORS['accent_green'])
        
        golden_layout.addWidget(self._golden_label)
        image_layout.addWidget(golden_group)
        
        main_layout.addWidget(image_group)
        
        # Defect information
        info_group = QGroupBox("Defect Classification")
        info_layout = QVBoxLayout(info_group)
        info_layout.setSpacing(8)
        
        defect_label = QLabel(f"Defect Type: {defect_type}")
        defect_label.setFont(QFont(FONTS['body'][0], FONTS['body'][1],
                                   QFont.Weight.Bold))
        defect_label.setStyleSheet(f"color: {COLORS['accent_red']};")
        
        conf_label = QLabel(f"Confidence Score: {confidence}")
        conf_label.setFont(QFont(FONTS['body'][0], FONTS['body'][1],
                                 QFont.Weight.Bold))
        conf_label.setStyleSheet(f"color: {COLORS['text_primary']};")
        
        info_layout.addWidget(defect_label)
        info_layout.addWidget(conf_label)
        
        main_layout.addWidget(info_group)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.setFixedHeight(LAYOUT['button_height'])
        close_btn.clicked.connect(self.accept)
        main_layout.addWidget(close_btn)
    
    def _set_placeholder_image(self, label: QLabel, color: str):
        """Set a placeholder image for the label."""
        from aims_hmi.hardware import MockCamera
        
        # Create a simple placeholder using QPixmap
        pixmap = QPixmap(250, 250)
        pixmap.fill(QColor(COLORS['bg_dark']))
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Draw a simple bottle cap shape
        painter.setBrush(QBrush(QColor(color)))
        painter.setPen(QColor(COLORS['text_dim']))
        
        center = QPoint(125, 125)
        painter.drawEllipse(center, 80, 80)
        
        painter.setBrush(QBrush(QColor(COLORS['bg_dark'])))
        painter.drawEllipse(center, 50, 50)
        
        painter.end()
        
        label.setPixmap(pixmap)
    
    @staticmethod
    def get_defect_image(item_id: str):
        """
        Get the captured snapshot image for a defective item.
        
        In production, this would load from the snapshots directory.
        For now, returns a mock image.
        
        Returns:
            QPixmap or None
        """
        # In production:
        # snapshot_path = os.path.join(
        #     SESSION['snapshot_dir'], 
        #     f"{item_id}.jpg"
        # )
        # if os.path.exists(snapshot_path):
        #     return QPixmap(snapshot_path)
        
        return None


# =============================================================================
# Metric Card Widget (reusable)
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
                border: 2px solid {COLORS['border']};
                border-radius: 8px;
            }}
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(4)
        
        self.label = QLabel(label.upper())
        self.label.setObjectName("metric_label")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = QFont(FONTS['metric_label'][0], FONTS['metric_label'][1], 
                     QFont.Weight.Bold)
        self.label.setFont(font)
        
        self.value_label = QLabel(value)
        self.value_label.setObjectName("metric_value")
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = QFont(FONTS['metric_value'][0], FONTS['metric_value'][1],
                     QFont.Weight.Bold)
        self.value_label.setFont(font)
        self.value_label.setStyleSheet(f"color: {value_color};")
        
        layout.addWidget(self.label)
        layout.addWidget(self.value_label)
    
    def update_value(self, value: str, color: str = None):
        """Update the displayed value."""
        self.value_label.setText(value)
        if color:
            self.value_label.setStyleSheet(f"color: {color};")


# =============================================================================
# Chart Widget (using matplotlib)
# =============================================================================

class ChartWidget(QWidget):
    """
    Widget for displaying matplotlib charts.
    
    Supports pie charts and line charts for defect analysis.
    """
    
    def __init__(self, chart_type: str = "pie", parent=None):
        super().__init__(parent)
        self.setMinimumHeight(250)
        self.setStyleSheet(f"background-color: {COLORS['bg_card']}; border-radius: 4px;")
        
        self.chart_type = chart_type
        self._figure = None
        self._axes = None
        
        # Initialize matplotlib figure
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        
        self._figure = plt.Figure(
            figsize=(5, 3),
            dpi=100,
            facecolor=COLORS['bg_card'],
            edgecolor='none'
        )
        self._axes = self._figure.add_subplot(111)
        
        # Apply styling
        self._style_axes()
        
        # Set initial empty chart
        if chart_type == "pie":
            self._axes.pie([1], labels=['No Data'], 
                          colors=[COLORS['text_dim']],
                          autopct='%1.1f%%')
        else:
            self._axes.plot([], [])
        
        self._figure.tight_layout()
        
        # Create canvas
        from matplotlib.backends.backend_qt import FigureCanvasQTAgg as FigureCanvas
        self._canvas = FigureCanvas(self._figure)
        self._canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding
        )
        
        layout = QVBoxLayout(self)
        layout.addWidget(self._canvas)
    
    def _style_axes(self):
        """Apply dark theme styling to axes."""
        if self._axes is None:
            return
        
        self._axes.set_facecolor(COLORS['bg_card'])
        self._axes.tick_params(colors=COLORS['text_secondary'])
        self._axes.spines['bottom'].set_color(COLORS['border'])
        self._axes.spines['top'].set_color(COLORS['border'])
        self._axes.spines['left'].set_color(COLORS['border'])
        self._axes.spines['right'].set_color(COLORS['border'])
        
        # Grid styling
        self._axes.grid(True, alpha=0.3, color=COLORS['border'])
    
    def update_chart(self, data, chart_type: str = None):
        """
        Update the chart with new data.
        
        Args:
            data: Chart data depending on type:
                  - pie: dict of {label: value}
                  - line: list of (x, y) tuples
            chart_type: Override chart type
        """
        chart_type = chart_type or self.chart_type
        
        # Clear existing chart
        self._axes.clear()
        self._style_axes()
        
        if chart_type == "pie":
            self._update_pie_chart(data)
        elif chart_type == "line":
            self._update_line_chart(data)
        
        self._figure.tight_layout()
        self._canvas.draw()
    
    def _update_pie_chart(self, data: dict):
        """Update pie chart with defect distribution data."""
        if not data or all(v == 0 for v in data.values()):
            self._axes.pie([1], labels=['No Defects Recorded'],
                          colors=[COLORS['text_dim']],
                          autopct='%1.1f%%')
            return
        
        # Filter out zero values
        filtered_data = {k: v for k, v in data.items() if v > 0}
        
        if not filtered_data:
            self._axes.pie([1], labels=['No Defects Recorded'],
                          colors=[COLORS['text_dim']],
                          autopct='%1.1f%%')
            return
        
        labels = list(filtered_data.keys())
        values = list(filtered_data.values())
        
        # Color palette for defect types
        colors = [
            '#E74C3C', '#C0392B', '#E67E22', '#F39C12',
            '#3498DB', '#2ECC71', '#9B59B6', '#1ABC9C'
        ]
        
        wedges, texts, autotexts = self._axes.pie(
            values,
            labels=labels,
            autopct='%1.1f%%',
            colors=colors[:len(values)],
            startangle=90,
            pctdistance=0.85
        )
        
        # Style text
        for text in texts:
            text.set_color(COLORS['text_primary'])
            text.set_fontsize(9)
        
        for autotext in autotexts:
            autotext.set_color(COLORS['text_primary'])
            autotext.set_fontsize(8)
            autotext.set_weight('bold')
    
    def _update_line_chart(self, data: list):
        """Update line chart with timeline data."""
        if not data:
            self._axes.plot([], [])
            self._axes.set_title("No Timeline Data")
            return
        
        # Extract x, y values
        x_values = [item[0] for item in data]
        y_values = [item[1] for item in data]
        
        # Plot
        self._axes.plot(
            x_values, y_values,
            color=COLORS['accent_red'],
            linewidth=2,
            marker='o',
            markersize=6,
            markerfacecolor=COLORS['accent_red'],
            markeredgecolor=COLORS['bg_card'],
            markeredgewidth=1
        )
        
        # Labels and title
        self._axes.set_xlabel("Session Time (minutes)", 
                             color=COLORS['text_secondary'])
        self._axes.set_ylabel("Defect Count", 
                             color=COLORS['text_secondary'])
        
        # Format x-axis ticks
        if len(x_values) > 1:
            self._axes.set_xticks(x_values)
            self._axes.set_xticklabels([f"{x:.0f}" for x in x_values])
        
        # Title
        self._axes.set_title("Defect Spike Timeline", 
                            color=COLORS['text_primary'],
                            fontweight='bold')
        
        # Fill under curve for visual effect
        self._axes.fill_between(x_values, y_values, 
                               alpha=0.2, color=COLORS['accent_red'])


# =============================================================================
# Alarm Banner Widget
# =============================================================================

class AlarmBanner(QLabel):
    """Flashing alarm banner for critical alerts."""
    
    def __init__(self, parent=None):
        super().__init__("CRITICAL DEFECT RATE DETECTED - AUDIT LINE", parent)
        self.setObjectName("alarm_banner")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setWordWrap(True)
        self.setVisible(False)
        
        # Set up flashing effect
        from PyQt6.QtWidgets import QGraphicsOpacityEffect
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


# =============================================================================
# Annotated Video Canvas
# =============================================================================

class AnnotatedVideoCanvas(QWidget):
    """Widget for displaying video frames with YOLO annotations."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(320, 240)
        self.setStyleSheet(f"background-color: {COLORS['bg_dark']};")
        
        self._frame = None
        self._fps_counter = QLabel(self)
        self._fps_counter.setObjectName("fps_counter")
        self._fps_counter.setText("FPS: 0")
        self._fps_counter.setAlignment(Qt.AlignmentFlag.AlignRight | 
                                        Qt.AlignmentFlag.AlignTop)
        
        # Scale factor for display
        self._display_scale = VISION['display_scale_factor']
        
        # Layout for FPS counter
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addStretch()
        layout.addWidget(self._fps_counter)
    
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
        
        frame = self._frame
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
        y = 20  # Offset to leave room for FPS counter
        painter.drawImage(QPoint(x, y), q_image)
