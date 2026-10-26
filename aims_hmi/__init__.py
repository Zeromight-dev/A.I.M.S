"""
AIMS HMI - AI-driven Industrial Manufacturing Sorter
Human-Machine Interface Application

Tech Stack:
- PyQt6 for GUI framework
- OpenCV (cv2) for video capture at 60 FPS target
- ultralytics YOLO for inference in async worker thread
- pyserial for USB Serial communication with Arduino Uno
- matplotlib for end-of-session telemetry charts
- pandas for CSV export

Multi-Threading:
- Thread 1 (Main): GUI rendering, event handling, button clicks, tab navigation
- Thread 2 (VisionWorker): Reads cv2.VideoCapture, runs YOLO inference, pushes annotated frames
- Thread 3 (SerialWorker): Listens for USB Serial signals, triggers callbacks

State Machine:
- STATE_A: ACTIVE_RUN (Live Monitoring, Snapshot Inspection, Tracking Log, Hardware Telemetry)
- STATE_B: SESSION_REPORT (Post-Run Telemetry, Defect Pattern Analysis, Export Tools)
"""

__version__ = "1.0.0"
__author__ = "AIMS Development Team"
