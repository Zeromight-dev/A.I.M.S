"""
AIMS HMI Worker Threads Module

Contains background worker classes for:
1. VisionWorker: OpenCV frame capture + YOLO inference in async thread
2. InferenceResult: Data class for inference results
"""

import time
import threading
from typing import Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime
import random

import numpy as np
import cv2

from PyQt6.QtCore import QThread, pyqtSignal, QMutex

from aims_hmi.config import VISION, ALARM, SERIAL


# =============================================================================
# Inference Result Data Class
# =============================================================================

@dataclass
class InferenceResult:
    """Result of a YOLO inference run."""
    
    is_pass: bool
    confidence: float
    defect_type: Optional[str] = None
    bounding_box: Optional[tuple] = None  # (x1, y1, x2, y2)
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


# =============================================================================
# Vision Worker Thread
# =============================================================================

class VisionWorker(QThread):
    """
    Background worker for video capture and YOLO inference.
    
    This thread:
    1. Reads frames from the camera (OpenCV VideoCapture)
    2. Runs YOLO inference on frames when triggered
    3. Pushes annotated frames to the UI via signals
    4. Reports FPS statistics
    
    Signals:
        frame_ready: Emitted with annotated frame (numpy array)
        inference_result: Emitted when inference completes
        fps_updated: Emitted with current FPS count
    """
    
    # Signals for UI communication
    frame_ready = pyqtSignal(np.ndarray)
    inference_result = pyqtSignal(InferenceResult)
    fps_updated = pyqtSignal(float)
    
    def __init__(self, 
                 camera: Optional[Any] = None,
                 serial_manager: Optional[Any] = None,
                 confidence_threshold: int = VISION['threshold_default'],
                 mock_mode: bool = True):
        super().__init__()
        
        self.camera = camera
        self.serial_manager = serial_manager
        self.confidence_threshold = confidence_threshold
        self.mock_mode = mock_mode
        
        # Thread control
        self._running = False
        self._request_inference = threading.Event()
        self._inference_lock = QMutex()
        
        # FPS tracking
        self._frame_count = 0
        self._fps_start_time = time.time()
        self._current_fps = 0.0
        
        # Latest frame for display
        self._latest_frame: Optional[np.ndarray] = None
        
        # Mock inference state
        self._defect_probability = 0.15  # 15% base defect rate for mock
        
        # YOLO model (None = mock mode)
        self._model = None
    
    def set_confidence_threshold(self, value: int):
        """Update the confidence threshold."""
        self.confidence_threshold = value
    
    def request_inference(self):
        """Request inference on the latest frame."""
        self._request_inference.set()
    
    def run(self):
        """
        Main worker thread loop.
        
        Continuously captures frames and runs inference when triggered.
        """
        # Start camera if not already running
        if self.camera and hasattr(self.camera, 'start'):
            self.camera.start()
        
        self._running = True
        self._fps_start_time = time.time()
        self._frame_count = 0
        
        # Frame capture loop
        while self._running and not self.isInterruptionRequested():
            try:
                # Capture frame
                frame = self._capture_frame()
                
                if frame is not None:
                    self._latest_frame = frame
                    self._frame_count += 1
                    
                    # Calculate FPS
                    elapsed = time.time() - self._fps_start_time
                    if elapsed >= 1.0:
                        self._current_fps = self._frame_count / elapsed
                        self._fps_start_time = time.time()
                        self._frame_count = 0
                        self.fps_updated.emit(self._current_fps)
                    
                    # Check if inference requested
                    if self._request_inference.is_set():
                        self._request_inference.clear()
                        self._run_inference(frame)
                    
                    # Send frame for display
                    annotated = self._annotate_frame(frame)
                    self.frame_ready.emit(annotated)
                
                # Small sleep to prevent CPU spinning
                time.sleep(0.001)
                
            except Exception as e:
                print(f"VisionWorker error: {e}")
                time.sleep(0.1)
        
        # Cleanup
        self._running = False
        if self.camera and hasattr(self.camera, 'stop'):
            self.camera.stop()
    
    def _capture_frame(self) -> Optional[np.ndarray]:
        """
        Capture a frame from the camera.
        
        Returns:
            numpy array (BGR format) or None on failure
        """
        if self.mock_mode and self.camera:
            # Use mock camera
            success, frame = self.camera.read()
            if success and frame is not None:
                return frame
            return None
        
        # Real camera path (placeholder - would use cv2.VideoCapture)
        # In production, this would be:
        # cap = cv2.VideoCapture(0)
        # success, frame = cap.read()
        # return frame if success else None
        
        return None
    
    def _run_inference(self, frame: np.ndarray):
        """
        Run YOLO inference on a frame.
        
        In mock mode, simulates inference with random results.
        In production, would run the actual YOLO model.
        """
        if self.mock_mode:
            result = self._mock_inference(frame)
        else:
            result = self._real_inference(frame)
        
        if result:
            self.inference_result.emit(result)
    
    def _mock_inference(self, frame: np.ndarray) -> Optional[InferenceResult]:
        """
        Simulate YOLO inference for testing without a real model.
        
        Creates realistic-looking results with:
        - Random pass/fail classification
        - Simulated confidence scores
        - Random defect types for failures
        - Bounding box around the "bottle cap"
        """
        # Simulate processing delay
        time.sleep(0.05)
        
        # Determine if this is a pass or fail
        # Weight towards pass (85% pass rate in mock)
        is_pass = random.random() > self._defect_probability
        
        # Generate confidence score
        if is_pass:
            confidence = random.uniform(0.85, 0.99)
        else:
            confidence = random.uniform(0.60, 0.95)
        
        # Pick defect type if failing
        defect_type = None
        if not is_pass:
            defect_types = VISION['defect_types']
            defect_type = random.choice(defect_types)
        
        # Generate bounding box (simulate detection of bottle cap)
        h, w = frame.shape[:2]
        center_x = random.randint(w // 4, 3 * w // 4)
        center_y = random.randint(h // 4, 3 * h // 4)
        box_size = min(w, h) // 8
        x1 = max(0, center_x - box_size)
        y1 = max(0, center_y - box_size)
        x2 = min(w, center_x + box_size)
        y2 = min(h, center_y + box_size)
        bounding_box = (x1, y1, x2, y2)
        
        return InferenceResult(
            is_pass=is_pass,
            confidence=confidence,
            defect_type=defect_type,
            bounding_box=bounding_box
        )
    
    def _real_inference(self, frame: np.ndarray) -> Optional[InferenceResult]:
        """
        Run real YOLO inference.
        
        This would use the actual ultralytics YOLO model.
        Placeholder for production implementation.
        """
        # In production:
        # results = self._model(frame, verbose=False)
        # for result in results:
        #     boxes = result.boxes
        #     if len(boxes) > 0:
        #         # Process detection...
        #         pass
        
        # For now, fall back to mock
        return self._mock_inference(frame)
    
    def _annotate_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Annotate a frame with YOLO detection results.
        
        Draws:
        - Green bounding box for PASS detections
        - Red bounding box for FAIL detections
        - Text overlay with classification and confidence
        """
        annotated = frame.copy()
        h, w = annotated.shape[:2]
        
        # Add simulated annotation (for mock mode visualization)
        # In production, this would use actual detection results
        
        # Simulate a detection box for visual feedback
        box_x1 = w // 4
        box_y1 = h // 4
        box_x2 = 3 * w // 4
        box_y2 = 3 * h // 4
        
        # Alternate between pass/fail for visual demo
        is_pass = random.random() > 0.3
        
        if is_pass:
            color = VISION['color_pass']
            label = f"BOTTLE_CAP: PASS - {random.uniform(0.85, 0.99):.0%}"
        else:
            color = VISION['color_fail']
            defect = random.choice(VISION['defect_types'])
            label = f"BOTTLE_CAP: FAIL [{defect}] - {random.uniform(0.60, 0.89):.0%}"
        
        # Draw bounding box
        thickness = 3
        cv2.rectangle(annotated, (box_x1, box_y1), (box_x2, box_y2), color, thickness)
        
        # Draw text background
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        text_x = box_x1
        text_y = box_y1 - 10 if box_y1 > 30 else box_y2 + 30
        cv2.rectangle(annotated, 
                     (text_x, text_y - text_size[1] - 5),
                     (text_x + text_size[0] + 10, text_y + 5),
                     (0, 0, 0), -1)
        
        # Draw text
        cv2.putText(annotated, label, (text_x + 5, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        
        # Add timestamp overlay
        timestamp = datetime.now().strftime("%H:%M:%S")
        cv2.putText(annotated, f"TIME: {timestamp}", (10, h - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        
        return annotated
    
    def stop(self):
        """Stop the worker thread."""
        self._running = False
        self._request_inference.clear()
        self.requestInterruption()
        self.wait(2000)  # Wait up to 2 seconds for graceful shutdown
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        self.stop()


# =============================================================================
# Serial Worker (integrated into SerialManager)
# =============================================================================

class SerialListener:
    """
    Serial listener functionality.
    
    This is integrated into the SerialManager class in hardware.py.
    Provides callback-based listening for incoming serial messages.
    """
    
    def __init__(self, serial_manager: Any):
        self.serial_manager = serial_manager
        self._listening = False
        self._thread: Optional[threading.Thread] = None
    
    def start_listening(self, on_message: callable):
        """
        Start listening for serial messages.
        
        Args:
            on_message: Callback function(message) for incoming messages
        """
        if self._listening:
            return
        
        self._listening = True
        self.serial_manager.on_trig = on_message
        # Message handling is done in SerialManager._listen_loop()
    
    def stop_listening(self):
        """Stop listening for serial messages."""
        self._listening = False


# =============================================================================
# Inference Timer (for periodic inference without TRIG)
# =============================================================================

class PeriodicInferenceTimer:
    """
    Timer for running periodic inference (fallback when no TRIG received).
    
    Useful for continuous monitoring mode.
    """
    
    def __init__(self, interval_seconds: float = 1.0,
                 on_inference: callable = None):
        self.interval = interval_seconds
        self.on_inference = on_inference
        self._running = False
        self._thread: Optional[threading.Thread] = None
    
    def start(self):
        """Start the periodic timer."""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
    
    def stop(self):
        """Stop the periodic timer."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
    
    def _run(self):
        """Timer loop."""
        while self._running:
            time.sleep(self.interval)
            if self._running and self.on_inference:
                self.on_inference()
