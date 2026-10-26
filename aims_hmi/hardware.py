"""
AIMS HMI Hardware Interface Module

Manages USB Serial communication with Arduino Uno and provides
mock mode fallback when no physical hardware is available.

Serial Protocol (from spec Section 5):
- Arduino -> PC: "TRIG\n" (cap detected), "ERR_MOTOR\n" (motor fault)
- PC -> Arduino: '0' (PASS), '1' (FAIL), 'STOP', 'START'
"""

import threading
import time
import serial
from typing import Optional, Callable, List
from collections import deque

from aims_hmi.config import SERIAL, AppSettings


class SerialMessage:
    """Represents a serial message with timestamp."""
    
    def __init__(self, message: bytes, timestamp: float = None):
        self.message = message
        self.timestamp = timestamp or time.time()
        self.parsed = message.decode('utf-8', errors='ignore').strip()
    
    def __repr__(self):
        return f"SerialMessage({self.parsed!r})"


class SerialManager:
    """
    Manages USB Serial communication with Arduino Uno.
    
    Features:
    - Automatic fallback to mock mode if port unavailable
    - Thread-safe message queue for incoming messages
    - Callback system for TRIG and error messages
    - Command sending with error handling
    """
    
    def __init__(self, settings: AppSettings):
        self.settings = settings
        self._serial: Optional[serial.Serial] = None
        self._mock_mode: bool = settings.use_mock_serial
        self._running: bool = False
        self._lock = threading.Lock()
        
        # Message queue for incoming messages
        self.message_queue: deque = deque(maxlen=1000)
        
        # Callbacks
        self.on_trig: Optional[Callable[[], None]] = None
        self.on_error: Optional[Callable[[str], None]] = None
        self.on_connect: Optional[Callable[[bool], None]] = None
        self.on_disconnect: Optional[Callable[[], None]] = None
        
        # Statistics
        self.trigs_received: int = 0
        self.errors_received: int = 0
        self.commands_sent: int = 0
        
        # Mock mode simulation
        self._mock_trigger_count: int = 0
        self._mock_thread: Optional[threading.Thread] = None
        
        # Connection status
        self.connected: bool = False
        self.port_info: str = "Not connected"
    
    def connect(self) -> bool:
        """
        Attempt to connect to the serial port.
        
        Returns:
            True if connected (real or mock), False on failure.
        """
        if self._running:
            self.disconnect()
        
        # Try real serial connection first
        if not self._mock_mode:
            try:
                self._serial = serial.Serial(
                    port=self.settings.serial_port,
                    baudrate=self.settings.serial_baudrate,
                    timeout=SERIAL["timeout"],
                    write_timeout=SERIAL["write_timeout"]
                )
                self.connected = True
                self.port_info = f"{self.settings.serial_port} @ {self.settings.serial_baudrate}"
                if self.on_connect:
                    self.on_connect(True)
                self._running = True
                self._start_listener_thread()
                return True
            except (serial.SerialException, OSError) as e:
                # Fall back to mock mode
                print(f"Serial connection failed: {e}. Falling back to mock mode.")
                self._mock_mode = True
        
        # Start in mock mode
        return self._start_mock_mode()
    
    def _start_mock_mode(self) -> bool:
        """Start mock serial mode for testing without hardware."""
        self.connected = True
        self._mock_mode = True
        self.port_info = f"{self.settings.serial_port} (MOCK) @ {self.settings.serial_baudrate}"
        self._running = True
        
        if self.on_connect:
            self.on_connect(True)
        
        # Start mock trigger generator in background
        self._mock_thread = threading.Thread(target=self._mock_trigger_loop, daemon=True)
        self._mock_thread.start()
        
        return True
    
    def _start_listener_thread(self):
        """Start the serial listener thread (real mode only)."""
        if self._mock_mode:
            return
        
        listener_thread = threading.Thread(target=self._listen_loop, daemon=True)
        listener_thread.start()
    
    def _listen_loop(self):
        """Main loop for reading serial messages (real mode)."""
        while self._running and self._serial and self._serial.is_open:
            try:
                if self._serial.in_waiting > 0:
                    message = self._serial.readline()
                    if message:
                        self._handle_message(SerialMessage(message))
                else:
                    time.sleep(0.01)  # Small sleep to prevent CPU spinning
            except serial.SerialException:
                self._handle_disconnect()
                break
            except Exception as e:
                print(f"Serial read error: {e}")
                time.sleep(0.1)
    
    def _mock_trigger_loop(self):
        """
        Simulate incoming TRIG messages in mock mode.
        
        Generates TRIG messages at random intervals to simulate
        bottle caps passing the IR sensor on the conveyor.
        """
        while self._running:
            # Random interval between 0.5 and 3 seconds (simulated caps)
            import random
            interval = random.uniform(0.5, 3.0)
            time.sleep(interval)
            
            if not self._running:
                break
            
            # Simulate TRIG message
            self._handle_message(SerialMessage(SERIAL["MSG_TRIG"]))
            self._mock_trigger_count += 1
    
    def _handle_message(self, msg: SerialMessage):
        """Process an incoming serial message."""
        with self._lock:
            self.message_queue.append(msg)
        
        parsed = msg.parsed
        
        if parsed == "TRIG":
            self.trigs_received += 1
            if self.on_trig:
                self.on_trig()
        elif parsed == "ERR_MOTOR":
            self.errors_received += 1
            if self.on_error:
                self.on_error("Motor fault detected")
        else:
            # Unknown message - log it
            pass
    
    def _handle_disconnect(self):
        """Handle unexpected disconnection."""
        self.connected = False
        self._running = False
        self.port_info = "Disconnected"
        if self.on_disconnect:
            self.on_disconnect()
    
    def disconnect(self):
        """Disconnect from serial port and clean up."""
        self._running = False
        
        if self._serial and self._serial.is_open:
            try:
                self._serial.close()
            except serial.SerialException:
                pass
        
        self.connected = False
        self.port_info = "Not connected"
        
        if self.on_disconnect:
            self.on_disconnect()
    
    def send_command(self, command: bytes) -> bool:
        """
        Send a command to the Arduino.
        
        Args:
            command: Command bytes (e.g., b'0', b'1', b'STOP', b'START')
        
        Returns:
            True if sent successfully, False otherwise.
        """
        with self._lock:
            if self._mock_mode:
                # In mock mode, just log the command
                self.commands_sent += 1
                return True
            
            if not self._serial or not self._serial.is_open:
                return False
            
            try:
                self._serial.write(command)
                self._serial.flush()
                self.commands_sent += 1
                return True
            except serial.SerialException as e:
                print(f"Failed to send command: {e}")
                self._handle_disconnect()
                return False
    
    # Convenience methods for specific commands
    
    def send_pass(self) -> bool:
        """Send PASS command (servo stays at 0°)."""
        return self.send_command(SERIAL["CMD_PASS"])
    
    def send_fail(self) -> bool:
        """Send FAIL command (servo sweeps to 90°)."""
        return self.send_command(SERIAL["CMD_FAIL"])
    
    def send_stop(self) -> bool:
        """Send STOP command (halt conveyor)."""
        return self.send_command(SERIAL["CMD_STOP"])
    
    def send_start(self) -> bool:
        """Send START command (resume conveyor)."""
        return self.send_command(SERIAL["CMD_START"])
    
    def get_recent_messages(self, count: int = 10) -> List[SerialMessage]:
        """Get the most recent messages from the queue."""
        with self._lock:
            return list(self.message_queue)[-count:]
    
    def clear_queue(self):
        """Clear the message queue."""
        with self._lock:
            self.message_queue.clear()
    
    @property
    def is_mock_mode(self) -> bool:
        """Check if running in mock mode."""
        return self._mock_mode
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()


class MockCamera:
    """
    Generates synthetic video frames for testing without a real camera.
    
    Creates frames that simulate bottle caps on a conveyor belt with
    occasional defects for testing the YOLO inference pipeline.
    """
    
    def __init__(self, width: int = 640, height: int = 480):
        self.width = width
        self.height = height
        self.frame_count = 0
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._frame = None
        self._lock = threading.Lock()
        
        # Create a mock "golden template" image
        self._create_golden_template()
    
    def _create_golden_template(self):
        """Create a perfect bottle cap reference image."""
        import numpy as np
        import cv2
        
        # Create blank frame
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Add simulated conveyor belt background
        cv2.rectangle(frame, (0, self.height - 100), (self.width, self.height), 
                     (60, 60, 60), -1)
        
        # Draw a perfect bottle cap
        center_x = self.width // 2
        center_y = self.height // 2 - 20
        radius = min(self.width, self.height) // 6
        
        # Cap body (metallic silver)
        cv2.circle(frame, (center_x, center_y), radius, (192, 192, 192), -1)
        cv2.circle(frame, (center_x, center_y), radius - 5, (220, 220, 220), -1)
        
        # Rim
        cv2.circle(frame, (center_x, center_y), radius - 8, (180, 180, 180), 3)
        
        # Inner circle (opening)
        cv2.circle(frame, (center_x, center_y), radius - 15, (40, 40, 40), -1)
        
        # Subtle highlight
        cv2.circle(frame, (center_x - radius//3, center_y - radius//3), 
                   radius//4, (255, 255, 255), -1, cv2.LINE_AA)
        
        self._golden_template = frame
    
    def get_golden_template(self) -> Optional['np.ndarray']:
        """Get the golden template image."""
        import numpy as np
        with self._lock:
            return self._golden_template.copy() if self._golden_template is not None else None
    
    def start(self):
        """Start generating mock frames."""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._generate_frames, daemon=True)
        self._thread.start()
    
    def stop(self):
        """Stop frame generation."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
    
    def _generate_frames(self):
        """Generate mock frames simulating bottle caps on conveyor."""
        import numpy as np
        import cv2
        import random
        
        cap_position = 0
        cap_defect = False
        
        while self._running:
            # Create blank frame with conveyor background
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            
            # Conveyor belt texture
            cv2.rectangle(frame, (0, self.height - 80), (self.width, self.height), 
                         (70, 70, 70), -1)
            
            # Add belt lines
            for x in range(0, self.width, 40):
                cv2.line(frame, (x, self.height - 80), (x + 20, self.height), 
                        (80, 80, 80), 1)
            
            # Move cap position across screen
            cap_position += 5
            
            if cap_position > self.width + 50:
                cap_position = -50
                cap_defect = random.random() < 0.2  # 20% defect rate
            
            # Draw bottle cap
            center_x = int(cap_position)
            center_y = self.height // 2
            radius = min(self.width, self.height) // 8
            
            if cap_defect:
                # Draw defective cap
                defect_type = random.choice(['dent', 'scratch', 'missing_liner', 'misalignment'])
                
                # Base cap
                cv2.circle(frame, (center_x, center_y), radius, (180, 180, 180), -1)
                
                if defect_type == 'dent':
                    # Draw dent
                    cv2.circle(frame, (center_x + radius//3, center_y - radius//4), 
                              radius//5, (100, 100, 100), -1)
                elif defect_type == 'scratch':
                    # Draw scratch line
                    cv2.line(frame, (center_x - radius//2, center_y), 
                            (center_x + radius//3, center_y + radius//4), 
                            (50, 50, 50), 2)
                elif defect_type == 'missing_liner':
                    # Missing inner liner (odd color)
                    cv2.circle(frame, (center_x, center_y), radius - 15, 
                              (150, 50, 50), -1)
                elif defect_type == 'misalignment':
                    # Off-center inner circle
                    cv2.circle(frame, (center_x + radius//4, center_y), 
                              radius - 15, (40, 40, 40), -1)
            else:
                # Draw perfect cap
                cv2.circle(frame, (center_x, center_y), radius, (192, 192, 192), -1)
                cv2.circle(frame, (center_x, center_y), radius - 5, (220, 220, 220), -1)
                cv2.circle(frame, (center_x, center_y), radius - 8, (180, 180, 180), 3)
                cv2.circle(frame, (center_x, center_y), radius - 15, (40, 40, 40), -1)
            
            # Add some noise for realism
            noise = np.random.randint(0, 20, (self.height, self.width, 3), dtype=np.uint8)
            frame = cv2.add(frame, noise)
            
            with self._lock:
                self._frame = frame
            
            self.frame_count += 1
            time.sleep(0.016)  # ~60 FPS
    
    def read(self):
        """
        Read the latest frame (mimics cv2.VideoCapture.read()).
        
        Returns:
            Tuple of (success, frame)
        """
        import numpy as np
        with self._lock:
            if self._frame is not None:
                return True, self._frame.copy()
            return False, None
    
    def is_running(self) -> bool:
        """Check if frame generation is active."""
        return self._running
    
    def release(self):
        """Release resources."""
        self.stop()
