# AIMS - AI-driven Industrial Manufacturing Sorter (HMI)

Human-Machine Interface for the AIMS bottle-cap sorting line.

## What this is

A single-application HMI implementing the AIMS specification:

- **State A — Active Run:** live video feed with YOLO-style bounding-box overlays, metric cards (Total / Pass / Fail), a scrolling tracking log, an AI confidence threshold slider, and an alarm banner with acknowledge.
- **State B — End-of-Session Report:** executive metric cards (session time, units processed, yield, rejections), a defect distribution chart, and a defect-timeline spike chart, plus PDF/CSV export and "Start New Session".

## Implementation note

The specification permitted PyQt6/PySide6 **or** Streamlit. This workspace runs headless and does not have the Qt GUI runtime libraries available, so the HMI is shipped as a **Streamlit single-file app** (`aims_hmi/hmi_streamlit.py`).

Hardware integration (Arduino Uno serial + OpenCV camera + YOLO inference) is **simulated with in-process mocks** so the HMI is fully runnable here without a real camera, Arduino, or USB serial port:
- Mock serial generates `TRIG` messages at randomized intervals and accepts the PC->Arduino commands from the spec (`0`, `1`, `STOP`, `START`).
- Mock camera generates synthetic frames of bottle caps on a conveyor, including occasional defects.
- Mock inference produces realistic pass/fail labels, confidence scores, defect types, and annotated frames.

To use real hardware, point the app at a real camera index and a real pyserial port, and swap the mock inference for an ultralytics YOLO model.

## Run

Use the local venv:

```
.venv/bin/streamlit run aims_hmi/hmi_streamlit.py --server.headless true --server.port 3000 --server.address 0.0.0.0
```

Then open the local URL shown in the terminal (default `http://localhost:3000`).

When `QT_QPA_PLATFORM` is not available and you still want to validate the app without a display, the same command runs headless; the Streamlit server serves the app over HTTP and the UI is rendered in a browser.

## Files

- `aims_hmi/hmi_streamlit.py` — Streamlit HMI app (entry point for the Streamlit implementation).
- `aims_hmi/config.py` — shared UI/hardware/vision constants.
- `aims_hmi/hardware.py` — serial manager and mock camera helpers.
- `aims_hmi/workers.py` — threaded worker helpers and inference result types.
- `aims_hmi/views.py` — PyQt6 view classes (kept from the PyQt6 implementation for reference/reuse).
- `aims_hmi/exporters.py` — PDF report and CSV log export.
- `aims_hmi/app.py` — PyQt6 main-window implementation (kept from the PyQt6 implementation for reference/reuse).

## Shape of a session

1. Set Batch ID and Operator ID in the header.
2. Click **START SESSION**.
3. In active mode, the video feed animates; trigger TRIGs either from the simulated serial or by letting the mock serial run. The tracking log populates and metric cards update.
4. If 3 consecutive fails happen, or the defect rate exceeds 10%, the alarm banner appears; acknowledge it with the button.
5. Click **STOP SESSION** to enter the report view.
6. In the report view, review charts and export PDF/CSV, or start a new session.

## Ports

Default dev server: port 3000.

