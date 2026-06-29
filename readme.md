# Face Anonymizer

A simple Python application that detects and anonymizes (blurs) faces in real-time using OpenCV and MediaPipe. It supports processing images, videos, and live webcam feeds.

## Project Structure

```text
├── data/                         # Directory containing input media (images/videos)
├── output/                       # Directory where processed files are saved
├── blaze_face_short_range.tflite # MediaPipe face detection model
├── main.py                       # Main application entry point (handles modes & I/O)
├── process_img.py                # Image processing module (color conversions, detection, blurring)
├── requirements.txt              # Project dependencies
└── README.md                     # Project documentation
```

## File Overview

* **`main.py`**: Configures CLI arguments, loads the MediaPipe model, and handles the three operational modes (`image`, `video`, `webcam`).
* **`process_img.py`**: Contains the core logic to convert frame formats, feed frames to the face detector model, and draw/blur bounding boxes over detected faces.
* **`blaze_face_short_range.tflite`**: The lightweight model used by MediaPipe for face detection.

## Setup & Running

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run in Webcam Mode** (Default):
   ```bash
   python main.py --mode webcam
   ```

3. **Run on an Image**:
   ```bash
   python main.py --mode image --filePath data/test3.jpeg
   ```

4. **Run on a Video**:
   ```bash
   python main.py --mode video --filePath data/us.MOV
   ```
