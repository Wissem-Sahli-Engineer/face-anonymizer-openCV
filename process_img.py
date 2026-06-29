import os
from pathlib import Path
# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
import mediapipe as mp

def process_img(img, face_detection):
    # Convert to RGB for MediaPipe face detection
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    out = face_detection.detect(mp_img)

    print(" ======= DETECTIONS ======== ")
    print(out.detections)

    if out.detections:
        h_img, w_img, _ = img.shape
        for detection in out.detections:
            bbox = detection.bounding_box
            x1 = max(0, bbox.origin_x)
            y1 = max(0, int(bbox.origin_y * 0.8))
            w = min(bbox.width, w_img - x1)
            h = min(bbox.height + int(bbox.origin_y * 0.2), h_img - y1)

            if w > 0 and h > 0:
                cv2.rectangle(img, (x1, y1), (x1+w, y1+h), (0, 255, 0), 2)
                # blurring the face
                img[y1:y1+h, x1:x1+w] = cv2.blur(img[y1:y1+h, x1:x1+w], (200, 200))

    return img
