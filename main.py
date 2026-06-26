# pyrefly: ignore [missing-import]
import cv2
from pathlib import Path
# pyrefly: ignore [missing-import]
import mediapipe as mp

# Reading the image
img_path = Path("data") / "test1.jpeg"
img = cv2.imread(str(img_path))

# detecting faces

mp_face_detection = mp.solutions.face_detection

with mp_face_detection.FaceDetection(model_selection = 0, min_detection_confidence = 0.8) as face_detection :
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = face_detection.process(img)

    print(out.detections)



"""
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""