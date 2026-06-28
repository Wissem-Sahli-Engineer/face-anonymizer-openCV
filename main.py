# pyrefly: ignore [missing-import]
import cv2
from pathlib import Path
# pyrefly: ignore [missing-import]
import mediapipe as mp

# Reading the image
img_path = Path("data") / "test3.jpeg"
img = cv2.imread(str(img_path))

# detecting faces

FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
BaseOptions = mp.tasks.BaseOptions


options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path='blaze_face_short_range.tflite'),
    min_detection_confidence=0.5
)

with FaceDetector.create_from_options(options) as face_detection :
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)

    out = face_detection.detect(mp_image)

    print (out.detections)

    for detection in out.detections:

        bbox = detection.bounding_box

        x1 , y1 , w , h = bbox.origin_x , int(bbox.origin_y * 0.8) , bbox.width , bbox.height + int(bbox.origin_y * 0.2)

    
    cv2.rectangle(img , (x1, y1) , (x1+w , y1+h) , (0,255,0))
    cv2.imshow("Image", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    



"""
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""