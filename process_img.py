import os
from pathlib import Path
# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
import mediapipe as mp

def process_img(img,face_detection):

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)

    out = face_detection.detect(mp_img)

    print(" ======= DETECTIONS ======== ")
    print (out.detections)

    if out.detections:

        for detection in out.detections:

            bbox = detection.bounding_box

            x1 , y1 , w , h = bbox.origin_x , int(bbox.origin_y * 0.8) , bbox.width , bbox.height + int(bbox.origin_y * 0.2)

            cv2.rectangle(img , (x1, y1) , (x1+w , y1+h) , (0,255,0),2)


            # blurring the face 

            img[y1:y1+h,x1:x1+w] = cv2.blur(img[y1:y1+h,x1:x1+w],(200,200))

        img= cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        return img
