import os
from pathlib import Path
# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
import mediapipe as mp
import argparse as ap
from process_img import process_img


# Paths

img_path = str(Path("data") / "test3.jpeg")
video_path = str(Path("data") / "us.MOV")

# user choices

args = ap.ArgumentParser()

args.add_argument("--mode", default='webcam')
args.add_argument("--filePath",default= video_path)

args = args.parse_args()

# detecting faces

FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
BaseOptions = mp.tasks.BaseOptions

options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path='blaze_face_short_range.tflite'),
    min_detection_confidence=0.5
)

with FaceDetector.create_from_options(options) as face_detection :
    
    if not os.path.exists("./output"):
        os.makedirs("output")
        
    # Image

    if args.mode == 'image':
        img = cv2.imread(args.filePath)

        img = process_img(img,face_detection)

        cv2.imshow("Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # saving it 

        cv2.imwrite((Path("output")/"new_test3.jpeg"),img)

        print("---- Image Saved! -----")


    # Video

    elif args.mode == 'video':

        cap = cv2.VideoCapture(args.filePath)
        
        ret , frame = cap.read()

        fps = 25
        fps = cap.get(cv2.CAP_PROP_FPS)


        output_video = cv2.VideoWriter((Path("output")/"output.mp4"),
                                        cv2.VideoWriter_fourcc(*'mp4v'),
                                        fps,
                                        (frame.shape[1], frame.shape[0])
                                        )

        while True:
            frame = process_img(frame, face_detection)

            output_video.write(frame)

            ret , frame = cap.read()

            if not ret or frame is None:
                print("\n Video reached the end or frame could not be read. Exiting loop.\n")
                break
        
        print("=======> fps :\n ",fps)
        cap.release()
        output_video.release()
        cv2.destroyAllWindows()

    # Webcam

    elif args.mode == 'webcam':

        webcam = cv2.VideoCapture(0)
        
        ret , frame = webcam.read()

        while ret:
            
            frame = process_img(frame, face_detection)

            frame = cv2.flip(frame, 1) 
            
            cv2.imshow('Live Face Anonymizer',frame)

            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

            ret , frame = webcam.read()

        webcam.release()
        cv2.destroyAllWindows()

