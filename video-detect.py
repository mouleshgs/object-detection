from ultralytics import YOLO
import numpy as np
import torch
import cv2 as cv
import sys

cap = cv.VideoCapture(0)
model = YOLO('yolov8n.pt')


while True:
    res, frame = cap.read()

    if not res:
        sys.exit(34)

    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    results = model(frame)
    annotation_frame = results[0].plot()
    cv.imshow("camera", annotation_frame)



    if cv.waitKey(1) == ord('q'):
        cap.release()
        cv.destroyAllWindows()
        break