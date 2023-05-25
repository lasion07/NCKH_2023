import argparse
import json
import os
import time
from datetime import datetime

import cv2 as cv
from src.nckh2023 import Model

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("--source", required = True,
	help = "Path to the source to be used")
ap.add_argument("--model", type=str, default='yolov8m2-nckh2023.pt',
	help = "Select Model")
ap.add_argument("--log", type=int, default=0,
	help = "Write log")
ap.add_argument("--show", type=int, default=1,
	help = "Show visualized image")
ap.add_argument("--conf", type=float, default=0.8,
help = "Confidence score")
ap.add_argument("--iou", type=float, default=0.5,
help = "Confidence score")
args = vars(ap.parse_args())

src = str(args["source"]) # Path
write_log = args['log']
show_result = args['show']

# Predictor configs  
conf = args['conf']
iou = args['iou']

webcame = src.isnumeric()
image = cv.imread(src)
cap = None

if webcame:
    cap = cv.VideoCapture(int(src))
elif image is None:
    cap = cv.VideoCapture(src)
    if not cap.isOpened():
        print("Cannot open the source")
        quit()

# Define model
model = Model(f"models/{args['model']}", 1280, conf, iou, write_log, show_result)

if image is not None:
    # Infer image
    result = model.infer(image)

    # Display the resulting frame
    if show_result:
        cv.imshow('frame', result)
        cv.imwrite('result.jpg', result)

    cv.waitKey(20000)
else:
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Infer frame
        frame = model.infer(frame)
            
        # Display the resulting frame
        if show_result:
            cv.imshow('frame', frame)
            # cv.imwrite('result.jpg', frame)

        if cv.waitKey(20) == ord('q'):
            break

    # When everything done, release the capture
    cap.release()

cv.destroyAllWindows()

if model.log is not None:
    model.log.close()
