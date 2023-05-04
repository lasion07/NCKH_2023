import argparse
import json
import os
import time
from datetime import datetime

import cv2 as cv
from src.lp_recognition import LicensePlateRecognizer
from src.nckh2023 import has_helmet, has_license_plate, astype_int, in_detection_zone

from ultralytics import YOLO

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("--source", required = True,
	help = "Path to the source to be used")
ap.add_argument("--model", type=str, default='yolov8n-nckh2023.pt',
	help = "Select Model")
ap.add_argument("--log", type=int, default=0,
	help = "Write log")
ap.add_argument("--show", type=int, default=1,
	help = "Show visualized image")
ap.add_argument("--conf", type=float, default=0.5,
help = "Confidence score")
ap.add_argument("--iou", type=float, default=0.5,
help = "Confidence score")
ap.add_argument("--fontscale", type=float, default=1,
help = "Font size")
args = vars(ap.parse_args())

model = YOLO(f"models/{args['model']}") # Select YOLO model
recognizer = LicensePlateRecognizer()
src = str(args["source"]) # Path
write_log = args['log']
show_result = args['show']

if write_log:
    date_time = datetime.today()
    log_start_time = date_time.strftime("%y-%m-%d_%H-%M") # Day Month Year Hour Minute

    os.mkdir(f"logs/{log_start_time}")

    with open(f"logs/{log_start_time}/log_{log_start_time}.json", mode='w') as f:
        f.close()

    f = open(f"logs/{log_start_time}/log_{log_start_time}.json", mode='a')

webcame = src.isnumeric()

if webcame:
    cap = cv.VideoCapture(int(src))
else:
    cap = cv.VideoCapture(src)

if not cap.isOpened():
    print("Cannot open camera")
    quit()

# Predictor configs  
conf = args['conf']
iou = args['iou']

# Detection_zone
detection_zone_threshold = []

# font
font = cv.FONT_HERSHEY_SIMPLEX

# fontScale
fontScale = args['fontscale']

# Color (BGR)
WHITE = (255, 255, 255)
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
LIGHT_BLUE = (238, 129, 49)
YELLOW = (0, 255, 255)

# Visualization settings
thickness = 2
radius = 10
cnt = 0

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    results = model.predict(source=frame, conf=conf, iou=iou, stream=True, save=False, show=False)
 
    for result in results:
        boxes = result.boxes
        M = [] # Motorcyclists
        H = [] # Helmets
        P = [] # Liencse_plates
        mwh = [] # Motorcyclists with helmet
        mwoh = [] # Motorcyclists without helmet
        temp = [] # Temporary motorcyclists storing
        detection_zone_xyxy = astype_int([frame.shape[1] * .1, frame.shape[0] * .3, frame.shape[1] * .9, frame.shape[0] * .9])

        unidentified_count = 0

        # Put objects into lists
        for box in result.boxes:
            """
                0: helmet - green
                1: license_plate - blue
                2: motorcyclist - red
            """
            if int(box.cls) == 0:
                H.append(box)
            elif int(box.cls) == 1:
                P.append(box)
            else:
                if box.conf < 0.5:
                    continue                
                M.append(box)

        # Check motorcyclists with helmet
        for motorcyclist in M:
            x0_m, y0_m, x1_m, y1_m = motorcyclist.xyxy[0]
            xc_m, yc_m, w_m, h_m = motorcyclist.xywh[0]

            if not in_detection_zone([xc_m, yc_m], detection_zone_xyxy):
                mwh.append(motorcyclist)
                break

            for helmet in H:
                xc_h, yc_h, w_h, h_h = helmet.xywh[0]
            
                if has_helmet(x0_m, x1_m, y0_m, yc_m, xc_h, yc_h):
                    x0_h = int(helmet.xyxy[0][0].item())
                    y0_h = int(helmet.xyxy[0][1].item())
                    x1_h = int(helmet.xyxy[0][2].item())
                    y1_h = int(helmet.xyxy[0][3].item())

                    mwh.append(motorcyclist)

                    # Visualize Helmet detection
                    cv.putText(frame, "Helmet", (x0_h, y0_h - 5), font, fontScale, BLUE, thickness, cv.LINE_AA)
                    cv.rectangle(frame, (x0_h, y0_h), (x1_h, y1_h), BLUE, thickness)
                    break
        
        # Check motorcyclists without helmet
        for motorcyclist in M:
            if not motorcyclist in mwh:
                mwoh.append(motorcyclist)
        
        # Check and recognize license plates of motorcyclists without helmet
        for motorcyclist in mwoh:
            x0_m, y0_m, x1_m, y1_m = motorcyclist.xyxy[0]
            xc_m, yc_m, w_m, h_m = motorcyclist.xywh[0]

            for license_plate in P:
                if license_plate.conf < 0.5:
                    continue

                xc_p, yc_p, w_p, h_p = license_plate.xywh[0]

                # Recognize license plate
                if has_license_plate(x0_m, x1_m, yc_m, y1_m, xc_p, yc_p):
                    x0_p = int(license_plate.xyxy[0][0].item())
                    y0_p = int(license_plate.xyxy[0][1].item())
                    x1_p = int(license_plate.xyxy[0][2].item())
                    y1_p = int(license_plate.xyxy[0][3].item())

                    # Save motorcyclist
                    # temp.append(motorcyclist)

                    # Crop image
                    license_plate_image = frame[y0_p-10:y1_p+10, x0_p-10:x1_p+10]

                    license_plate_number = f'Unidentified_{unidentified_count}'

                    # Recognize image
                    start_time = time.time()
                    try:
                        license_plate_number = recognizer.predict(license_plate_image)
                    except:
                        print("Can not recognize this license plate")
                        unidentified_count += 1
                    end_time = time.time()

                    if write_log:
                        # Save information
                        print(f'Detected license plate "{license_plate_number}" after {(end_time - start_time)*1000} ms')
                        motorcyclist_image = frame[int(y0_m):int(y1_m), int(x0_m):int(x1_m)]
                        motorcyclist_image_path = f'logs/{log_start_time}/{license_plate_number}.jpg'
                        cv.imwrite(motorcyclist_image_path, motorcyclist_image)
                        information = {"license plate": license_plate_number, "image": motorcyclist_image_path, "detection confidence": float(license_plate.conf)}
                        json.dump(information, f)

                    if show_result:
                        # Visualize License plate detection
                        cv.putText(frame, f'{license_plate_number}', (x0_p, y0_p - 5), font, fontScale, LIGHT_BLUE, thickness, cv.LINE_AA)
                        cv.rectangle(frame, (x0_p, y0_p), (x1_p, y1_p), LIGHT_BLUE, thickness)
                    break
        
        # Visualize
        ## Draw detection zone
        cv.rectangle(frame, (detection_zone_xyxy[0], detection_zone_xyxy[1]), (detection_zone_xyxy[2], detection_zone_xyxy[3]), WHITE, thickness, cv.LINE_8)
        ## Draw bounding box of motorcyclists with helmet
        for motorcyclist in mwh:
            x0 = int(motorcyclist.xyxy[0][0].item())
            y0 = int(motorcyclist.xyxy[0][1].item())
            x1 = int(motorcyclist.xyxy[0][2].item())
            y1 = int(motorcyclist.xyxy[0][3].item())
            xc = int(motorcyclist.xywh[0][0].item())
            yc = int(motorcyclist.xywh[0][1].item())
            
            if show_result:
                cv.putText(frame, "With helmet", (x0, y0 - 5), font, fontScale, GREEN, thickness, cv.LINE_AA)
                cv.rectangle(frame, (x0, y0), (x1, y1), GREEN, thickness)
                cv.circle(frame, (xc, yc), radius, YELLOW, -1)
                
        ## Draw bounding box of motorcyclists without helmet
        for motorcyclist in mwoh:
            x0 = int(motorcyclist.xyxy[0][0].item())
            y0 = int(motorcyclist.xyxy[0][1].item())
            x1 = int(motorcyclist.xyxy[0][2].item())
            y1 = int(motorcyclist.xyxy[0][3].item())
            xc = int(motorcyclist.xywh[0][0].item())
            yc = int(motorcyclist.xywh[0][1].item())

            if show_result:
                cv.putText(frame, "Without helmet", (x0, y0 - 5), font, fontScale, RED, thickness, cv.LINE_AA)
                cv.rectangle(frame, (x0, y0), (x1, y1), RED, thickness)
                cv.circle(frame, (xc, yc), radius, YELLOW, -1)
        
    # Display the resulting frame
    if show_result:
        cv.imshow('frame', frame)
    # cv.imwrite(f'result_{cnt}.jpg', frame)
    # cnt += 1
    if cv.waitKey(20) == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()

f.close()
