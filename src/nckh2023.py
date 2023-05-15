import os
import cv2 as cv
import time
import json
from ultralytics import YOLO

from src.lp_recognition import LicensePlateRecognizer



def has_helmet(x0_m, x1_m, y0_m, yc_m, xc_h, yc_h):
    """
        input: int, int, int, int, int, int
        output: bool
    """
    return x0_m < xc_h and xc_h < x1_m and y0_m < yc_h and yc_h < yc_m

def has_license_plate(x0_m, x1_m, yc_m, y1_m, xc_p, yc_p):
    """
        input: int, int, int, int, int, int
        output: bool
    """
    return x0_m < xc_p and xc_p < x1_m and yc_m < yc_p and yc_p < y1_m

def astype_int(array):
    """
        input: an old array
        output a new (int) array
    """
    temp = []
    for i in array:
        temp.append(int(i))
    return temp

def in_detection_zone(target_xy, zone_xyxy):
    """
        Check target in zone
        input: [int, int], [int, int, int, int]
        output: bool
    """
    return target_xy[0] > zone_xyxy[0] and target_xy[1] > zone_xyxy[1] and target_xy[0] < zone_xyxy[2] and target_xy[1] < zone_xyxy[3]

# font
font = cv.FONT_HERSHEY_SIMPLEX
font_scale = 1

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

class Model:
    def __init__(self, detector, conf, iou, write_log, show_result):
        self.detector = YOLO(detector)
        self.recognizer = LicensePlateRecognizer()

        self.conf = conf
        self.iou = iou

        # Detection_zone
        self.detection_zone_threshold = [.1, .3, .9, .9]

        self.write_log = write_log
        self.log_start_time = None
        self.log = None
        self.initial_log()
        self.show_result = show_result

    def initial_log(self):
        if self.write_log:
            date_time = datetime.today()
            self.log_start_time = date_time.strftime("%y-%m-%d_%H-%M") # Day Month Year Hour Minute

            os.mkdir(f"logs/{log_start_time}")

            with open(f"logs/{log_start_time}/log_{log_start_time}.json", mode='w') as f:
                f.close()

            self.log = open(f"logs/{log_start_time}/log_{log_start_time}.json", mode='a')

    def infer(self, frame):
        results = self.detector.predict(source=frame, conf=self.conf, iou=self.iou, stream=True, save=False, show=True)
    
        for result in results:
            boxes = result.boxes
            M = [] # Motorcyclists
            H = [] # Helmets
            P = [] # Liencse_plates
            mwh = [] # Motorcyclists with helmet
            mwoh = [] # Motorcyclists without helmet
            temp = [] # Temporary motorcyclists storing
            detection_zone_xyxy = astype_int([frame.shape[1] * self.detection_zone_threshold[0], frame.shape[0] * self.detection_zone_threshold[1], frame.shape[1] * self.detection_zone_threshold[2], frame.shape[0] * self.detection_zone_threshold[3]])

            unidentified_count = 0

            # Put objects into lists
            for box in result.boxes:
                """
                    0: helmet - green
                    1: license_plate - blue
                    2: motorcyclist - red
                """
                xc = int(box.xywh[0][0].item())
                yc = int(box.xywh[0][1].item())

                if int(box.cls) == 0:
                    H.append(box)
                elif int(box.cls) == 1:
                    P.append(box)
                else:
                    if box.conf < 0.5 or not in_detection_zone([xc, yc], detection_zone_xyxy):
                        continue                
                    M.append(box)

            # Check motorcyclists with helmet
            for motorcyclist in M:
                x0_m, y0_m, x1_m, y1_m = motorcyclist.xyxy[0]
                xc_m, yc_m, w_m, h_m = motorcyclist.xywh[0]

                for helmet in H:
                    xc_h, yc_h, w_h, h_h = helmet.xywh[0]
                
                    if has_helmet(x0_m, x1_m, y0_m, yc_m, xc_h, yc_h):
                        x0_h = int(helmet.xyxy[0][0].item())
                        y0_h = int(helmet.xyxy[0][1].item())
                        x1_h = int(helmet.xyxy[0][2].item())
                        y1_h = int(helmet.xyxy[0][3].item())

                        mwh.append(motorcyclist)

                        if self.show_result:
                            # Visualize Helmet detection
                            # print(f'\n\n{helmet.conf}\n\n')
                            cv.putText(frame, "Helmet", (x0_h, y0_h - 5), font, font_scale, BLUE, thickness, cv.LINE_AA)
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
                    xc_p, yc_p, w_p, h_p = license_plate.xywh[0]

                    # if license_plate.conf < 0.5 or w_p < 90 or h_p < 80:
                    #     continue

                    # Recognize license plate
                    if has_license_plate(x0_m, x1_m, yc_m, y1_m, xc_p, yc_p):
                        x0_p = int(license_plate.xyxy[0][0].item())
                        y0_p = int(license_plate.xyxy[0][1].item())
                        x1_p = int(license_plate.xyxy[0][2].item())
                        y1_p = int(license_plate.xyxy[0][3].item())

                        # Crop image
                        license_plate_image = frame[y0_p-10:y1_p+10, x0_p-10:x1_p+10]

                        license_plate_number = f'Unidentified_{unidentified_count}'

                        # Recognize image  
                        try:
                            start_time = time.time()
                            # license_plate_number = self.recognizer.predict(license_plate_image)
                            end_time = time.time()
                            print(f'Detected license plate "{license_plate_number}" after {(end_time - start_time)*1000} ms')
                        except:
                            print("Can not recognize this license plate")
                            unidentified_count += 1

                        if self.write_log:
                            # Save information
                            motorcyclist_image = frame[int(y0_m):int(y1_m), int(x0_m):int(x1_m)]
                            motorcyclist_image_path = f'logs/{self.log_start_time}/{license_plate_number}.jpg'
                            cv.imwrite(motorcyclist_image_path, motorcyclist_image)
                            information = {"license plate": license_plate_number, "image": motorcyclist_image_path, "detection confidence": float(license_plate.conf)}
                            json.dump(information, self.log)

                        if self.show_result:
                            # Visualize License plate detection
                            cv.putText(frame, f'{license_plate_number}', (x0_p, y0_p - 5), font, font_scale, LIGHT_BLUE, thickness, cv.LINE_AA)
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
                
                if self.show_result:
                    cv.putText(frame, "With helmet", (x0, y0 - 5), font, font_scale, GREEN, thickness, cv.LINE_AA)
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

                if self.show_result:
                    cv.putText(frame, "Without helmet", (x0, y0 - 5), font, font_scale, RED, thickness, cv.LINE_AA)
                    cv.rectangle(frame, (x0, y0), (x1, y1), RED, thickness)
                    cv.circle(frame, (xc, yc), radius, YELLOW, -1)
        
        return frame
