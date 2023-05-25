import os
import argparse
import csv
import cv2 as cv
import time
import json
import numpy as np

from lp_recognition import LicensePlateRecognizer
from lp_detection import LicensePlateDetector


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("--csv", required = True,
	help = "Path to the csv to be used")
ap.add_argument("--model", type=str, default='yolov8m-nckh2023.pt',
	help = "Select Model")
ap.add_argument("--log", type=int, default=1,
	help = "Write log")
ap.add_argument("--show", type=int, default=1,
	help = "Show visualized image")
ap.add_argument("--conf", type=float, default=0.15,
help = "Confidence score")
ap.add_argument("--iou", type=float, default=0.5,
help = "Confidence score")
ap.add_argument("--slt", type=float, default=0.3,
help = "Stop line threshold")
ap.add_argument("--fontscale", type=float, default=1,
help = "Font size")
args = vars(ap.parse_args())

def put_text(img, text, loc, font, font_scale, color, thickness, lineType):
    (text_offset_x, text_offset_y) = loc
    # set the rectangle background to white
    WHITE = (255, 255, 255)
    # get the width and height of the text box
    (text_width, text_height) = cv.getTextSize(text, font, fontScale=font_scale, thickness=thickness)[0]
    # make the coords of the box with a small padding of two pixels
    box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height - 2))
    cv.rectangle(img, box_coords[0], box_coords[1], color, cv.FILLED)
    cv.putText(img, text, (text_offset_x, text_offset_y), font, fontScale=font_scale, color=WHITE, thickness=thickness, lineType=lineType)


lp_detector = LicensePlateDetector()
lp_recognizer = LicensePlateRecognizer()
src = str(args['csv']) # Path
root_folder = 'data/GreenParking'

# Predictor configs  
conf = args['conf']
iou = args['iou']

# Font
font = cv.FONT_HERSHEY_SIMPLEX

# Font scale
font_scale = args['fontscale']

# Color (BGR)
WHITE = (255, 255, 255)
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
LIGHT_BLUE = (238, 129, 49)
YELLOW = (0, 255, 255)

# Visualization settings
unidentified_count = 0
thickness = 2
radius = 10

# Counter
true_cnt = 0
false_cnt = 0

with open(src, mode='r') as f:
    reader = csv.reader(f)

    # for file in sorted(os.listdir(root_folder)):
    #     print(file, ',', sep='')
    # quit()

    for filename, label in reader:
        path = os.path.join(root_folder, filename)
        print(f'Read file {filename}')

        img = cv.imread(path)
        
        reg_result = False

        # Crop image
        license_plate_image = img

        # license_plate_image = img[y0_p:y1_p, x0_p:x1_p]

        license_plate_number = f'license_plate'

        try:
            # Detect license plate
            lp_image, lp_labels = lp_detector.detect(license_plate_image)
            cv.imwrite('lp.jpg', lp_image)

            # Recognize license plate
            start_time = time.time()
            license_plate_number = lp_recognizer.predict(lp_image)
            end_time = time.time()
            print(f'Detected license plate "{license_plate_number}" after {(end_time - start_time)*1000} ms')
        except:
            print("Can not recognize this license plate")

        if license_plate_number == label:
            reg_result = True
            true_cnt += 1
        else:
            false_cnt += 1

        # Visualize License plate detection
        put_text(img, f'{license_plate_number}', (0, img.shape[0] - 50), font, font_scale, LIGHT_BLUE, thickness, cv.LINE_AA)
        # cv.rectangle(img, (x0_p, y0_p), (x1_p, y1_p), LIGHT_BLUE, thickness)

        # if not reg_result:
        #     cv.imshow('Result', img)
        #     cv.imwrite('result.jpg', img)
        #     cv.waitKey(20000)

total = true_cnt + false_cnt
print(f'Total: {total}')
print(f'True count: {true_cnt} / {true_cnt/total}')
print(f'False count: {false_cnt} / {false_cnt/total}')
        
cv.destroyAllWindows()