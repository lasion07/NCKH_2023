import os
import time
import argparse
import cv2 as cv
from src.lp_recognition import LicensePlateRecognizer
from src.nckh2023 import has_helmet, has_license_plate
from ultralytics import YOLO


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("--source", required = True,
	help = "Path to the source to be used")
ap.add_argument("--model", type=str, default='yolov8n2-nckh2023.pt',
	help = "Select Model")
ap.add_argument("--log", type=int, default=1,
	help = "Write log")
ap.add_argument("--show", type=int, default=1,
	help = "Show visualized image")
ap.add_argument("--conf", type=float, default=0.5,
help = "Confidence score")
ap.add_argument("--iou", type=float, default=0.5,
help = "Confidence score")
ap.add_argument("--slt", type=float, default=0.3,
help = "Stop line threshold")
ap.add_argument("--fontscale", type=float, default=1,
help = "Font size")
args = vars(ap.parse_args())

model = YOLO(f"models/{args['model']}") # Select YOLO model
recognizer = LicensePlateRecognizer()
src = str(args["source"]) # Path

# Predictor configs  
conf = args['conf']
iou = args['iou']

# Stop line
stop_line_threshold = args['slt']

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
unidentified_count = 0
thickness = 2
radius = 10

img = cv.imread(src)

# # Show result
# cv.imshow('Result', img)

results = model.predict(source=img, conf=conf, iou=iou, stream=True, save=False, show=False)
 
for result in results:
    boxes = result.boxes

    # Put objects into lists
    for box in result.boxes:
        if int(box.cls) == 1:
            # Get location
            x0_p = int(box.xyxy[0][0].item())
            y0_p = int(box.xyxy[0][1].item())
            x1_p = int(box.xyxy[0][2].item())
            y1_p = int(box.xyxy[0][3].item())

            # Crop image
            license_plate_image = img[y0_p-10:y1_p+10, x0_p-10:x1_p+10]

            license_plate_number = f'Unidentified_{unidentified_count}'

            # Recognize image
            start_time = time.time()
            try:
                license_plate_number = recognizer.predict(license_plate_image)
            except:
                print("Can not recognize this license plate")
                unidentified_count += 1
            end_time = time.time()

            print(f'Detected license plate "{license_plate_number}" after {(end_time - start_time)*1000} ms')

            # Visualize
            cv.putText(img, f'{license_plate_number}', (x0_p, y0_p - 5), font, fontScale, GREEN, thickness, cv.LINE_AA)
            cv.rectangle(img, (x0_p, y0_p), (x1_p, y1_p), GREEN, thickness)
            cv.rectangle(img, (int(img.shape[1] * .1), int(img.shape[0] * .3)), (int(img.shape[1] * .9), int(img.shape[0] * .9)), GREEN, thickness)

cv.imshow('Result', img)

# Save result
path = f'data/lp_recognized/result_{len(os.listdir("data/lp_recognized"))}.jpg'
cv.imwrite(path, img)
print("Saved result at", path)

cv.waitKey(2000)
cv.destroyAllWindows()