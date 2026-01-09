#this uses imageTiler.py and detect.py to take the GIVEN IMAGE and return a full resolution FINAL IMAGE with detection boxes on it
from ultralytics import YOLO
import cv2
import numpy as np
import detect, imageTiler
import time

# runtime
currentTime = time.time()

# get full resolution image
IMAGEPATH = "images/1.jpg"
MODEL = YOLO('yolov8n.pt') # can use any CNN e.g.: yolov8n.pt, yolov8s.pt or custom trained pt
fullImg = cv2.imread(IMAGEPATH)

# use imageTiler.py to get all tiles as an array
tiles = imageTiler.main(IMAGEPATH)

# detect every tile and retrieve local and global offset coordinates
detections = 0
for tile, ygOffset, xgOffset in tiles:
    
    results = MODEL.predict(tile, iou=0.7, conf=0.7, device=0, verbose=False)
    coords = detect.showDetections(results) # get the local coords from detect function which extracts the results from that tile

    for box in coords:
        #grab top-left and bottom-right from coords
        x0 = int(box[0] + xgOffset)
        y0 = int(box[1] + ygOffset)
        x1 = int(box[2] + xgOffset)
        y1 = int(box[3] + ygOffset)

        #draw boxes on the original image
        cv2.rectangle(fullImg, (x0,y0), (x1,y1), (255,0,0), 1)
        detections+=1

finalTime = (time.time()-currentTime)

cv2.imshow("Full Resolution Detections", fullImg)
print(f"Processing complete (took {finalTime:.2f}s). There are {detections} detections. Press any key on the image window to close.")
cv2.imwrite("detectedImage.jpg", fullImg)
cv2.waitKey(0)
cv2.destroyAllWindows()