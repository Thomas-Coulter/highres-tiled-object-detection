# dec 27-2025 - updated code, Thomas
import numpy as np
from ultralytics import YOLO
import cv2
import time

def showDetections(results):
    detect = results[0].boxes.shape                     

    if detect[0] > 0 : #at least 1 detection
        coords_GPU = results[0].boxes.data          # eg. tensor([[310.7162,  95.0658, 639.3807, 636.6490]], device='cuda:0')
        npcpu = coords_GPU.cpu()                    #  get data from cuda
        coords = npcpu.numpy()                      # convert to array, size = +6 for each detection (kinda just making it readable data)
        return coords[:,0:4]
    #in case of no detections
    return np.array([])

    #             COORDS NOW HAS THE OBJECT DETECTION DATA IN THE FORMAT:   
    #             COORDS = [ x0, y0, x1, y2, conf, class ], (PERSON = 0, CAR = 2, TEDDYBEAR = 77 ETC.) <-- you find these from the COCO data set (COmmon objects in COntext))
    #                      [ x0, y0, x1, y2, conf, class ],
    #                      [ x0, y0, x1, y2, conf, class ]
    #               *** where x0,y0 is the top left of the box! ***
    
        #print("NUMBER OF DETECTIONS = ",detect[0])
        #print("----------------------------------------")
        #print("######## COORDS ##########")
        #for i in range(detect[0]):
            #print("DETECTION:",i,": ",end='')
            #print(coords[i])

#--TEST AREA-- 
if __name__ == "__main__":

    model = YOLO('8n300.pt')
    images = ["1.jpg", "2.jpg", "3.jpg", "4.jpg"]

    for img_path in images:
        #Run prediction WITHOUT show=True, use numpy array instead and cv2 wait func
        results = model.predict(img_path, iou=0.7, conf=0.5, device=0, verbose=False)
        #print(results[0])
        # Get the image with boxes drawn on it (the "plotted" image)
        # plot() creates a standard numpy image array
        annotated_frame = results[0].plot()
        
        # 3. Use OpenCV to show it in a window we control
        cv2.imshow("YOLO Detection"+img_path, annotated_frame)
        print(f"Showing detections for: {img_path}.")
        showDetections(results)
        
    cv2.waitKey(0)
    cv2.destroyAllWindows()