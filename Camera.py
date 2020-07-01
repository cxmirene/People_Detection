import os
import sys
import cv2 as cv
from Detect import Detect
from SGBM import SGBM
from time import time
import numpy as np

detector = Detect()
detector.Init_Net()
sgbm = SGBM()
sgbm.Init_SGBM()

Camera = cv.VideoCapture(1)
if not Camera.isOpened():
    print("Could not open the Camera")
    sys.exit()

ret, Fream = Camera.read()
cv.imwrite("Two.jpg",Fream)
os.system("./camera.sh")

def SegmentFrame(Fream):
    double = cv.resize(Fream, (640, 240), cv.INTER_AREA)
    left = double[0:240,0:320]
    right = double[0:240,320:640]
    return left, right

while(True):
    ret, Fream = Camera.read()
    if not ret:
        break
    start = time()
    # DoubleImage = cv.resize(Fream, (640, 240), cv.INTER_AREA)
    # LeftImage = DoubleImage[0:240,0:320]
    # RightImage = DoubleImage[0:240,320:640]
    LeftImage, RightImage = SegmentFrame(Fream)

    start2 = time()
    result_rect = detector.detect(LeftImage)
    print("Detect time:" + str(start2-start))
    print(result_rect)
    start3 = time()
    distance, disp = sgbm.Coordination(LeftImage, RightImage, result_rect)
    print("Coordination time:" + str(start3-start))
    coordinate = LeftImage.copy()
    for i in range(0,len(result_rect)):
        cv.rectangle(coordinate, (result_rect[i][0], result_rect[i][1]), (result_rect[i][2], result_rect[i][3]), (255, 255, 0), 4)
        cv.putText(coordinate, str(distance[i]),
                        (result_rect[i][0] + 20, result_rect[i][1] + 40),
                        cv.FONT_HERSHEY_SIMPLEX,
                        1,  # font scale
                        (255, 0, 255),
                        2)  # line type
    end = time()
    seconds = end - start
    fps = 1/seconds
    print( "Estimated frames per second : {0}".format(fps))
    # cv.imshow("left",LeftImage)
    # cv.imshow("right",RightImage)
    # cv.imshow("disp",disp)
    # cv.imshow("result", coordinate)
    disp = cv.cvtColor(disp, cv.COLOR_GRAY2BGR)
    htich1 = np.hstack((LeftImage, RightImage))
    htich2 = np.hstack((disp, coordinate))
    vtich = np.vstack((htich1, htich2))
    cv.imshow("result", vtich)

    if cv.waitKey(1)==ord('q'):
        break

sys.exit()
