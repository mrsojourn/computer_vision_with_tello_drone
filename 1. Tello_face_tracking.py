import cv2
import mediapipe as mp
import time
import numpy as np
from djitellopy import Tello
from utils.PlotModule import LivePlot

width, height = 640, 480

xPID, yPID, zPID = [0.21, 0, 0.1], [0.27, 0, 0.1], [0.0021, 0, 0.1]
xTarget, yTarget, zTarget = width // 2, height // 2, 11500
pError, pTime, I = 0, 0, 0
myPlotX = LivePlot(yLimit=[-width // 2, width // 2], char='X')
myPlotY = LivePlot(yLimit=[-height // 2, height // 2], char='Y')
myPlotZ = LivePlot(yLimit=[-100, 100], char='z')

mpFaces = mp.solutions.face_detection
Faces = mpFaces.FaceDetection(min_detection_confidence=0.5, model_selection=1)
mpDraw = mp.solutions.drawing_utils

# cap = cv2.VideoCapture(0)

my_drone = Tello()
my_drone.connect()
print(my_drone.get_battery())
my_drone.streamoff()
my_drone.streamon()
# my_drone.takeoff()
# my_drone.move_up(80)


def PIDController(PID, img, target, cVal, limit=[-100, 100], pTime=0, pError=0, I=0, draw=False):
    """
    PIDController calculates the control value based on the PID algorithm.

    Args:
        PID (list): List of PID coefficients [P, I, D].
        img (numpy.ndarray): Input image.
        target (float): Target value.
        cVal (float): Current value.
        limit (list, optional): Control value limits. Defaults to [-100, 100].
        pTime (float, optional): Previous time. Defaults to 0.
        pError (float, optional): Previous error. Defaults to 0.
        I (float, optional): Integral term. Defaults to 0.
        draw (bool, optional): Flag to draw control value on image. Defaults to False.

    Returns:
        int: Control value.

    """
    t = time.time() - pTime
    error = target - cVal
    P = PID[0] * error
    I = I + (PID[1] * error * t)
    D = PID[2] * (error - pError) / t

    val = P + I + D
    val = float(np.clip(val, limit[0], limit[1]))
    if draw:
        cv2.putText(img, str(int(val)), (50, 70), cv2.FONT_HERSHEY_PLAIN, 4, (255, 0, 255), 3)

    pError = error
    pTime = time.time()

    return int(val)


while True:
    # _, img = cap.read()
    img = my_drone.get_frame_read().frame
    img = cv2.resize(img, (width, height))
    xVal, yVal, zVal = 0, 0, 0
    # -----------------------------------FACE DETECTION----------------------------------------- #
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = Faces.process(imgRGB)
    # print(results.detections)
    bboxs = []
    if results.detections:

        for id, detection in enumerate(results.detections):
            bboxC = detection.location_data.relative_bounding_box
            # print(bboxC)
            ih, iw, _ = img.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            cx, cy = bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2
            bboxInfo = {'id': id, 'bbox': bbox, 'score': detection.score, 'center': (cx, cy)}
            bboxs.append(bboxInfo)

            cv2.rectangle(img, bbox, (0, 255, 0), 2)
        cv2.putText(img, str(int(bboxs[0]['score'][0] * 100)) + ' %', (bbox[0] + 5, bbox[1] - 10),
                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
        cx, cy = bboxs[0]['center']
        x, y, w, h = bboxs[0]['bbox']
        area = w * h
        cv2.putText(img, str(area), (50, 200), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.circle(img, (cx, cy), 5, (255, 255, 0), cv2.FILLED)
        # print(bboxs)
        # ---------------------------------------------------------------------------------------- #

        cv2.line(img, (width // 2, 0), (width // 2, height), (255, 255, 255), 1)
        cv2.line(img, (width // 2, cy), (cx, cy), (255, 255, 255), 1)

        cv2.line(img, (0, height // 2),  (width, height // 2), (255, 255, 255), 1)
        cv2.line(img, (cx, height // 2), (cx, cy), (255, 255, 255), 1)

        xVal = PIDController(xPID, img, xTarget, cx)
        yVal = PIDController(yPID, img, yTarget, cy)
        zVal = PIDController(zPID, img, zTarget, area, limit=[-20, 15], draw=True)

        # t = time.time() - pTime
        # P = xPID[0] * errorX
        # I = I + (xPID[1] * errorX * t)
        # D = xPID[2] * (errorX - pError) / t
        #
        # xVal = int(P + I + D)
        # cv2.putText(img, str(xVal), (50, 70), cv2.FONT_HERSHEY_PLAIN, 4, (255, 0, 255), 3)
        # pTime = time.time()
        # pError = errorX
        # # print(error, xVal)

        imgPlotX = myPlotX.update(xVal)
        imgPlotY = myPlotY.update(yVal)
        imgPlotZ = myPlotZ.update(zVal)

        stackImg1 = np.hstack((img, imgPlotX))
        stackImg2 = np.hstack((imgPlotY, imgPlotZ))
        stackImg = np.vstack((stackImg1, stackImg2))
    else:
        blank = np.zeros((height,width, 3), np.uint8)
        stackImg1 = np.hstack((img, blank))
        stackImg2 = np.hstack((blank, blank))
        stackImg = np.vstack((stackImg1, stackImg2))

    # my_drone.send_rc_control(0, zVal, yVal, -xVal)
    my_drone.send_rc_control(0, zVal, yVal, -xVal)
    # print(zVal)
    cv2.imshow('Image', stackImg)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        my_drone.land()
        break
