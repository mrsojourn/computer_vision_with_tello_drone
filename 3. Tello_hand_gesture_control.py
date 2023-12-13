import cv2
import mediapipe as mp
from djitellopy import Tello
from utils.HandTrackingModule import HandDetector
from utils.FaceDetectionModule import FaceDetector

detectorH = HandDetector(maxHands=2, detectionCon=0.8)
detectorF = FaceDetector(minDetectionCon=0.5, modelSelection=1)

# cap = cv2.VideoCapture(0)

gesture = ''

tello = Tello()
tello.connect()
print(tello.get_battery())
tello.streamon()
# tello.takeoff()
# tello.move_up(80)

while True:
    # _, img = cap.read()
    img = tello.get_frame_read().frame

    img = detectorH.findHands(img)
    lmList, bbox = detectorH.findPosition(img)
    img, bboxsFace = detectorF.findFaces(img)

    if bboxsFace:
        x, y, w, h = bboxsFace[0]['bbox']
        detectRegion = x - 175 - 25, y - 75, 175, h + 75
        cv2.rectangle(img, detectRegion, (0, 0, 255), 2)
        if bbox and detectorH.handType() == 'Right':
            cx, cy = bbox['center']
            if detectRegion[0] < cx < detectRegion[0] + detectRegion[2] and \
                    detectRegion[1] < cy < detectRegion[1] + detectRegion[3]:
                cv2.rectangle(img, detectRegion, (0, 255, 0), 2)
                fingers = detectorH .fingersUp()
                if fingers == [1, 1, 1, 1, 1]:
                    gesture = 'Open hand'
                elif fingers == [0, 1, 0, 0, 0]:
                    gesture = 'Up'
                    tello.move_up(40)
                elif fingers == [0, 1, 1, 0, 0]:
                    gesture = 'Victory'
                    tello.move_down(40)
                elif fingers == [0, 1, 0, 0, 1]:
                    gesture = 'Spider Man'
                    tello.flip_left()
                elif fingers == [0, 0, 0, 0, 0]:
                    gesture = 'Stop'
                elif fingers == [0, 0, 0, 0, 1]:
                    gesture = 'Left'
                    tello.move_left(20)
                elif fingers == [1, 0, 0, 0, 0]:
                    gesture = 'Right'
                    tello.move_right(20)

                cv2.putText(img, f'{gesture}'.upper(), (detectRegion[0] + 15, detectRegion[1] + detectRegion[3] + 50), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 2)

    cv2.imshow('window', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        tello.land()
        break
cv2.destroyAllWindows()