import cv2, math
import mediapipe as mp
from utils.PoseDetectorModule import PoseDetector
from djitellopy import Tello

# mpBody = mp.solutions.pose
# body = mpBody.Pose()
# mpDraw = mp.solutions.drawing_utils
detector = PoseDetector(upBody=True)

# cap = cv2.VideoCapture(0)

def calAngle(lmList, p1, p2, p3, draw=True):
    if len(lmList) != 0:
        x1, y1 = lmList[p1][1:]
        x2, y2 = lmList[p2][1:]
        x3, y3 = lmList[p3][1:]
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
        if angle < 0: angle += 360
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.line(img, (x2, y2), (x3, y3), (255, 0, 255), 2)
            cv2.circle(img, (x1, y1), 5, (255, 255, 0), cv2.FILLED)
            cv2.circle(img, (x2, y2), 5, (255, 255, 0), cv2.FILLED)
            cv2.circle(img, (x3, y3), 5, (255, 255, 0), cv2.FILLED),
            cv2.putText(img, str(int(angle)), (x2 - 20, y2 - 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        return angle
    return 0

def calcDist(lmList, p1, p2, draw=True):
    if len(lmList) != 0:
        x1, y1 = lmList[p1][1:]
        x2, y2 = lmList[p2][1:]
        dist = math.hypot(x2 - x1, y2 - y1)
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 4)
            cv2.circle(img, (x1, y1), 5, (255, 255, 0), cv2.FILLED)
            cv2.circle(img, (x2, y2), 5, (255, 255, 0), cv2.FILLED)
            cv2.putText(img, str(int(dist)), (x2 + 20, y2 + 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
        return dist
    return 0

tello = Tello()
tello.connect()
print(tello.get_battery())
tello.streamon()
# tello.takeoff()
# tello.move_up(80)

while True:
    # _, img = cap.read()
    lr, fb, ud, rot = 0, 0, 0, 0
    img = tello.get_frame_read().frame
    img = cv2.flip(img, 1)

    img = detector.findPose(img, draw=False)
    lmList, bboxInfo = detector.findPosition(img, bboxWithHands=False)
    if lmList != 0:
        angleR = calAngle(lmList, 13, 11, 23, draw=False)
        angleL = calAngle(lmList, 24, 12, 14, draw=False)
        elbowR = calAngle(lmList, 15, 13, 11)
        elbowL = calAngle(lmList, 12, 14, 16)
        distR = calcDist(lmList, 12, 15)
        distL = calcDist(lmList, 11, 16)

        if 80 < angleR < 110 and 80 < angleL < 110:
            cv2.putText(img, 'T Pose', (50, 60), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)
            fb = 40
        elif 165 < angleR < 185 and 165 < angleL < 185:
            cv2.putText(img, 'Up', (50, 60), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)
            ud = 40
        elif 110 < elbowR < 130 and 110 < elbowL < 130:
            cv2.putText(img, 'Hippie', (50, 60), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)
            ud = -40
        elif distR:
            if distR < 60 and distL < 60:
                cv2.putText(img, 'Cross Arm', (50, 60), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)
                fb = -40

    tello.send_rc_control(lr, fb, ud, rot)
    cv2.imshow('window', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# cap.release()
cv2.destroyAllWindows()