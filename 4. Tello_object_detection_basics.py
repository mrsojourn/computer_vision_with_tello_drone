import cv2
import os
from djitellopy import Tello

tello = Tello()
tello.connect()
print(tello.get_battery())
tello.streamon()

path = r'utils/pics'  # Use raw string to avoid escape characters
myList = os.listdir(path)
myImages = []
myClasses = []
for img in myList:
    image = cv2.imread(os.path.join(path, img))  # Use os.path.join to handle file path
    myImages.append(image)
    myClasses.append(img.split('.')[0])

# cap = cv2.VideoCapture(1)

orb = cv2.ORB_create(nfeatures=1000)


def findDes(myImages):
    desList = []
    for i in myImages:
        k, d = orb.detectAndCompute(i, None)
        desList.append(d)
    return desList


desList = findDes(myImages)

def findIDs(img, desList, thres=15):
    k2, d2 = orb.detectAndCompute(img, None)
    bf = cv2.BFMatcher()
    lenMatches = []
    matchID = -1
    try:
        for des in desList:
            matches = bf.knnMatch(des, d2, k=2)
            good = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good.append([m])
            lenMatches.append(len(good))
    except:
        pass

    if len(lenMatches) != 0:
        if max(lenMatches) > thres:
            matchID = lenMatches.index(max(lenMatches))

    return matchID


while True:
    img = tello.get_frame_read().frame
    # _, img = cap.read()
    imgOrg = img.copy()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    matchID = findIDs(img, desList)
    if matchID != -1:
        cv2.putText(imgOrg, myClasses[matchID], (180, 460), cv2.FONT_HERSHEY_PLAIN, 4, (255, 0, 255), 4)
    cv2.imshow('Detect Image', imgOrg)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break