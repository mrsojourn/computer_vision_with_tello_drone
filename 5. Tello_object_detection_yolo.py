import cv2
import numpy as np
from djitellopy import Tello

tello = Tello()
tello.connect()
print(tello.get_battery())
tello.streamon()

# cap = cv2.VideoCapture(0)

classNames = []
with open('utils/resources/coco.names', 'r') as f:
    classNames = f.read().splitlines()
# print(len(classNames))
print(classNames)

model = 'utils/resources/yolov3.weights'
config = 'utils/resources/yolov3.cfg'
net = cv2.dnn.readNet(model, config)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def findObject(outputs, img):
    height, width, c = img.shape
    bboxes = []
    confidences = []
    classIds = []
    for output in outputs:
        for detection in output:
            score = detection[5:]
            classId = np.argmax(score)
            conf = score[classId]
            if conf > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                bboxes.append([x, y, w, h])
                classIds.append(classId)
                confidences.append(float(conf))
    # print(len(bboxes), len(classIds))
    # Non maximum suppression (NMS)
    indices = cv2.dnn.NMSBoxes(bboxes, confidences, 0.5, 0.2)

    for i in indices:
        x, y, w, h = bboxes[i]
        label = str(classNames[classIds[i]]).upper()
        confidence = str(round(confidences[i], 2))
        color = (255, 0, 255)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label + ' ' + confidence, (x, y + 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)


while True:
    img = tello.get_frame_read().frame

    # _, img = cap.read()

    blob = cv2.dnn.blobFromImage(img, 1 / 255, (320, 320), (0, 0, 0), crop=False)
    net.setInput(blob)
    layerNames = net.getLayerNames()
    # print(len(layerNames))
    # print(net.getUnconnectedOutLayers())
    outputNames = [layerNames[i-1] for i in net.getUnconnectedOutLayers()]
    # print(outputNames)
    outputs = net.forward(outputNames)
    # print(outputs[0].shape, outputs[1].shape, outputs[2].shape)
    # outputs: cx cy w h conf prob_score_for_80_class

    findObject(outputs, img)

    cv2.imshow('my window', img)
    cv2.waitKey(1)