import cv2
import time
import os
import handtrackingmodule as htm

wCam, hCam = 640, 480
cap = cv2.VideoCapture(0)  # webcam choice
cap.set(3, wCam)  # set the width of camera
cap.set(4, hCam)  # set the height of camera

# read the images file
folderPath = "FingerImages"
myList = os.listdir(folderPath)
myList.sort()
# print(myList)

overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    # print(f'{folderPath}/{imPath}')
    overlayList.append(image)

# print(len(overlayList))

pTime = 0

detector = htm.handDetector(maxHands=1)

tipIds = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:  # if hand is detected
        fingers = []
        # for thumb, use the x -coordinate
        if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        for id in range(1, 5):  # only for 4 fingers except thumb, use y-coordinate
            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:  # for all fingers
                fingers.append(1)
            else:
                fingers.append(0)
        # print(fingers)
        totalFingers = fingers.count(1)  # count the number of 1 existed
        print(totalFingers)

        h, w, c = overlayList[totalFingers - 1].shape  # more dynamic
        img[0:h, 0:w] = overlayList[totalFingers - 1]  # slicing

        cv2.rectangle(img, (20, 225), (170, 425), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(totalFingers), (45, 375), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 25)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord('q'):  # end this program when Q key is pressed
        break
