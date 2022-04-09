import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils  # method to draw the landmarks and line

pTime = 0  # previous time
cTine = 0  # current time
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)  # save the processed result

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:  # we have 21 landmarks
            for id, lm in enumerate(handLms.landmark):  # number and coordinates of each red point
                # print(id, lm)  # print it out
                h, w, c = img.shape  # height, weight, and c for our image.
                cx, cy = int(lm.x * w), int(lm.y * h)  # pixel coordinates for each landmarks
                #print(id, cx, cy)
                #label the landmarks
                cv2.putText(img, str(id), (cx+5, cy+5), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)
            mpDraw.draw_landmarks(img, handLms,
                                  mpHands.HAND_CONNECTIONS)  # now it should draw the landmarks(red points)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
