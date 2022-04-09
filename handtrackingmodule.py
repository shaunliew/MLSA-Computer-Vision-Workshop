import cv2
import mediapipe as mp
import time


class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        # initialize the handDetector, you may refer to HandDetection module in MediaPipe
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, 1, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils  # method to draw the landmarks and line

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)  # save the processed result

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:  # we have 21 landmarks
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)  # now it should draw the landmarks(red points)

        return img

    def findPosition(self, img, handNo=0, draw=True):

        lmList = []
        # need to make sure that there is landmarks detected
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):  # number and coordinates of each red point
                h, w, c = img.shape  # height, weight, and c for our image.
                cx, cy = int(lm.x * w), int(lm.y * h)  # pixel coordinates for each landmarks
                # label the landmarks
                lmList.append([id, cx, cy])
                if draw: # only for one hand
                    cv2.putText(img, str(id), (cx + 5, cy + 5), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)

        return lmList  # return landmarks


def main():
    pTime = 0  # previous time
    cTime = 0  # current time
    cap = cv2.VideoCapture(0)
    detector = handDetector()  # call the class
    while True:
        success, img = cap.read()
        img = detector.findHands(img)  # pass the image into the hand detector class
        lmList = detector.findPosition(img)
        if len(lmList) != 0:  # if we can detect the hands
            print(lmList[0])  # index is referring to id

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.imshow("Image", img)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break


if __name__ == "__main__":
    main()
