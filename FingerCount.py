import cv2
import time
import os
import HandTrack as htm
import time as t
import pandas as pd
from pynput.keyboard import  Key, Controller

wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

folderPath = "FingerImages"
myList = os.listdir(folderPath)
print(myList)
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    # print(f'{folderPath}/{imPath}')
    overlayList.append(image)

print(len(overlayList))
pTime = 0

detector = htm.handDetector(detectionCon=0.75)

tipIds = [4, 8, 12, 16, 20]

keyboard = Controller()

previousNumber = None
currentNumber = None
changedNumber = []
hasNumberChanged = False
roundedValue = False

def pressButton(button):
    global hasNumberChanged
    keyboard.press(button)
    keyboard.release(button)
    hasNumberChanged = False
    changedNumber.clear()
def pressEnter():
    global hasNumberChanged
    keyboard.press(Key.enter)
    keyboard.release(Key.enter)
    hasNumberChanged = False
    changedNumber.clear()

def typeLetters(number):
    global previousNumber
    global currentNumber
    global hasNumberChanged
    global roundedValue

    currentNumber = number
    print(roundedValue)

    if not previousNumber == currentNumber or hasNumberChanged:
        hasNumberChanged = True
        changedNumber.append(number)
        if len(changedNumber) >= 10:
            avg = sum(changedNumber) / len(changedNumber)
            del changedNumber[9]
            roundedValue = int(round(avg, 0))
            if roundedValue == 0:
                pressEnter()
            elif roundedValue == 1:
                pressButton('w')
            elif roundedValue == 2:
                pressButton('a')
            elif roundedValue == 3:
                pressButton('s')
            elif roundedValue == 4:
                pressButton('d')
            elif roundedValue == 5:
                pressButton('e')

    print(changedNumber)
    previousNumber = currentNumber



while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    # print(lmList)

    if len(lmList) != 0:
        fingers = []

        # Thumb
        if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # 4 Fingers
        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        # print(fingers)
        totalFingers = fingers.count(1)
        typeLetters(totalFingers)

        h, w, c = overlayList[totalFingers - 1].shape
        img[0:h, 0:w] = overlayList[totalFingers - 1]

        # cv2.rectangle(img, (20, 225), (170, 425), (0, 255, 0), cv2.FILLED)
        # cv2.putText(img, str(totalFingers), (45, 375), cv2.FONT_HERSHEY_PLAIN,
        #             10, (255, 0, 0), 25)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN,
                3, (255, 0, 0), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)