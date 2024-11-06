import cv2 as cv
import numpy as np
import mediapipe as mp
import time 


cap = cv.VideoCapture(0)
cap.set(3, 160)  # Width
cap.set(4, 120)  # Height

mpHands = mp.solutions.hands
hands = mpHands.Hands(False)

mpDraw = mp.solutions.drawing_utils

pTime = 0 
cTime = 0

if not cap.isOpened():
    print("Couldn't open Camera")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Frame not read")
        break

    if frame is None or frame.size == 0:
        print("Frame is empty")
        break
    
    imgRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    print(results.multi_hand_landmarks)
    
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                print(id, lm)
                h, w, c = frame.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                print(id, cx, cy)
                
                # if id == 4:
                #     cv.circle(frame, (cx, cy), 15, (255,0,255), cv.FILLED)
                
                
            mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS) # to plot the dots and the connectors 
                
    cTime = time.time() # current Time
    fps = 1/(cTime-pTime) # calculating the fps
    pTime = cTime 
    
    cv.putText(frame, str(int(fps)), (10,70), cv.FONT_HERSHEY_COMPLEX_SMALL, 3, (255,0,255), 2) # printing the fps

    cv.imshow('detected Hand', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    
    
cap.release()
cv.destroyAllWindows()
