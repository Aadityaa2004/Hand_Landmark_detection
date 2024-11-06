import cv2 as cv
import mediapipe as mp 
import time
import math 
import Module as Module
import cv2 as cv

def hand_detection():
    pTime = 0
    cTime = 0
    
    cap = cv.VideoCapture(0)
    cap.set(3, 1920)  # Width
    cap.set(4, 1080)  # Height
    
    detector = Module.handDetector()
    
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
        frame = detector.findHands(frame)

        cTime = time.time()  # current Time
        fps = 1 / (cTime - pTime)  # calculating the fps
        pTime = cTime

        cv.putText(frame, str(int(fps)), (10, 30), fontScale=0.7, fontFace=cv.FONT_HERSHEY_SCRIPT_SIMPLEX, color=(10, 10, 10), thickness=2)  # printing the fps

        cv.imshow('detected Pose', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv.destroyAllWindows()
    
    
def pose_detection():
    pTime = 0
    cTime = 0
    
    cap = cv.VideoCapture(0)
    cap.set(3, 640)  # Width
    cap.set(4, 480)  # Height
    
    detector = Module.poseDetector()
    
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
        frame = detector.findPose(frame)

        cTime = time.time()  # current Time
        fps = 1 / (cTime - pTime)  # calculating the fps
        pTime = cTime

        cv.putText(frame, str(int(fps)), (10, 30), fontScale=0.7, fontFace=cv.FONT_HERSHEY_SCRIPT_SIMPLEX, color=(10, 10, 10), thickness=2)  # printing the fps

        cv.imshow('detected Hand', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv.destroyAllWindows()

def face_detection():
    pTime = 0
    cTime = 0
    
    cap = cv.VideoCapture(0)
    cap.set(3, 640)  # Width
    cap.set(4, 480)  # Height
    
    detector = Module.faceDetector()
    
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
        frame, bboxes = detector.findFaces(frame)
        print(bboxes)

        cTime = time.time()  # current Time
        fps = 1 / (cTime - pTime)  # calculating the fps
        pTime = cTime

        cv.putText(frame, f'FPS: {int(fps)}', (20, 70), cv.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)  # printing the fps

        cv.imshow('detected Face', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv.destroyAllWindows()


def mesh_detection():
    cap = cv.VideoCapture(0)
    pTime = 0

    detector = Module.FaceMeshDetector()

    while True:
        success, img = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        img, faces = detector.findFaceMesh(img)
        
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        
        cv.putText(img, f'FPS: {int(fps)}', (20, 70), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        cv.imshow("Image", img)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

    
   
    
if __name__ == '__main__':
    face_detection()

