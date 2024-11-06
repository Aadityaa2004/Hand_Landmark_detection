import cv2 as cv
import mediapipe as mp
import time
import math
import osascript

class handDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.7, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, frame, draw=True):
        imgRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(self.results.multi_hand_landmarks)
        
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(frame, handLms, self.mpHands.HAND_CONNECTIONS)
                    
                for id, lm in enumerate(handLms.landmark):
                    # print(id, lm)
                    h, w, c = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    # print(id, cx, cy)
                    
                    if id == 0:
                        cv.circle(frame, (cx, cy), 15, (255, 0, 255), cv.FILLED)
        return frame
    
    def findPosition(self, img, handNo=0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(id, cx, cy)
                lmList.append([id, cx, cy])
                if draw:
                    cv.circle(img, (cx, cy), 15, (255, 0, 255), cv.FILLED)
        return lmList

def set_volume(volume):
    """Set the system volume on macOS using osascript"""
    volume = max(0, min(volume, 100))  # Ensure volume is between 0 and 100
    osascript.osascript(f"set volume output volume {volume}")

class poseDetector():
    def __init__(self, mode=False, modelComplexity=1, smooth=True,
                 detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.modelComplexity = modelComplexity
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(
            static_image_mode=self.mode,
            model_complexity=self.modelComplexity,
            smooth_landmarks=self.smooth,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        
    def findPose(self, img, draw=True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS)
        return img
    
    # def findPosition(self, img, draw=True):
    #     self.lmList = []
    #     if self.results.pose_landmarks:
    #         for id, lm in enumerate(self.results.pose_landmarks.landmark):
    #             h, w, c = img.shape
    #             # print(id, lm)
    #             cx, cy = int(lm.x * w), int(lm.y * h)
    #             self.lmList.append([id, cx, cy])
    #             if draw:
    #                 cv.circle(img, (cx, cy), 5, (255, 0, 0), cv.FILLED)
    #     return self.lmList
    
    # def findAngle(self, img, p1, p2, p3, draw=True):
    #     # Get the landmarks
    #     x1, y1 = self.lmList[p1][1:]
    #     x2, y2 = self.lmList[p2][1:]
    #     x3, y3 = self.lmList[p3][1:]
    #     # Calculate the Angle
    #     angle = math.degrees(math.atan2(y3 - y2, x3 - x2) -
    #                          math.atan2(y1 - y2, x1 - x2))
    #     if angle < 0:
    #         angle += 360
    #     # print(angle)
    #     # Draw
    #     if draw:
    #         cv.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
    #         cv.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
    #         cv.circle(img, (x1, y1), 10, (0, 0, 255), cv.FILLED)
    #         cv.circle(img, (x1, y1), 15, (0, 0, 255), 2)
    #         cv.circle(img, (x2, y2), 10, (0, 0, 255), cv.FILLED)
    #         cv.circle(img, (x2, y2), 15, (0, 0, 255), 2)
    #         cv.circle(img, (x3, y3), 10, (0, 0, 255), cv.FILLED)
    #         cv.circle(img, (x3, y3), 15, (0, 0, 255), 2)
    #         cv.putText(img, str(int(angle)), (x2 - 50, y2 + 50),
    #                     cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
    #     return angle

class faceDetector:
    def __init__(self, minDetectionCon=0.25):
        self.minDetectionCon = minDetectionCon
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionCon)

    def findFaces(self, img, draw=True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        bboxs = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                bboxs.append([id, bbox, detection.score])
                if draw:
                    img = self.blur(img, bbox)
                    cv.putText(img, f'{int(detection.score[0] * 100)}%',
                               (bbox[0], bbox[1] - 20), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
        return img, bboxs

    def blur(self, img, bbox, l=30, t=5, rt=1):
        x, y, w, h = bbox
        x1, y1 = x + w, y + h

        # Ensure coordinates are within the image dimensions
        if x < 0 or y < 0 or x1 > img.shape[1] or y1 > img.shape[0]:
            print(f"Invalid bounding box coordinates: {bbox}")
            return img

        roi = img[y:y1, x:x1]

        # Check if ROI is valid
        if roi.size == 0:
            print(f"Empty ROI with bounding box: {bbox}")
            return img

        blur = cv.GaussianBlur(roi, (51, 51), 0)
        img[y:y1, x:x1] = blur
        cv.rectangle(img, bbox, (0, 0, 0), rt)
        return img


        # # Top Left  x,y
        # cv.line(img, (x, y), (x + l, y), (255, 0, 255), t)
        # cv.line(img, (x, y), (x, y+l), (255, 0, 255), t)
        # # Top Right  x1,y
        # cv.line(img, (x1, y), (x1 - l, y), (255, 0, 255), t)
        # cv.line(img, (x1, y), (x1, y+l), (255, 0, 255), t)
        # # Bottom Left  x,y1
        # cv.line(img, (x, y1), (x + l, y1), (255, 0, 255), t)
        # cv.line(img, (x, y1), (x, y1 - l), (255, 0, 255), t)
        # # Bottom Right  x1,y1
        # cv.line(img, (x1, y1), (x1 - l, y1), (255, 0, 255), t)
        # cv.line(img, (x1, y1), (x1, y1 - l), (255, 0, 255), t)

        


        return img


class FaceMeshDetector():
    def __init__(self, staticMode=False, maxFaces=2, minDetectionCon=0.5, minTrackCon=0.5):
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(
            static_image_mode=self.staticMode,
            max_num_faces=self.maxFaces,
            min_detection_confidence=self.minDetectionCon,
            min_tracking_confidence=self.minTrackCon
        )
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)

    def findFaceMesh(self, img, draw=True):
        self.imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        faces = []
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        img, faceLms, self.mpFaceMesh.FACEMESH_TESSELATION,
                        self.drawSpec, self.drawSpec
                    )
                face = []
                for id, lm in enumerate(faceLms.landmark):
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    face.append([x, y])
                    cv.putText(img, str(id), (x,y), cv.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)

                faces.append(face)
        return img, faces
    


def test_case():
    pTime = 0
    cTime = 0
    
    cap = cv.VideoCapture(0)
    cap.set(3, 640)  # Width
    cap.set(4, 480)  # Height
    
    # detector = handDetector()
    
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

        # frame = detector.findHands(frame)

        cTime = time.time()  # current Time
        fps = 1 / (cTime - pTime)  # calculating the fps
        pTime = cTime
        
        cv.putText(frame, str(int(fps)), (10, 70), cv.FONT_HERSHEY_COMPLEX_SMALL, 3, (255, 0, 255), 2)  # printing the fps

        cv.imshow('detected Hand', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv.destroyAllWindows()



if __name__ == "__main__":
    test_case()
