from PoseModule import PoseDetector
import cv2
import numpy as np

cap = cv2.VideoCapture("squat3_clip.mp4")
detector = PoseDetector()
dir = 0 
count = 0
while True:
    success, img = cap.read()
    if success:
        img = cv2.resize(img, (640, 480))
        h, w, c = img.shape
        pose, img = detector.findPose(img, draw=True)
        if pose:
            lmList = pose["lmList"]
            angle, img = detector.findAngle(lmList[24], lmList[26],
                                            lmList[28], img)
            bar = np.interp(angle, (85, 180), (w//2-100, w//2+100))
            cv2.rectangle(img, (w//2-100, 50), (int(bar), 100),
                               (0, 255, 0), cv2.FILLED)
            if angle <= 105: 
                if dir == 0:
                    count = count + 0.5
                    dir = 1 
            if angle >= 170: 
                if dir == 1:
                    count = count + 0.5
                    dir = 0  
            msg = str(int(count))        
            cv2.putText(img, msg, (100, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 5,
                        (255, 255, 255), 20)
        cv2.imshow("Pose", img)        
    else:
        break
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

