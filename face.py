# modifying the realtime face detection so that we can append time and date when the face appears
from datetime import datetime
import cv2 
import time
import pandas as pd
first_frame = None
status_list = [None, None]
times = []
df = pd.DataFrame(columns = ["Start", "END"])
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml');

video = cv2.VideoCapture(0);

while True:
    check, frame = video.read()
    status = 0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    if first_frame is None:
        first_frame = gray
        continue
    delta_frame = cv2.absdiff(first_frame, gray)
    faces = face_cascade.detectMultiScale(frame,
                                          scaleFactor=1.1, minNeighbors=5);
    for x,y,w,h in faces:
        frame = cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 3);
        status = 1
    status_list.append(status)
        
    status_list = status_list[-2:]
        
    if status_list[-1] == 1 and status_list[-2] == 0:
        times.append(datetime.now())
    if status_list[-1] == 0 and status_list[-2] == 1:
        times.append(datetime.now())
    cv2.imshow('Face Detector', frame)
    cv2.imshow('delta_frame', delta_frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break;
for i in range(0, len(times), 2):
    df = df.append({"Start": times[i], "END": times[i+1]}, ignore_index = True)
    df.to_csv("time.csv")
video.release()
cv2.destroyAllWindows()
