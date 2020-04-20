import cv2
from model import FacialExpressionModel
import numpy as np

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = FacialExpressionModel("model.json", "model_weights.h5")
font = cv2.FONT_HERSHEY_SIMPLEX

video_capture = cv2.VideoCapture(0)
# video_capture = cv2.VideoCapture("1.mp4")

while True:
    ret, fr = video_capture.read()
    fr = cv2.flip(fr,1)
    gray_fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray_fr,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    for (x, y, w, h) in faces:
        fc = gray_fr[y:y+h, x:x+w]
        roi = cv2.resize(fc, (48, 48))
        pred = model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])

        cv2.putText(fr, pred, (x, y), font, 1, (255, 255, 0), 2)
        cv2.rectangle(fr,(x,y),(x+w,y+h),(255,0,0),2)
        _, jpeg = cv2.imencode('.jpg', fr)

    cv2.imshow('Video', fr)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
