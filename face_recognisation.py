import numpy as np
import cv2 as cv

people = ['Elon Musk', 'Jeff Bezos', 'Donald Trump']

hear_cascade = cv.CascadeClassifier('hear_face.xml')

features = np.load("features.npy", allow_pickle=True)
labels = np.load("labels.npy")

face_recogniser = cv.face.LBPHFaceRecognizer_create()
face_recogniser.read("face_trained.yaml")

img = cv.imread("assets/test data/mark.webp")

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

faces_rect = hear_cascade.detectMultiScale(gray, 1.1, 4)

for (x,y,w,h) in faces_rect:
    faces_rect = gray[y:y+h, x:x+h]
    label, confidenceValue = face_recogniser.predict(faces_rect)
    
    cv.putText(img, str(people[label]), (20,20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0))
    cv.rectangle(img, (x,y), (x+w, y+h), (0,255,0), thickness=2)
    cv.imshow("face Recognisation", img)
    
    print("Label", people[label])
    print("Confidence", confidenceValue)

cv.waitKey(0)