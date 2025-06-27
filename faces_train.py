import os
import  cv2 as cv
import numpy as np

people = ['Elon Musk', 'Jeff Bezos', 'Donald Trump']
DIR = os.path.join(os.path.dirname(__file__), 'assets/traning data')
hear_cascade = cv.CascadeClassifier('hear_face.xml')

features = []
labels = []



def create_train() :
    for person in people :
        path = os.path.join(DIR, person)
        label = people.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            
            img_cv = cv.imread(img_path)
            gray = cv.cvtColor(img_cv, cv.COLOR_BGR2GRAY)
            face_rect = hear_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            for (x,y,w,h) in face_rect:
                face_roi = gray[y:y+h, x:x+w]
                features.append(face_roi)
                labels.append(label)


create_train()

features = np.array(features, dtype='object')
labels = np.array(labels)

face_recogniser = cv.face.LBPHFaceRecognizer_create()
face_recogniser.train(features, labels)

face_recogniser.save("face_trained.yaml")
np.save("features.npy", features)
np.save("labels.npy", labels)


print('Length of features list', len(features))
print('Length of labels list', len(labels))
print("Training done !!!")
