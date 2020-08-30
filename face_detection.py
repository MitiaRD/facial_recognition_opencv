import cv2
import sys
import os

path = 'known_faces'
cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

os.chdir(path)

for images in os.listdir():
    if images.endswith('.jpg'):
        image = cv2.imread(images)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=2,
            minSize=(10, 10)
            #flags = cv2.CV_HAAR_SCALE_IMAGE
        )
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.imshow("Faces found", image)
        cv2.waitKey(0)
