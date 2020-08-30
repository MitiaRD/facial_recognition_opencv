import cv2
import sys
import os

#defining paths for the images and haar cascade
path = 'known_faces'
cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

#change the directory to loop through the images
os.chdir(path)

#looping through images in the current directory
for images in os.listdir():
    #avoiding filenot found error
    if images.endswith('.jpg'):
        #reading the image
        image = cv2.imread(images)
        #converting image to gray to reduce information in image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #using the haar cascade to find faces
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=2,
            minSize=(10, 10)
            #flags = cv2.CV_HAAR_SCALE_IMAGE
        )
        #drawing boxes around the faces found
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.imshow("Faces found", image)
        cv2.waitKey(0)
