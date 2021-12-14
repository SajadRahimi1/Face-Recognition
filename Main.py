import cv2
import sys
import numpy as np
from time import sleep
import argparse

#Load Image
ap = argparse.ArgumentParser()
ap.add_argument("-i","--image",required=True,help="Path of Image")
args = vars(ap.parse_args())
frame = cv2.imread(args["image"])

faceCascade = cv2.CascadeClassifier(r'xml\haarcascade_frontalface_default.xml')
faceNeighborsMax = 10
neighborStep = 1


frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Perform multi scale detection of faces 
for neigh in range(1, faceNeighborsMax, neighborStep):
    faces = faceCascade.detectMultiScale(frameGray, 1.2, neigh)
    frameClone = np.copy(frame)

# Display the image
for (x, y, w, h) in faces:
    cv2.rectangle(frameClone, (x, y), (x + w, y + h), (255, 0, 0), 2)

cv2.putText(frameClone, "=>SajadRahimi1<= # Neighbors = {}".format(neigh), (10, 50), 
cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
cv2.imshow('Face Detection Demo', frameClone)
cv2.waitKey(0)
