import cv2 as cv
import mediapipe as mp
import time

cap = cv.VideoCapture(0)

while True:
    success,img = cap.read()


    cv.imshow('Image', img)
    cv.waitKey(1)
