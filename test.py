import cv2
import pytesseract
import numpy as np
import operator
import os
import  imutils

import dlib
from imutils import face_utils

##################################################################### detect photo
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    r'C:\shape_predictor_68_face_landmarks.dat')
##################################################################### detect photo

####################################################################
cap = cv2.VideoCapture("http://192.168.2.35:4747/video")

while(True):
    ret,frame = cap.read()

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    ret,thr = cv2.threshold(gray,60,255,cv2.THRESH_BINARY)

    ############### tesseract#############
    #config_img=('-l tha --oem 1 --tessdata-dir ./tha_base/ --psm 4')
    text = pytesseract.image_to_string(thr,lang='tha')  #config=config_img
    ############### tesseract#############
    text = text.replace(' ','')

    ########################### profle photo #####################

    rects = detector(gray,1)

    for (i,rect) in enumerate(rects):
        (x,y,w,h) = face_utils.rect_to_bb(rect)
        #cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    crop = frame[y-40:y+h+70,x-50:x+w+50]
    ########################### profle photo #####################



    print(text)
    print('###########################################\n\n')
    #cv2.imshow('Gray frame',gray)
    #cv2.imshow('Threshold frame',thr)
    cv2.imshow('frame',frame)
    cv2.imshow('photo',crop)


    cv2.waitKey(1)

#cap.release()
#cv2.destroyAllWindows()
####################################################################

