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


    ########################### profle photo #####################
    rects = detector(gray,1)

    for (i,rect) in enumerate(rects):
        (x,y,w,h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    crop = frame[y-40:y+h+70,x-50:x+w+50]
    ########################### profle photo #####################


    # ################ contours
    # _,cnts,_ = cv2.findContours(thr,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    #
    # indexAreaMax = 0
    # max = 0
    # i = 0
    # for c in cnts:
    #     if cv2.contourArea(c)>max:
    #         max = cv2.contourArea(c)
    #         indexAreaMax = i
    #         (x,y,w,h) = cv2.boundingRect(c)
    #     i = i + 1
    # #cv2.drawContours(frame, indexAreaMax,-1,(255,0,0),3)
    # cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
    # ################ contours


    print(text)
    print('###########################################\n\n')
    #cv2.imshow('Gray frame',gray)
    cv2.imshow('Threshold frame',thr)
    cv2.imshow('frame',frame)
    cv2.imshow('photo',crop)


    cv2.waitKey(1)

#cap.release()
#cv2.destroyAllWindows()
####################################################################



# ##################################################################### photo
# img = cv2.imread('D:/Wick/Univeraity/4/computer_vision/ass4/2.png')
# img = imutils.resize(img,width=1000)
#
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#
# #####################
# rects = detector(gray,1)
#
# for (i,rect) in enumerate(rects):
#     (x,y,w,h) = face_utils.rect_to_bb(rect)
#     cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
# #####################
#
#
# blur = cv2.GaussianBlur(gray,(5,5),0)
# ret,thr = cv2.threshold(blur,90,255,cv2.THRESH_BINARY)
#
#
# cv2.imshow('threshold',thr)
# cv2.imshow('Detect',img)
# cv2.waitKey()
# ##################################################################### photo
