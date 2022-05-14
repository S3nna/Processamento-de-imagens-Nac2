#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np

img1 = cv2.imread("carta7.png",0) # carrregada em escala de cinza

scale_percent = 200 # percent of original size
width = int(img1.shape[1] * scale_percent / 100)
height = int(img1.shape[0] * scale_percent / 100)
dim = (width, height)


img1 = cv2.resize(img1, dim, interpolation = cv2.INTER_AREA)

orb = cv2.SIFT_create()


kp1, des1 = orb.detectAndCompute(img1,None)

cap = cv2.VideoCapture("q1.mp4")

if not cap.isOpened():
    print("Cannot open camera")
    exit()
    
while True:
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")        
        break
    
    # Seu código aqui. 
    kp2, des2 = orb.detectAndCompute(frame,None)
    gray2 = cv2.drawKeypoints(frame, kp2, outImage=np.array([]), flags=0)

    #fazendo o match, encontrando as referencias
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)


    # Apply ratio test  varre as relações encontradas e cria a lista good com as melhores 
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])

    if(len(good) > 10):
        print("carta encontrada")
        img3 = cv2.drawMatchesKnn(img1,kp1,frame,kp2,good[:20], None, flags=2)
    else:
        print("não encontrei bons matchs")
        img3 = frame


    # Exibe resultado
    cv2.imshow("frame", img3)

    # Wait for key 'ESC' to quit
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

# That's how you exit
cap.release()
cv2.destroyAllWindows()