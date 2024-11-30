import cv2
import numpy as np

input = cv2.imread('book.jpeg')
input= cv2.resize(input,(400,550),interpolation = cv2.INTER_AREA)
gray_image= cv2.cvtColor(input , cv2.COLOR_BGR2GRAY)

orb = cv2.ORB_create(nfeatures= 1000)

keypoints, descriptors=orb.detectAndCompute(gray_image,None)
final_keypoints = cv2.drawKeypoints(gray_image,keypoints,input,(0,255,0))
cv2.imshow('ORB_KEYPOINTS',final_keypoints)
cv2.waitKey()