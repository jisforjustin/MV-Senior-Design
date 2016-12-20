"""******************************************************************
File Name:main.py

Authors: Justin Jordan, David Ikemba, Thuong Nguyen
Date:
Project: Machine Vision Burr Detection System
Sponsor: Hunt and Hunt Ltd.
Faculty Advisor: Dr. Fred Chen
Instructor: Dr. Compeau
University: Texas State University Ingram School of Engineering

Description: The Machine Vision Burr Detection System (MVBDS) will
be designed by Texas State University Electrical Engineering
students, for Hunt and Hunt Ltd., to detect burrs at the
intersection of the keyway and inner-diameter threading of precision
machined pipes. The system will generate a pass/fail signal to let
the user know if burrs are detected. This is a proof of concept
design to demonstrate that machine vision can be used to automate
burr detection.

Minimum Requirements: The MVBDS must comply with the following:

Must detect defects within 1.5 minute time frame
Must generate pass/fail signal with 95% accuracy or greater
Must use Hunt & Hunt Ltd. Machined pipes
Must detect defects 1mm or larger
Project budget needs to be $500 or less
******************************************************************"""

import numpy as np
import cv2
from matplotlib import pyplot as plt

# create webcam object. set to 0 when running on ODROID.
cam = cv2.VideoCapture(0)

choice = input ("Enter 'b' for begin test or 'q' for quit: ")

#TMN 12/20/2016
while choice == "d" or choice == "D":
    img = cv2.imread('meeseeks.jpg', 0)#need correct image
    r = 500.0 / img.shape[1]
    dim = (500, int(img.shape[0] * r))
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    cv2.imshow("resized", resized) #display resized image
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    choice = input("Enter 'b' for begin test or 'q' for quit: ")



while choice != "q" and choice != "Q":
    if choice != "b" and choice != "B":
        choice = input ("Invalid choice. Enter 'b; for begin test or 'q' for quit: ")
        continue

    # capture 1 frame from cam
    returnVal, workingImg = cam.read()
    if returnVal == False:
        print("No communication with camera. Program terminating.")
        exit(-1)

    # plot working image. this code is for testing and can be removed later.
    plt.imshow(workingImg, cmap='brg', interpolation='bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()

    # convert to grayscale
    grayImg = cv2.cvtColor(workingImg, cv2.COLOR_BGR2GRAY)

    # plot grayscale image. this code is for testing and can be removed later.
    plt.imshow(grayImg, cmap='gray', interpolation='bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()

    # apply Gaussian bilateral filter to grayscale image
    filterImg = cv2.bilateralFilter(grayImg, 5, 100, 100)

    # plot filtered image. this code is for testing and can be removed later.
    plt.imshow(filterImg, cmap='gray', interpolation='bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()

    # apply Sobel filter
    sobelImg = cv2.Sobel(grayImg, cv2.CV_64F, 1, 1, ksize = 5)

    # plot results of Sobel filter. this code is for testing and can be removed later.
    plt.imshow(sobelImg, cmap='gray', interpolation='bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()

    # apply thresholding for binary image
    binaryImg = cv2.Canny(filterImg, 100, 125)

    # plot results of thresholding. this code is for testing and can be removed later.
    plt.imshow(binaryImg, cmap='gray', interpolation='bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()

    choice = input ("Enter 'b' for begin test or 'q' for quit: ")

cam.release()
cv2.destroyAllWindows()
