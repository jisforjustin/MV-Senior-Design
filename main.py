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

choice = input ("Enter 'b' for begin test or 'q' for quit: ")

while choice != "q" and choice != "Q":
    if choice != "b" and choice != "B":
        choice = input ("Invalid choice. Enter 'b; for begin test or 'q' for quit: ")
        continue

    """ Main code to loop will go here """

    choice = input ("Enter 'b' for begin test or 'q' for quit: ")