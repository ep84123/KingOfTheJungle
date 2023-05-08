import cv2
import numpy as np
from contour_merge import *

def detect_edge():
    # Read the original image
    img = cv2.imread("photo_tree_2.jpg")
    # Display original image
    cv2.imshow("photo_tree_2.jpg", img)
    cv2.waitKey(0)

    # Convert to graycsale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Blur the image for better edge detection
    img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)

    # Sobel Edge Detection
    sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)  # Sobel Edge Detection on the X axis
    sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)  # Sobel Edge Detection on the Y axis
    sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)  # Combined X and Y Sobel Edge Detection
    # Display Sobel Edge Detection Images
    # cv2.imshow('Sobel X', sobelx)
    cv2.waitKey(0)
    # cv2.imshow('Sobel Y', sobely)
    cv2.waitKey(0)
    # cv2.imshow('Sobel X Y using Sobel() function', sobelxy)
    cv2.waitKey(0)

    # Canny Edge Detection
    edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)  # Canny Edge Detection
    # Display Canny Edge Detection Image
    # cv2.imshow('Canny Edge Detection', edges)
    cv2.waitKey(0)

    # Finding Contours
    # Use a copy of the image e.g. edged.copy()
    # since findContours alters the image
    contours, hierarchy = cv2.findContours(edges,
                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = agglomerative_cluster(contours)
    cnt1 = contours[0]
    # cv2.imshow('Canny Edges After Contouring', edges)
    cv2.waitKey(0)

    print("Number of Contours found = " + str(len(contours)))
    # Draw all contours
    # -1 signifies drawing all contours
    cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
    # megred_contours = agglomerative_cluster(contours)
    #
    # print("Number of Contours found = " + str(len(megred_contours)))
    # cv2.drawContours(img, megred_contours, -1, (0, 255, 0), 3)
    cv2.imshow('Contours', img)
    cv2.waitKey(0)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        img1 = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow('BBOX straight', img1)
    cv2.waitKey(0)

    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        img2 = cv2.drawContours(img, [box], 0, (0, 255, 255), 2)
        cv2.imshow('BBOX not straight', img2)
    cv2.waitKey(0)

    cv2.destroyAllWindows()

