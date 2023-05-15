import os

import cv2
import numpy as np
from utils import img_show, init_clock

# new height and width for the image
IMG_WIDTH = 640
IMG_HEIGHT = 480


# Draw bboxes of contours
def draw_bboxes(contours, img):
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        img2 = cv2.drawContours(img, [box], 0, (0, 255, 255), 2)
        cv2.imshow('BBOX not straight', img2)
    cv2.waitKey(0)


def process_img(img):
    # Resize the image
    resized_img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
    img_show("Low resolution", resized_img)

    # Convert to graycsale
    prs_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_show("Grayscale", prs_img)

    # Blur the image for better edge detection
    prs_img = cv2.GaussianBlur(prs_img, (7, 7), 0)  # play with values
    img_show("Blur", prs_img)

    # Sobel Edge Detection - only for testing
    # sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)
    # img_show("Sobel X", sobelx)
    # sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)
    # img_show("Sobel Y", sobelx)
    # sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)
    # img_show("Sobel X Y", sobelx)

    return img, prs_img


def detect_edge(path: str):
    init_clock()  # for runtime test only

    # Read the original image
    img = cv2.imread(path)
    img_show("Raw image", img)

    # raw_img, prss_img = process_img(img)

    # Canny Edge Detection
    edges = cv2.Canny(image=img, threshold1=100, threshold2=200)  # play with  threshold values
    img_show("Canny Edge Detection", edges)

    # Finding Contours - Contours is a Python list of all the contours in the image.
    # Each individual contour is a Numpy array of (x,y) coordinates of boundary points of the object.
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    print("Number of Contours found = " + str(len(contours)))

    # Draw all contours
    cv2.drawContours(img, contours, -1, (0, 255, 0), 3)  # -1: draw all, color, thickness
    img_show("Canny after contouring", img)

    # merged_contours = agglomerative_cluster(contours)
    # print("Number of Contours after merge = " + str(len(merged_contours)))
    # cv2.drawContours(img, merged_contours, -1, (0, 255, 0), 3)
    # img_show('Contours', img)

    draw_bboxes(contours, img)
    # cv2.destroyAllWindows()

# def calculate_contour_distance(contour1, contour2):
#     x1, y1, w1, h1 = cv2.boundingRect(contour1)
#     c_x1 = x1 + w1 / 2
#     c_y1 = y1 + h1 / 2
#
#     x2, y2, w2, h2 = cv2.boundingRect(contour2)
#     c_x2 = x2 + w2 / 2
#     c_y2 = y2 + h2 / 2
#
#     return max(abs(c_x1 - c_x2) - (w1 + w2) / 2, abs(c_y1 - c_y2) - (h1 + h2) / 2)
#
#
# def merge_contours(contour1, contour2):
#     return np.concatenate((contour1, contour2), axis=0)
#
#
# def agglomerative_cluster(contours, threshold_distance=40.0):
#     current_contours = contours
#     while len(current_contours) > 1:
#         min_distance = None
#         min_coordinate = None
#
#         for x in range(len(current_contours) - 1):
#             for y in range(x + 1, len(current_contours)):
#                 distance = calculate_contour_distance(current_contours[x], current_contours[y])
#                 if min_distance is None:
#                     min_distance = distance
#                     min_coordinate = (x, y)
#                 elif distance < min_distance:
#                     min_distance = distance
#                     min_coordinate = (x, y)
#
#         if min_distance < threshold_distance:
#             index1, index2 = min_coordinate
#             current_contours[index1] = merge_contours(current_contours[index1], current_contours[index2])
#             del current_contours[index2]
#         else:
#             break
#
#     return current_contours


def substract_frames():
    root = "C:\\Users\\TLP-300\\Desktop\\King-Of-The-Jungle\\perception\\Obstacle_Detection\\video_out.mp4v"
    root1 = "C:\\Users\\TLP-300\\Desktop\\King-Of-The-Jungle\\perception\\Obstacle_Detection"
    source = cv2.VideoCapture(root)

    for i in range(0, 50):
        ret, img1 = source.read()
        ret, img2 = source.read()
        for j in range(10):
            ret, img2 = source.read()
        if i%5 == 0:
            # img1 = cv2.imread(f"{root}-{i}.jpg")
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            img1 = cv2.GaussianBlur(img1, (3, 3), 0)
            # img2 = cv2.imread(f"{root}-{i-2}.jpg")
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            img2 = cv2.GaussianBlur(img2, (3, 3), 0)

            sub_img = cv2.subtract(img1, img2)
            cv2.imwrite(os.path.join(root1, "SUB_IMAGE" + '-' + str(i) + '.jpg'), sub_img)
            cv2.imshow(f"sub {i} and {i-2} img", sub_img)
            cv2.waitKey(0)