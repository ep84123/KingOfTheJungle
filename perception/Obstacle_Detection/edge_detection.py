import os

import cv2
import numpy as np
from utils import img_show, init_clock
from utils import open_video
# new height and width for the image
IMG_WIDTH = 640
IMG_HEIGHT = 480


# Draw bboxes of contours
def draw_bboxes(contours, img):
    bboxes = []
    for cnt in contours:
        # print(cv2.contourArea(cnt))
        if cv2.contourArea(cnt) > 2:
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            bboxes.append(box)
            box = np.array(box).reshape((-1, 1, 2)).astype(np.int32)
            img2 = cv2.drawContours(img, [box], 0, (0, 255, 255), 2)
    # return bboxes
    cv2.imshow('BBOX not straight', img2)
    cv2.waitKey(0)


def process_img(raw_img, t0, index):
    # Convert to graycsale
    prs_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
    img_show("Grayscale", prs_img, t0)

    # mask = cv2.cvtColor(substract_frames(index), cv2.COLOR_BGR2GRAY)
    # img2 = cv2.imread(path2)
    # ret, mask = cv2.threshold(substract_frames(index), 50, 200, cv2.THRESH_BINARY)
    #
    # # # Bitwise-AND mask and original image
    # prs_img = cv2.bitwise_and(prs_img, prs_img, mask=mask)
    # img_show("Masked image", prs_img, t0)

    # Blur the image for better edge detection
    prs_img = cv2.GaussianBlur(prs_img, (3, 3), 0)  # play with values
    img_show("Blur image", prs_img, t0)

    return prs_img

def detect_edge(path1, index):
    t0 = init_clock()  # for runtime test only
    # Read the original image

    raw_img = cv2.imread(path1)

    # raw_img = frame
    img_show("Raw image", raw_img, t0)

    prss_img = process_img(raw_img, t0, index)
    kernel = np.ones((5, 5), np.uint8)


    # img = cv2.dilate(prss_img, kernel, iterations=2)
    # img_show("img_erosion", img, t0)
    img = cv2.erode(prss_img, kernel, iterations=1)
    # img_show("img_erosion", img, t0)
    img = cv2.dilate(prss_img, kernel, iterations=2)
    # img_show("img_erosion", img, t0)
    img = cv2.erode(img, kernel, iterations=3)
    # img_show("img_erosion", img, t0)
    img = cv2.dilate(img, kernel, iterations=1)
    # img_show("img_erosion", img, t0)
    img = cv2.erode(img, kernel, iterations=2)
    img_show("img_erosion", img, t0)
    # Canny Edge Detection
    img = cv2.Canny(image=prss_img, threshold1=250, threshold2=450)  # play with  threshold values
    img_show("Canny Edge Detection", img, t0)

    # Finding Contours - Contours is a Python list of all the contours in the image.
    # Each individual contour is a Numpy array of (x,y) coordinates of boundary points of the object.
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    print("Number of Contours found = " + str(len(contours)))

    # Draw all contours
    cv2.drawContours(raw_img, contours, -1, (0, 255, 0), 3)  # -1: draw all, color, thickness
    img_show("Canny after contouring", raw_img, t0)

    # draw_bboxes(contours, raw_img)
    cv2.destroyAllWindows()
    return contours


def substract_frames(index):
    root = "../../data/DJI_0457.MP4"
    root1 = "Video_Frames"
    source = open_video(root, 2e4)

    for i in range(0, index - 20):
        ret, img1 = source.read()
    for i in range(20):
        ret, img2 = source.read()
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img1 = cv2.GaussianBlur(img1, (3, 3), 0)
    img_show("img1", img1, 0)

    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img2 = cv2.GaussianBlur(img2, (3, 3), 0)
    img_show("img2", img2, 0)

    sub_img = cv2.subtract(img1, img2)
    img_show("Mask", sub_img, 0)
    return sub_img
    # cv2.imwrite(os.path.join(root1, "SUB_IMAGE" + '-' + str(i) + '.jpg'), sub_img)
    #     ret, img2 = source.read()
    #     for j in range(5):
    #         ret, img2 = source.read()
    #     if i % 5 == 0:
    #
    #
    #         # cv2.imshow(f"sub {i} and {i-2} img", sub_img)
    #         #             # cv2.waitKey(0)
