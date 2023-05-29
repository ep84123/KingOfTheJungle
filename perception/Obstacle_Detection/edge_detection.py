import os

import cv2
import numpy as np
from utils import img_show, init_clock
from masking import create_mask

# new height and width for the image
IMG_WIDTH = 640
IMG_HEIGHT = 480


# Draw bboxes of contours
def draw_bboxes(contours, img):
    for cnt in contours:
        # print(cv2.contourArea(cnt))
        if cv2.contourArea(cnt) > 2:
            #print("add")
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            img2 = cv2.drawContours(img, [box], 0, (0, 255, 255), 2)
    cv2.imshow('BBOX not straight', img2)
    cv2.waitKey(0)


def process_img(img, t0):
    # Convert to graycsale
    prs_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_show("Grayscale", prs_img, t0)
    # ret, prs_img = cv2.threshold(prs_img, 170, 255, cv2.THRESH_BINARY)
    # visualize the binary image
    cv2.imshow('Binary image', prs_img)
    # Blur the image for better edge detection
    # img_show("Blur", prs_img)
    mask = cv2.cvtColor(create_mask(), cv2.COLOR_BGR2GRAY)

    # Bitwise-AND mask and original image
    prs_img = cv2.bitwise_and(prs_img, prs_img, mask=mask)
    img_show("Mask image", prs_img, t0)
    prs_img = cv2.GaussianBlur(prs_img, (3, 3), 0)  # play with values
    img_show("Blur image", prs_img, t0)

    # Sobel Edge Detection - only for testing
    # sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)
    # img_show("Sobel X", sobelx)
    # sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)
    # img_show("Sobel Y", sobelx)
    # sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)
    # img_show("Sobel X Y", sobelx)

    return prs_img


def detect_edge(path: str):
    alpha_min = 1/3
    alpha_max = 2/3
    t0 = init_clock()  # for runtime test only
    # img_show("str", None, t0)
    # Read the original image
    raw_img = cv2.imread(path)
    img_show("Raw image", raw_img, t0)

    prss_img = process_img(raw_img, t0)
    kernel = np.ones((5, 5), np.uint8)
    img_show("proccesing_img", prss_img, t0)
    img_width = prss_img.shape[1]
    img_height = prss_img.shape[0]
    cropped_image = prss_img[int(alpha_min*img_height):int(alpha_max*img_height), :]
    raw_img_cropped = raw_img[int(alpha_min*img_height):int(alpha_max*img_height), :]
    # Canny Edge Detection
    img = cv2.Canny(image=cropped_image, threshold1=250, threshold2=450)  # play with  threshold values
    img_show("Canny Edge Detection", img, t0)

    img = cv2.dilate(img, kernel, iterations=2)
    img_show("img_erosion", img, t0)
    img = cv2.erode(img, kernel, iterations=1)
    img_show("img_erosion", img, t0)
    img = cv2.dilate(img, kernel, iterations=2)
    img_show("img_erosion", img, t0)
    img = cv2.erode(img, kernel, iterations=3)
    img_show("img_erosion", img, t0)
    img = cv2.dilate(img, kernel, iterations=1)
    img_show("img_erosion", img, t0)
    img = cv2.erode(img, kernel, iterations=2)
    img_show("img_erosion", img, t0)
    img = cv2.dilate(img, kernel, iterations=1)
    img_show("img_erosion", img, t0)
    img = cv2.erode(img, kernel, iterations=2)
    img_show("img_erosion", img, t0)
    img = change_row_to_white(img)
    img_show("img_erosion", img, t0)
    # img = cv2.Canny(image=prss_img, threshold1=250, threshold2=450)  # play with  threshold values
    # img_show("Canny Edge Detection", img, t0)
    # Finding Contours - Contours is a Python list of all the contours in the image.
    # Each individual contour is a Numpy array of (x,y) coordinates of boundary points of the object.
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)     ## I CHANGED NONE->SIMPLE
    print("Number of Contours found = " + str(len(contours)))

    # Draw all contours
    cv2.drawContours(raw_img_cropped, contours, -1, (0, 255, 0), 3)  # -1: draw all, color, thickness
    img_show("Canny after contouring", raw_img, t0)

    draw_bboxes(contours, raw_img_cropped)
    cv2.destroyAllWindows()


def change_row_to_white(img):
    # img = cv2.imread(path)
    img_width = img.shape[1]
    img_height = img.shape[0]
    for cols_index in range(img_width):
        for j in range(img_height-600):
            img[j][cols_index] = 255  # White pixel
        # for k in range(15):
        #     img[img_height-k-1][cols_index] = 255
    # for rows_index in range(img_height):
    #     for j in range(15):
    #         img[rows_index][j] = 0  # White pixel
    #     for k in range(15):
    #         img[rows_index][img_width-k-1] = 0
    return img


def substract_frames():
    root = "../../data/DJI_0457.MP4"
    root1 = "Video_Frames"
    source = cv2.VideoCapture(root)

    for i in range(0, 100):
        ret, img1 = source.read()
        ret, img2 = source.read()
        for j in range(5):
            ret, img2 = source.read()
        if i % 5 == 0:
            # img1 = cv2.imread(f"{root}-{i}.jpg")
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            img1 = cv2.GaussianBlur(img1, (5, 5), 0)
            # img2 = cv2.imread(f"{root}-{i-2}.jpg")
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            img2 = cv2.GaussianBlur(img2, (5, 5), 0)

            sub_img = cv2.subtract(img1, img2)
            cv2.imwrite(os.path.join(root1, "SUB_IMAGE" + '-' + str(i) + '.jpg'), sub_img)
            cv2.imshow(f"sub {i} and {i - 2} img", sub_img)
            cv2.waitKey(0)
