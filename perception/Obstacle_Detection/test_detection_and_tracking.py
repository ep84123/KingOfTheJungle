"""This File demonstrates the algorithm for object tracking and detection running on a simple video"""
import cv2
import numpy as np
from utils import *
import sys
from detect_simulation_tree import detect_edge_tree


# ----------------------------- Object Dection - simple version ---------------------------------------
def draw_bboxes(contours, img):
    bboxes = []
    for cnt in contours:
        # print(cv2.contourArea(cnt))
        if cv2.contourArea(cnt) > 4:
            # x, y, w, h = cv2.boundingRect(cnt)
            # cv2.rectangle(input, (x, y), (x + w - 1, y + h - 1), 255, 2)

            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            bboxes.append(box)
            box = np.array(box).reshape((-1, 1, 2)).astype(np.int32)
            img2 = cv2.drawContours(img, [box], 0, (0, 255, 255), 2)
    # cv2.imshow('BBOX not straight', img2)
    # cv2.waitKey(0)
    return bboxes


def process_img(raw_img, t0):
    prs_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
    # img_show("Grayscale", prs_img, t0)

    kernel = np.ones((3, 3), np.uint8)
    prs_img = cv2.dilate(prs_img, kernel, iterations=1)
    # img_show("img_erosion", prs_img, t0)

    prs_img = cv2.GaussianBlur(prs_img, (3, 3), 0)
    # img_show("Blur image", prs_img, t0)

    return prs_img


def simple_detect_edge(frame):
    t0 = init_clock()  # for runtime test only
    # width = 300
    # height = 250
    # dim = (width, height)
    #
    # frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

    prss_img = process_img(frame, t0)

    # Canny Edge Detection
    img = cv2.Canny(image=prss_img, threshold1=50, threshold2=150)  # play with  threshold values
    # img_show("Canny Edge Detection", img, t0)

    # Finding Contours
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # print("Number of Contours found = " + str(len(contours)))

    # Draw all contours
    cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)  # -1: draw all, color, thickness
    # img_show("Canny after contouring", frame, t0)

    cv2.destroyAllWindows()
    draw_bboxes(contours, frame)
    return contours


# -------------------------- Object tracking for obstacle detection ----------------------------------


THREAT_THRESHOLD = 3

tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']


# Create a tracker of type 'tracker_type'
def create_tracker(tracker_type: int):
    if tracker_type == 'BOOSTING':
        return cv2.legacy.TrackerBoosting_create()  # - SLOW
    if tracker_type == 'MIL':
        return cv2.legacy.TrackerMIL_create()  # - SLOW
    if tracker_type == 'KCF':
        return cv2.legacy.TrackerKCF_create()  # - FAST, NOT CHAINING SIZE
    if tracker_type == 'TLD':
        return cv2.legacy.TrackerTLD_create()  # - SUPER SLOW  - "Also, tracks best over scale changes"?
    if tracker_type == 'MEDIANFLOW':
        return cv2.legacy.TrackerMedianFlow_create()  # - FAST, NOT SO ACCURATE, CHAINING SIZE
    if tracker_type == 'GOTURN':
        return cv2.legacy.TrackerGOTURN_create()  # - NOT WORKING, FIX?
    if tracker_type == 'MOSSE':
        return cv2.legacy.TrackerMOSSE_create()  # - REALLY FAST, NOT CHAINING SIZE
    if tracker_type == "CSRT":
        return cv2.legacy.TrackerCSRT_create()  # - SLOW


# Get objects' bbox to track
def get_bbox_manually(frame):
    bbox_list = []
    color_list = []
    from_center = False  # Draw bounding box from upper left
    show_cross_hair = False  # Don't show the cross hair

    while True:
        print("press enter or space after marking the object")
        # Add bounding box from user, with random color
        bbox_list.append(cv2.selectROI('Multi-Object Tracker', frame, from_center, show_cross_hair))
        color_list.append(get_random_color())
        print("\nPress Esc to begin tracking objects or press " +
              "another key to draw the next bounding box\n")
        if (0, 0, 0, 0) in bbox_list:
            bbox_list.remove((0, 0, 0, 0))
        # Exit if ESC pressed
        if (cv2.waitKey(0) & 0xff) == EXIT_KEY:
            break

    print(f"tracking {len(bbox_list)}")
    return color_list, bbox_list


def get_bbox_from_detector(frame):
    bbox_list = []
    contours = simple_detect_edge(frame)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        bbox_list.append((x, y, w, h))
    # color = get_random_color()
    if (0, 0, 0, 0) in bbox_list:
        bbox_list.remove((0, 0, 0, 0))
    print(f"tracking {len(bbox_list)}")
    return bbox_list


# Draw the bounding boxes on the video frame
def draw_bbox(bboxes, frame):
    # for i, bbox in enumerate(bboxes):
    #     point_1 = (int(bbox[0]), int(bbox[1]))
    #     point_2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    #     cv2.rectangle(frame, point_1, point_2, get_random_color(), 5)
    #     M = cv2.moments(bbox)
    #     cX = int(M["m10"] / M["m00"])
    #     cY = int(M["m01"] / M["m00"])
    #     # print(cY, cX)
    #     cv2.circle(frame, (cX, cY), 10, get_random_color(), -1)
    #     # cv2.waitKey(0)
    for i, bbox in enumerate(bboxes):
        point_1 = (int(bbox[0]), int(bbox[1]))
        point_2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, point_1, point_2, get_random_color(), 5)
        # cv2.waitKey(0)


def init_tracker(frame):
    print("tracking failure")
    multi_tracker = cv2.legacy.MultiTracker_create()
    for contour in detect_edge_tree(frame):
        if cv2.contourArea(contour) > 15000:
            x, y, w, h = cv2.boundingRect(contour)

            # Create a tracker instance (e.g., using cv2.legacy.TrackerMedianFlow_create())
            tracker = create_tracker(tracker_types[2])

            # Initialize the tracker with the current frame and bounding box
            tracker.init(frame, (x, y, w, h))

            # Add the tracker and bounding box to the multi-object tracker
            multi_tracker.add(tracker, frame, (x, y, w, h))
    return multi_tracker


def display_tracker(frame, timer):
    # Display tracker type on frame
    cv2.putText(frame, "KCF Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

    # Display FPS on frame
    cv2.putText(frame, "FPS : " + calc_fps(timer), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

    # Display result
    cv2.imshow('frame', frame)


def is_threat(bbox_per_frame) -> bool:
    delta = 1
    for i in range(len(bbox_per_frame) - 1):
        delta = delta * cv2.contourArea(bbox_per_frame[i + 1]) / cv2.contourArea(bbox_per_frame[i])
    if delta >= THREAT_THRESHOLD:
        return True
    return False


def multiple_object_tracking(path: str):

    width = 300
    height = 250
    dim = (width, height)
    # video = open_video(path, 0)
    images = []
    for filename in os.listdir(path):
        new_path = os.path.join(path, filename)
        img = cv2.imread(new_path)
        if img is not None:
            images.append(img)
    fps = 1000
    # Calculate the delay between frames based on the frame rate
    # delay = int(10 / fps)  # Delay in milliseconds
    delay = 50
    # while True:

    # Read first frame - for detection
    # ok, frame = video.read()
    multi_tracker = init_tracker(images[0])

    for i in range(len(images)):
        # print(1)
        # print("i: ", i)
        # img_show("frame", frame, 0)
        # frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

        # if not ok:
        #     print('Cannot read video file')
        #     sys.exit()
        # print(2)

        # track objects

        # "objects: ", multi_tracker.getObjects()
        # print("frame index: ", i)
        # Read a new frame
        # ok, frame = video.read()
        # frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
        frame = images[i]
        # if not ok:
        #     break
        #
        # Start timer
        timer = cv2.getTickCount()

        ok, bboxes = multi_tracker.update(frame)
        # print(4)
        if cv2.getWindowProperty('sunset', cv2.WND_PROP_VISIBLE) < 1:
            print(" 1 ALL WINDOWS ARE CLOSED")
        # Draw bounding box
        if ok:
            draw_bbox(bboxes, frame)
        else:
            if cv2.getWindowProperty('sunset', cv2.WND_PROP_VISIBLE) < 1:
                print("2 ALL WINDOWS ARE CLOSED")
            print("tracking failure")
            multi_tracker = init_tracker(frame)
            if cv2.waitKey(delay) & 0xFF == -1:
                continue
            continue
        if cv2.getWindowProperty('sunset', cv2.WND_PROP_VISIBLE) < 1:
            print("3 ALL WINDOWS ARE CLOSED")
        display_tracker(frame, timer)
        if cv2.waitKey(delay) & 0xFF == 27:  # Wait for ESC key to exit
            break
        # print(6)


        # print(7)

        # # Exit if ESC pressed
        # if cv2.waitKey(1) & 0xff == EXIT_KEY:
        #     break


if __name__ == '__main__':
    multiple_object_tracking(
        "../../data/simulation_data")
