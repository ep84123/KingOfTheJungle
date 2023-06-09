#  Object tracking for obstacle detection
import cv2
import sys
from utils import get_random_color, open_video, calc_fps, EXIT_KEY
from vidstab.VidStab import VidStab
from edge_detection import detect_edge_tree

tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']

THREAT_THRESHOLD = 3


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
    contours = detect_edge_tree(frame)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        bbox_list.append((x, y, w, h))
    # color = get_random_color()
    if (0, 0, 0, 0) in bbox_list:
        bbox_list.remove((0, 0, 0, 0))
    print(f"tracking {len(bbox_list)}")
    return  bbox_list


# Draw the bounding boxes on the video frame
def draw_bbox(bboxes, color, frame):
    for i, bbox in enumerate(bboxes):
        point_1 = (int(bbox[0]), int(bbox[1]))
        point_2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, point_1, point_2, get_random_color(), 5)
        M = cv2.moments(bbox)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        print(cY, cX)
        cv2.circle(frame, (cX, cY), 10, color, -1)
        cv2.waitKey(0)


def tracking_failure(frame):
    cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)


def display_tracker(frame, timer):
    # Display tracker type on frame
    cv2.putText(frame, "KCF Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

    # Display FPS on frame
    cv2.putText(frame, "FPS : " + calc_fps(timer), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

    # Display result
    cv2.imshow("Tracking, press Esc to exit", frame)


def is_threat(bbox_per_frame) -> bool:
    delta = 1
    for i in range(len(bbox_per_frame) - 1):
        delta = delta * cv2.contourArea(bbox_per_frame[i + 1]) / cv2.contourArea(bbox_per_frame[i])
    if delta >= THREAT_THRESHOLD:
        return True
    return False


def multiple_object_tracking(path: str):
    multi_tracker = cv2.legacy.MultiTracker_create()
    video = open_video(path, 2e4)
    stabilizer = VidStab()
    fps = video.get(cv2.CAP_PROP_FPS)
    print("fps: ", fps)
    # Read first frame - for detection
    ok, frame = video.read()
    if not ok:
        print('Cannot read video file')
        sys.exit()

    bbox_list = get_bbox_from_detector(frame)
    for bbox in bbox_list:
        print(bbox)
        # Add tracker to the multi-object tracker - TODO: Why some tracker aren't working?
        multi_tracker.add(create_tracker(tracker_types[4]), frame, bbox)

    i = 0
    # track objects
    while True:
        i += 1
        # Read a new frame
        ok, frame = video.read()
        if not ok:
            break
        # stabilized_frame = stabilizer.stabilize_frame(input_frame=frame, smoothing_window=30)
        # if stabilized_frame is None:
        #     # There are no more frames available to stabilize
        #     break
        # Start timer
        if i%10 == 0:
            bbox_list.extend(get_bbox_from_detector(frame))
            boxes_ids = multi_tracker.update(frame)
            for box_id in boxes_ids:
                x, y, w, h, id = box_id
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        timer = cv2.getTickCount()

        ok, bboxes = multi_tracker.update(frame)

        # Draw bounding box
        if ok:
            draw_bbox(bboxes, frame)
        else:
            tracking_failure(frame)
        display_tracker(frame, timer)

        # Exit if ESC pressed
        if cv2.waitKey(1) & 0xff == EXIT_KEY:
            break
