#  Object tracking for obstacle detection
# ----------------- VERSION SINGLE OBJECT ------------------------------
import cv2
import sys
import numpy as np
from random import randint
import keyboard

def single_object_tracking(path: str):
    # Set up tracker.
    # Instead of MIL, you can also use
    #
    # tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
    # tracker_type = tracker_types[3]
    #
    # if tracker_type == 'BOOSTING':
    #     tracker = cv2.TrackerBoosting_create() - SLOW
    # if tracker_type == 'MIL':
    #     tracker = cv2.TrackerMIL_create() - SLOW
    # if tracker_type == 'KCF':
    #     tracker = cv2.TrackerKCF_create() - FAST, NOT CHAINING SIZE
    # if tracker_type == 'TLD':
    #     tracker = cv2.TrackerTLD_create()- SUPER SLOW  - "Also, tracks best over scale changes"?
    # if tracker_type == 'MEDIANFLOW':
    #     tracker = cv2.TrackerMedianFlow_create() - FAST, NOT SO ACCURATE, CHAINING SIZE
    # if tracker_type == 'GOTURN':
    #     tracker = cv2.TrackerGOTURN_create() - NOT WORKING, FIX?
    # if tracker_type == 'MOSSE':
    #     tracker = cv2.TrackerMOSSE_create() - REALLY FAST, NOT CHAINING SIZE
    # if tracker_type == "CSRT":
    #     tracker = cv2.TrackerCSRT_create() - SLOW
    tracker = cv2.legacy.TrackerMedianFlow_create()
    # Read video
    video = cv2.VideoCapture(path)
    video.set(cv2.CAP_PROP_POS_MSEC, 1e4)
    low_green = np.array([25, 52, 72])
    high_green = np.array([102, 255, 255])
    # Exit if video not opened.
    if not video.isOpened():
        print("Could not open video")
        sys.exit()

    # Read first frame.
    ok, frame = video.read()
    if not ok:
        print('Cannot read video file')
        sys.exit()
    bounding_box_list = []
    color_list = []
    from_center = False  # Draw bounding box from upper left
    show_cross_hair = False  # Don't show the cross hair
    while True:

        print("press enter or space after marking the object")
        # Press ENTER or SPACE after you've drawn the bounding box
        bounding_box = cv2.selectROI('Multi-Object Tracker', frame, from_center,
                                     show_cross_hair)

        # Add a bounding box
        bounding_box_list.append(bounding_box)

        # Add a random color_list
        blue = 255  # randint(127, 255)
        green = 0  # randint(127, 255)
        red = 255  # randint(127, 255)
        color_list.append((blue, green, red))
        # Press 'q' (make sure you click on the video frame so that it is the
        # active window) to start object tracking. You can press another key
        # if you want to draw another bounding box.
        print("\nPress q to begin tracking objects or press " +
              "another key to draw the next bounding box\n")
        # Exit if ESC pressed
        k = cv2.waitKey(0)
        if k == ord('q'):
            print("exit")
            break
        if keyboard.read_key() == 'a':
            print("a")
            break

    print(f"tracking {len(color_list)}, {len(bounding_box_list)}")
    # # Define an initial bounding box
    # bbox = (287, 23, 86, 320)
    #
    # # Uncomment the line below to select a different bounding box
    # bbox = cv2.selectROI(frame, False)
    multi_tracker = cv2.legacy.MultiTracker_create()

    for bbox in bounding_box_list:
        # Add tracker to the multi-object tracker
        multi_tracker.add(cv2.legacy.TrackerMedianFlow_create(), frame, bbox)
    # Initialize tracker with first frame and bounding box
    # ok = tracker.init(frame, bbox)
    # cap = video.set(cv2.CAP_PROP_FPS, 30)
    while True:
        # Read a new frame
        ok, frame = video.read()
        if not ok:
            break

        # Start timer
        timer = cv2.getTickCount()

        # Update tracker
        ok, bboxes = multi_tracker.update(frame)

        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        # Draw bounding box
        if ok:

            # Draw the bounding boxes on the video frame
            for i, bbox in enumerate(bboxes):
                point_1 = (int(bbox[0]), int(bbox[1]))
                point_2 = (int(bbox[0] + bbox[2]),
                           int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, point_1, point_2, color_list[i], 5)
        else:
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        # Display tracker type on frame
        cv2.putText(frame, "KCF Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        # Display FPS on frame
        cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        # Display result
        cv2.imshow("Tracking", frame)

        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break

def multiple_obejcts_tracking(filename):

    # Make sure the video file is in the same directory as your code
    # file_prefix = 'fish'
    file_size = (1920, 1080)  # Assumes 1920x1080 mp4

    # We want to save the output to a video file
    output_frames_per_second = 60.0

    # OpenCV has a bunch of object tracking algorithms. We list them here.
    type_of_trackers = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN',
                        'MOSSE', 'CSRT']

    # CSRT is accurate but slow. You can try others and see what results you get.
    desired_tracker = 'CSRT'

    # Generate a MultiTracker object
    multi_tracker = cv2.legacy.MultiTracker_create()

    # Set bounding box drawing parameters
    from_center = False  # Draw bounding box from upper left
    show_cross_hair = False  # Don't show the cross hair

    # def generate_tracker(type_of_tracker):
    #     """
    #     Create object tracker.
    #
    #     :param type_of_tracker string: OpenCV tracking algorithm
    #     """
    #     if type_of_tracker == type_of_trackers[0]:
    #         tracker = cv2.TrackerBoosting_create()
    #     elif type_of_tracker == type_of_trackers[1]:
    #         tracker = cv2.TrackerMIL_create()
    #     elif type_of_tracker == type_of_trackers[2]:
    #         tracker = cv2.TrackerKCF_create()
    #     elif type_of_tracker == type_of_trackers[3]:
    #         tracker = cv2.TrackerTLD_create()
    #     elif type_of_tracker == type_of_trackers[4]:
    #         tracker = cv2.TrackerMedianFlow_create()
    #     elif type_of_tracker == type_of_trackers[5]:
    #         tracker = cv2.TrackerGOTURN_create()
    #     elif type_of_tracker == type_of_trackers[6]:
    #         tracker = cv2.TrackerMOSSE_create()
    #     elif type_of_tracker == type_of_trackers[7]:
    #         tracker = cv2.TrackerCSRT_create()
    #     else:
    #         tracker = None
    #         print('The name of the tracker is incorrect')
    #         print('Here are the possible trackers:')
    #         for track_type in type_of_trackers:
    #             print(track_type)
    #     return tracker


    # Load a video
    cap = cv2.VideoCapture(filename)
    cap.set(cv2.CAP_PROP_POS_MSEC, 1e4)

    # Create a VideoWriter object so we can save the video output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    result = cv2.VideoWriter("first_output",
                             fourcc,
                             output_frames_per_second,
                             file_size)

    # Capture the first video frame
    success, frame = cap.read()

    bounding_box_list = []
    color_list = []
    i = 0
    # Do we have a video frame? If true, proceed.
    if success:

        while i<5:

            # Draw a bounding box over all the objects that you want to track_type
            # Press ENTER or SPACE after you've drawn the bounding box
            bounding_box = cv2.selectROI('Multi-Object Tracker', frame, from_center,
                                         show_cross_hair)

            # Add a bounding box
            bounding_box_list.append(bounding_box)

            # Add a random color_list
            blue = 255  # randint(127, 255)
            green = 0  # randint(127, 255)
            red = 255  # randint(127, 255)
            color_list.append((blue, green, red))
            # Press 'q' (make sure you click on the video frame so that it is the
            # active window) to start object tracking. You can press another key
            # if you want to draw another bounding box.
            print("\nPress q to begin tracking objects or press " +
                  "another key to draw the next bounding box\n")
            i+=1
            # Exit if ESC pressed
            k = cv2.waitKey(1) & 0xff
            if k == ord('q'):
                print("exit")
                break

        cv2.destroyAllWindows()

        print("\nTracking objects. Please wait...")

        # Set the tracker
        type_of_tracker = desired_tracker

        for bbox in bounding_box_list:
            # Add tracker to the multi-object tracker
            multi_tracker.add(cv2.legacy.TrackerMedianFlow_create(), frame, bbox)

        # Process the video
        while cap.isOpened():
            print("proccesisng")
            # Capture one frame at a time
            success, frame = cap.read()
            # Start timer
            timer = cv2.getTickCount()
            # Do we have a video frame? If true, proceed.
            if success:

                # Update the location of the bounding boxes
                success, bboxes = multi_tracker.update(frame)

                # Draw the bounding boxes on the video frame
                for i, bbox in enumerate(bboxes):
                    point_1 = (int(bbox[0]), int(bbox[1]))
                    point_2 = (int(bbox[0] + bbox[2]),
                               int(bbox[1] + bbox[3]))
                    cv2.rectangle(frame, point_1, point_2, color_list[i], 5)
                fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

                # Write the frame to the output video file
                # result.write(frame)
                cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50),
                            2)

                cv2.imshow("Tracking", frame)

            # No more video frames left
            else:
                break

            # Exit if ESC pressed
            k = cv2.waitKey(1) & 0xff
            if k == 27:
                break

    # Stop when the video is finished
    cap.release()

    # Release the video recording
    result.release()

