from datetime import datetime

import airsim
import numpy as np
import os
import tempfile
import pprint
import cv2
import math
import time

from matplotlib.backends.backend_pdf import PdfPages

import setup_path
import matplotlib.pyplot as plt
from perception.horizon_detection.find_direction import get_direction_from_image

# connect to the AirSim simulator.


# **************************************
angle_sum = 0


def take_photo(iteration,pdf=None):
    global angle_sum
    responses = client.simGetImages(
        [airsim.ImageRequest("1", airsim.ImageType.DepthPerspective, True)])  # depth in perspective projection
    print('Retrieved images: %d' % len(responses))

    response = responses[0]
    if response.pixels_as_float:
        print("Type %d, size %d" % (response.image_type, len(response.image_data_float)))
        arr = airsim.get_pfm_array(response)

        yaw, pitch = get_direction_from_image(arr, iteration,pdf)
        print(f"{yaw}, {pitch}")
        angle_sum = (angle_sum + yaw) % 360
        fly_to(client, yaw, pitch, angle_sum)


# **************************************
def take_of():
    # airsim.wait_key('Press any key to takeoff')
    print("Taking off...")
    client.armDisarm(True)
    client.takeoffAsync().join()


# yaw - the direction to fly in degrees
# pitch - the height to get in degrees, positive = down
# sum - the facing direction in degrees
def fly_to(client, yaw, pitch, angle_sum):
    # set the drone's velocity vector to turn left at a speed of 5 m/s
    vx = math.cos(math.radians(yaw))  # forward speed in m/s
    vy = math.sin(math.radians(yaw))  # lateral speed in m/s
    vz = math.sin(math.radians(pitch))  # vertical speed in m/s
    duration = 4  # duration of movement in seconds

    client.moveByVelocityBodyFrameAsync(vx, vy, vz, duration, drivetrain=0,
                                        yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=angle_sum))


# # for loop for full circle
# # **********************************
# airsim.wait_key('press any key to move 2 second')
# print("moving 2 second")
#
# angle_sum = 0
# for i in range(8):
#     time.sleep(2)
#     yaw = 45
#     pitch = 0
#     angle_sum = (angle_sum + yaw) % 360
#     fly_to(client, yaw, pitch, angle_sum)
# # ***********************************
if __name__ == '__main__':
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    take_of()
    with PdfPages(f"../mission_reports/mission_report_{datetime.now().strftime('%m_%d_%H%M')}.pdf") as pdf:
        for i in range(20):
            take_photo(i,pdf)

    # that's enough fun for now. let's quit cleanly
    client.enableApiControl(False)
