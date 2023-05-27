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
max_speed = 2
duration = 4  # duration of movement in seconds
epsilon = 0.01


def take_photo(iteration, pdf=None):
    global angle_sum
    responses = client.simGetImages(
        [airsim.ImageRequest("1", airsim.ImageType.DepthPerspective, True)])  # depth in perspective projection
    print('Retrieved images: %d' % len(responses))

    response = responses[0]
    if response.pixels_as_float:
        print("Type %d, size %d" % (response.image_type, len(response.image_data_float)))
        arr = airsim.get_pfm_array(response)
        yaw, pitch, success_rate = get_direction_from_image(arr, iteration, pdf)
        print(f"{yaw}, {pitch}")
        angle_sum = (angle_sum + yaw) % 360
        fly_to(client, yaw, pitch, angle_sum, success_rate)


# **************************************
def take_of():
    # airsim.wait_key('Press any key to takeoff')
    print("Taking off...")
    client.armDisarm(True)
    time.sleep(2)
    client.takeoffAsync().join()


def stop(success_rate):
    global angle_sum
    # if success rate < epsilon it means that the drone have no where to go
    print("STOPPPPPPPPPPPPPPPP")
    angle_sum += 45
    client.moveByVelocityBodyFrameAsync(0, 0, 0, 1, drivetrain=0,
                                        yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=angle_sum))

    time.sleep(0.5)


# yaw - the direction to fly in degrees
# pitch - the height to get in degrees, positive = down
# angle_sum - the facing direction in degrees
# success_rate - the freedom of flying fast (between 0-1)
def fly_to(client, yaw, pitch, angle_sum, success_rate):
    # if success rate < epsilon it means that the drone have no where to go
    if success_rate < epsilon:
        stop(success_rate)
        return    # set the drone's velocity vector to turn left at a speed of 5 m/s
    vx = math.cos(math.radians(pitch)) * math.cos(math.radians(yaw))  # forward speed in m/s
    vy = math.cos(math.radians(pitch)) * math.sin(math.radians(yaw))  # lateral speed in m/s
    vz = math.sin(math.radians(pitch))  # vertical speed in m/s
    vx = max_speed * success_rate * vx
    vy = max_speed * success_rate * vy
    vz = max_speed * success_rate * vz

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
    start = time.time()
    with PdfPages(f"../mission_reports/mission_report_{datetime.now().strftime('%m_%d_%H%M')}.pdf") as pdf:
        for i in range(1000):
            take_photo(i,pdf)
            if(time.time() - start > 100):
                break

    # that's enough fun for now. let's quit cleanly
    client.enableApiControl(False)
