from datetime import datetime

import airsim
import numpy as np
import math
import time
from tools.pfm_tool import display_pdf, display_np
from matplotlib.backends.backend_pdf import PdfPages
import setup_path
import matplotlib.pyplot as plt
from perception.horizon_detection.find_direction import get_direction_from_image

# global variables
max_speed = 2
duration = 2  # duration of movement in seconds
epsilon = 0.01


# function to get the multirotor current yaw
def get_yaw():
    state = client.getMultirotorState()
    kinematics = state.kinematics_estimated
    yaw = math.degrees(airsim.utils.to_eularian_angles(kinematics.orientation)[2])
    return yaw


def Iteration(iteration, pdf=None):
    t1 = time.time()
    responses = client.simGetImages(
        [airsim.ImageRequest("1", airsim.ImageType.DepthPerspective, True)])  # depth in perspective projection
    prev_yaw = get_yaw()
    print(f"real stats: Yaw: {prev_yaw}")

    response = responses[0]
    if response.pixels_as_float:
        arr = airsim.get_pfm_array(response)
        print(f"response_time: {time.time() - t1}")
        # getting the desire yaw and pitch and algorithm
        yaw, pitch, success_rate = get_direction_from_image(arr, iteration, pdf)
        print(f"algorithm stats: Yaw: {yaw},  Pitch: {pitch}")
        responses = client.simGetImages(
            [airsim.ImageRequest("1", airsim.ImageType.DepthPerspective, True)])  # depth in perspective projection
        response = responses[0]
        if response.pixels_as_float:
            arr = airsim.get_pfm_array(response)
            display_np(np.clip(arr, 0, 10), f"image_after_computing{iteration}", pdf_list)

        fly_to(client, yaw, pitch, prev_yaw + yaw, success_rate)


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
    client.moveByVelocityBodyFrameAsync(0, 0, 0, 1, drivetrain=0,
                                        yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=get_yaw() + 30))
    time.sleep(0.5)


# yaw - the direction to fly in degrees
# pitch - the height to get in degrees, positive = down
# angle_sum - the facing direction in degrees
# success_rate - the freedom of flying fast (between 0-1)
def fly_to(client, yaw, pitch, angle_sum, success_rate):
    # if success rate < epsilon it means that the drone have no where to go
    if success_rate < epsilon:
        stop(success_rate)
        return

    vx = math.cos(math.radians(pitch)) * math.cos(math.radians(yaw)) * max_speed * success_rate  # forward speed in m/s
    vy = math.cos(math.radians(pitch)) * math.sin(math.radians(yaw)) * max_speed * success_rate  # lateral speed in m/s
    vz = math.sin(math.radians(pitch)) * max_speed * success_rate  # vertical speed in m/s
    client.moveByVelocityBodyFrameAsync(vx, vy, vz, duration, drivetrain=0,
                                        yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=angle_sum))


if __name__ == '__main__':
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    take_of()
    start = time.time()
    pdf_list = []
    for i in range(1000):
        print(f"\niter {i}\n")
        Iteration(i, pdf_list)
        if (time.time() - start > 60):
            break
    display_pdf(pdf_list, f"../mission_reports/mission_report_{datetime.now().strftime('%m_%d_%H%M')}.pdf")
    # that's enough fun for now. let's quit cleanly
    client.enableApiControl(False)
