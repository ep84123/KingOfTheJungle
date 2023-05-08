import airsim
import numpy as np
import os
import tempfile
import pprint
import cv2
import math

# connect to the AirSim simulator.
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)

state = client.getMultirotorState()
s = pprint.pformat(state)
print("state: %s" % s)

imu_data = client.getImuData()
s = pprint.pformat(imu_data)
print("imu_data: %s" % s)

barometer_data = client.getBarometerData()
s = pprint.pformat(barometer_data)
print("barometer_data: %s" % s)

magnetometer_data = client.getMagnetometerData()
s = pprint.pformat(magnetometer_data)
print("magnetometer_data: %s" % s)

gps_data = client.getGpsData()
s = pprint.pformat(gps_data)
print("gps_data: %s" % s)

airsim.wait_key('Press any key to takeoff')
print("Taking off...")
client.armDisarm(True)
client.takeoffAsync().join()

airsim.wait_key('Press any key to move forward 5 second')
print("moving forward 5 second")
# set the drone's velocity vector to turn left at a speed of 5 m/s
theta=0
alpha=60
vx = math.cos(alpha)  # forward speed in m/s
vy = math.sin(alpha)  # lateral speed in m/s
vz = 0  # vertical speed in m/s
duration = 5  # duration of movement in seconds

client.moveByVelocityAsync(vx, vy, vz, duration,yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=alpha))
theta=theta+alpha

airsim.wait_key('Press any key to move forward 5 second')
print("moving forward 2 second")

alpha=60
vx = math.cos(alpha)  # forward speed in m/s
vy = math.sin(alpha)  # lateral speed in m/s
vz = 0  # vertical speed in m/s
client.moveByVelocityAsync(vx, vy, vz, duration,yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=theta+alpha))
theta=theta+alpha
#*******************************
airsim.wait_key('finish test')
print("moving forward 2 second")
# set the drone's velocity vector to turn left at a speed of 5 m/s
vx = 2  # forward speed in m/s
vy = 2  # lateral speed in m/s
vz = 0  # vertical speed in m/s
duration = 5  # duration of movement in seconds
# set the yaw angle to turn the drone by
angle_degrees = 45  # 45 degree turn
angle_radians = math.radians(angle_degrees)  # convert to radians

# Get the drone's current pose and orientation
pose = client.simGetVehiclePose()
orientation = pose.orientation

# Convert the orientation quaternion to Euler angles
_, _, yaw = airsim.to_eularian_angles(orientation)

# Add 45 degrees to the current yaw angle
new_yaw = yaw + angle_radians

client.moveByVelocityAsync(vx, vy, vz, duration,yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=30))


#*******************************
# wait for the drone to complete the movement
#client.simPause(False)  # if there is surprise action it will ignore it so for now we are ignoring this.

airsim.wait_key('Press any key to turn by 45 degrees')
print("turning by 45 degrees")
# set the yaw angle to turn the drone by
angle_degrees = 45  # 45 degree turn
angle_radians = math.radians(angle_degrees)  # convert to radians

# Get the drone's current pose and orientation
pose = client.simGetVehiclePose()
orientation = pose.orientation

# Convert the orientation quaternion to Euler angles
_, _, yaw = airsim.to_eularian_angles(orientation)

# Add 45 degrees to the current yaw angle
new_yaw = yaw + angle_radians

# Move the drone to the new yaw angle
client.moveByAngleZAsync(0, 0, -1, new_yaw, 5)
# wait for the drone to complete the movement
client.simPause(False)  # if there is surprise action it will ignore it so for now we are ignoring this.

airsim.wait_key('Press any key to move forward 5 second and turning 45 degrees')
print("moving forward 5 second second and turning 45 degrees")
# set the yaw angle to turn the drone by
angle_degrees = 45  # 45 degree turn
angle_radians = math.radians(angle_degrees)  # convert to radians

# set the drone's velocity vector and orientation to move forward and turn
speed = 5  # forward speed in m/s
vx = speed  # forward speed in m/s
vy = 0  # lateral speed in m/s
vz = 0  # vertical speed in m/s
duration = 5  # duration of movement in seconds

# Get the drone's current pose and orientation
pose = client.simGetVehiclePose()
orientation = pose.orientation

# Convert the orientation quaternion to Euler angles
_, _, yaw = airsim.to_eularian_angles(orientation)

# Add 45 degrees to the current yaw angle
new_yaw = yaw + airsim.utils.to_quaternion(0, 0, angle_radians)

# Move the drone to the new yaw angle
client.moveByAngleZAsync(0, 0, -1, new_yaw, duration)

# Move the drone forward and laterally
airsim_resp = client.moveByVelocityAsync(vx * math.cos(angle_radians) + vy * math.sin(angle_radians),
                                         vy * math.cos(angle_radians) - vx * math.sin(angle_radians), vz, duration)

state = client.getMultirotorState()
print("state: %s" % pprint.pformat(state))

airsim.wait_key('Press any key to move vehicle to (-10, 10, -10) at 5 m/s')
client.moveToPositionAsync(-10, 10, -10, 5).join()

client.hoverAsync().join()

state = client.getMultirotorState()
print("state: %s" % pprint.pformat(state))

airsim.wait_key('Press any key to take images')
# get camera images from the car
responses = client.simGetImages([
    airsim.ImageRequest("0", airsim.ImageType.DepthVis),  # depth visualization image
    airsim.ImageRequest("1", airsim.ImageType.DepthPerspective, True),  # depth in perspective projection
    airsim.ImageRequest("1", airsim.ImageType.Scene),  # scene vision image in png format
    airsim.ImageRequest("1", airsim.ImageType.Scene, False, False)])  # scene vision image in uncompressed RGBA array
print('Retrieved images: %d' % len(responses))

tmp_dir = os.path.join(tempfile.gettempdir(), "airsim_drone")
print("Saving images to %s" % tmp_dir)
try:
    os.makedirs(tmp_dir)
except OSError:
    if not os.path.isdir(tmp_dir):
        raise

for idx, response in enumerate(responses):

    filename = os.path.join(tmp_dir, str(idx))

    if response.pixels_as_float:
        print("Type %d, size %d" % (response.image_type, len(response.image_data_float)))
        airsim.write_pfm(os.path.normpath(filename + '.pfm'), airsim.get_pfm_array(response))
    elif response.compress:  # png format
        print("Type %d, size %d" % (response.image_type, len(response.image_data_uint8)))
        airsim.write_file(os.path.normpath(filename + '.png'), response.image_data_uint8)
    else:  # uncompressed array
        print("Type %d, size %d" % (response.image_type, len(response.image_data_uint8)))
        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)  # get numpy array
        img_rgb = img1d.reshape(response.height, response.width, 3)  # reshape array to 4 channel image array H X W X 3
        cv2.imwrite(os.path.normpath(filename + '.png'), img_rgb)  # write to png

airsim.wait_key('Press any key to reset to original state')

client.reset()
client.armDisarm(False)

# that's enough fun for now. let's quit cleanly
client.enableApiControl(False)