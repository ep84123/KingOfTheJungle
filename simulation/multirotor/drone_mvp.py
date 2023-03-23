
import setup_path
import airsim 
from airsim.types import DrivetrainType, YawMode
import numpy as np
import os
import tempfile
import pprint
import cv2
import KingOfTheJungle.tools.pfm_tool as pfm_tool
import KingOfTheJungle.perception.blitz.ground_detection as ground_detection
import KingOfTheJungle.perception.blitz.clustering as perception

import open3d as o3d
# connect to the AirSim simulator
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

state = client.getMultirotorState()
print("state: %s" % pprint.pformat(state))

airsim.wait_key('Press any key to rotate')
client.rotateByYawRateAsync(-80,1).join()
airsim.wait_key('Press any key to move forward')
z=-2.5

result = client.moveOnPathAsync([airsim.Vector3r(0,-7,z)],2, 10, airsim.DrivetrainType.ForwardOnly, airsim.YawMode(False,0)).join()

responses = client.simGetImages([airsim.ImageRequest("1", airsim.ImageType.DepthPerspective, True)])#depth in perspective projection
print('Retrieved images: %d' % len(responses))

###test code
tmp_dir = os.path.join(tempfile.gettempdir(), "airsim_drone")
print ("Saving images to %s" % tmp_dir)
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

###

o3d.io.write_point_cloud("test.ply", pfm_tool.depth_np2pcd(airsim.get_pfm_array(responses[0])))
percepsion_data = perception.get_AABBs(np.asarray(pfm_tool.depth_np2pcd(airsim.get_pfm_array(responses[0])).points))
first_object = percepsion_data[0]

airsim.wait_key('Press any key to avoid collusion forward')
dx = first_object.get_extent()[0]
dy = first_object.get_extent()[2]
dist = first_object.get_center()[2]
client.rotateByYawRateAsync(-20,1).join()
result = client.moveOnPathAsync([airsim.Vector3r(-dx,-7,z)],2, 10, airsim.DrivetrainType.ForwardOnly, airsim.YawMode(False,0)).join()
client.rotateByYawRateAsync(20,1).join()
result = client.moveOnPathAsync([airsim.Vector3r(-dx,-7-dist-dy,z)],2, 10, airsim.DrivetrainType.ForwardOnly, airsim.YawMode(False,0)).join()

state = client.getMultirotorState()
print("state: %s" % pprint.pformat(state))

airsim.wait_key('Press any key to reset to original state')

client.reset()
client.armDisarm(False)

# that's enough fun for now. let's quit cleanly
client.enableApiControl(False)
