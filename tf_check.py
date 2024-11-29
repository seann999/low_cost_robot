import sys
sys.path.append('.')

import cv2
from robot import Robot
from robot_wrapper import RobotEnv, joints_to_posemat, pose7d_to_joints, plotter
import numpy as np
import copy
from scipy.spatial.transform import Rotation as R
import threading
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def main():
    try:
        env = RobotEnv(track_phone=True)
        env.connect()

        init_config = [2088, 2071, 1773, 3058, 2078, 2890]
        env.move_to_joints(init_config, duration=3.0)

        # posemat = np.eye(4)
        # posemat[:3, :3] = R.from_euler('xyz', [45, 0, 0], degrees=True).as_matrix()
        # posemat[0, 3] = 0
        # posemat[1, 3] = 0.15
        # posemat[2, 3] = 0.1
        # env.move_to_pose(posemat, duration=2)
        # # env.home_joints()
        # time.sleep(1)

        poses = dict()
        obs = env.get_observation()
        # poses['origin'] = (np.eye(4), 'black')
        poses['cam_pose'] = (obs.cam_pose, 'black')
        poses['ee_pose'] = (obs.ee_pose, 'black')
        poses['arm_base_pose'] = (obs.arm_base_pose, 'black')
        poses['phone_pose'] = (obs.phone_pose, 'black')
        # poses['calib_phone_pose'] = (env.calib_phone_pose, 'black')
        # poses['calib_world_base_pose'] = (env.calib_world_base_pose, 'black')
        poses['calib_new_world_base_pose'] = (env.calib_new_world_base_pose, 'black')

        print('distance B:', np.linalg.norm(obs.arm_base_pose[:3, 3] - obs.phone_pose[:3, 3]))
        print('arm_base_pose:', obs.arm_base_pose)
        print('phone_pose:', obs.phone_pose)

        # plotter.plot_poses({k: v for k, v in poses.items() if k.startswith('desired_ee')})
        plotter.plot_poses(poses)

    except KeyboardInterrupt:
        print("\nExiting gracefully...")
    finally:
        cv2.destroyAllWindows()
        env.close()
        print("Connection closed")

if __name__ == "__main__":
    main()