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

        posemat = np.eye(4)
        posemat[:3, :3] = R.from_euler('xyz', [45, 0, 0], degrees=True).as_matrix()
        posemat[0, 3] = 0
        posemat[1, 3] = 0.15
        posemat[2, 3] = 0.1
        env.move_to_pose(posemat, duration=2)
        # env.home_joints()
        time.sleep(1)

        poses = dict()
        index = 0

        def move_ee_test(goal_ee_pose, wait_base):
            nonlocal index  # Access the index variable from the outer scope
            index += 1  # Increment the index
            goal_arm_base_pose = env.move_ee_to(goal_ee_pose, wait_base=wait_base)

            poses[f'desired_arm_{index}'] = (goal_arm_base_pose.copy(), 'lightcoral')
            poses[f'desired_ee_{index}'] = (goal_ee_pose.copy(), 'lightcoral')
            # time.sleep(3)
            actual_base_pose, actual_ee_pose, _ = env.get_world_pose()
            poses[f'actual_ee_{index}'] = (actual_ee_pose, 'lightcoral')
            poses[f'actual_arm_{index}'] = (actual_base_pose, 'lightcoral')

        arm_pose, ee_pose, _ = env.get_world_pose()
        # z_vals = np.linspace(-0.05, 0.03, 5)

        # for i in range(5):
        #     rot_z = R.from_euler('z', 30, degrees=True).as_matrix()
        #     ee_pose[:3, :3] = rot_z @ ee_pose[:3, :3]
        #     ee_pose[2, 3] = z_vals[i]
        #     move_ee_test(ee_pose.copy())

        # ee_pose[2, 3] += 0.05
        # move_ee_test(ee_pose)

        # ee_pose[:3, 3] += arm_pose[:3, 0] * 0.1
        # move_ee_test(ee_pose)

        # ee_pose[:3, 3] += arm_pose[:3, 1] * 0.1
        # move_ee_test(ee_pose)

        # ee_pose[:3, 3] += arm_pose[:3, 1] * 0.1
        # move_ee_test(ee_pose)

        # ee_pose[:3, 3] -= arm_pose[:3, 0] * 0.1
        # move_ee_test(ee_pose)

        # rot_z = R.from_euler('z', 90, degrees=True).as_matrix()
        # ee_pose[:3, :3] = rot_z @ ee_pose[:3, :3]
        # move_ee_test(ee_pose)

        # rot_x = R.from_euler('x', -45, degrees=True).as_matrix()
        # ee_pose[:3, :3] = ee_pose[:3, :3] @ rot_x
        # move_ee_test(ee_pose)

        # rot_y = R.from_euler('y', -45, degrees=True).as_matrix()
        # ee_pose[:3, :3] = ee_pose[:3, :3] @ rot_y
        # move_ee_test(ee_pose)

        origin = ee_pose.copy()

        for i in range(100):
            # ee_pose[2, 3] = np.sin(i * 0.1) * 0.03 + 0.05
            ee_pose = origin.copy()
            # ee_pose[:3, 3] += arm_pose[:3, 0] * np.sin(i * 0.1) * 0.1
            # ee_pose[:3, 3] += ee_pose[:3, 2] * np.sin(i * 0.1) * 0.05
            # ee_pose[:3, 3] += ee_pose[:3, 0] * np.cos(i * 0.1) * 0.05
            rot_z = R.from_euler('z', i, degrees=True).as_matrix()
            ee_pose[:3, :3] = rot_z @ ee_pose[:3, :3]

            for _ in range(5):
                move_ee_test(ee_pose, False)
                time.sleep(0.02)

        env.stop_base()

        # plotter.plot_poses({k: v for k, v in poses.items() if k.startswith('desired_ee')})
        plotter.plot_poses({k: v for k, v in poses.items() if k.startswith('actual_ee')}, dots_only=True)

    except KeyboardInterrupt:
        print("\nExiting gracefully...")
    finally:
        cv2.destroyAllWindows()
        env.close()
        print("Connection closed")

if __name__ == "__main__":
    main()