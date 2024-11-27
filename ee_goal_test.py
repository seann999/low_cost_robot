import sys
sys.path.append('.')

import cv2
from robot import Robot
from robot_wrapper import RobotEnv, joints_to_posemat, pose7d_to_joints
import numpy as np
import copy
from scipy.spatial.transform import Rotation as R
from typing import Dict
import threading
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def main():
    try:
        env = RobotEnv(track_phone=True)
        env.connect()
        env.home_joints()

        desired_ee_poses = []
        actual_ee_poses = []
        desired_arm_poses = []
        actual_arm_poses = []

        arm_pose, ee_pose, _ = env.get_world_pose()
        # Calculate the fixed transform from arm to ee
        arm_to_ee_pose = np.linalg.inv(arm_pose) @ ee_pose
        goal_ee_pose = ee_pose.copy()

        for _ in range(50):
            arm_pose, _, phone_pose = env.get_world_pose()
            rot_z = R.from_euler('z', 10, degrees=True).as_matrix()
            goal_ee_pose[:3, :3] = rot_z @ goal_ee_pose[:3, :3]

            desired_ee_poses.append(goal_ee_pose.copy())
            
            # Calculate new arm pose that would put ee at goal
            goal_arm_pose = goal_ee_pose @ np.linalg.inv(arm_to_ee_pose)
            desired_arm_poses.append(goal_arm_pose.copy())

            env.move_arm_base_to(goal_arm_pose)

            actual_arm_pose, actual_ee_pose, _ = env.get_world_pose()
            actual_ee_poses.append(actual_ee_pose.copy())
            actual_arm_poses.append(actual_arm_pose.copy())

            # base_pose = env.tracker.get_latest_position()

            # print('ee_pose', ee_pose, '->', goal_ee_pose)
            # print('arm_pose', arm_pose, '->', goal_arm_pose)
            # print('phone_pose', phone_pose, '->', goal_phone_pose)
            # print('base_pose', base_pose, '->', goal_xyt)
            
            # # Add this line to visualize the poses
            # poses_to_plot = {
            #     'Origin': (np.eye(4), 'black'),
            #     'EE Current': (ee_pose, 'red'),
            #     'EE Goal': (goal_ee_pose, 'lightcoral'),
            #     'Arm Current': (arm_pose, 'blue'),
            #     'Arm Goal': (goal_arm_pose, 'lightblue'),
            #     'Phone Current': (phone_pose, 'green'),
            #     'Phone Goal': (goal_phone_pose, 'lightgreen'),
            #     'Real Phone': (real_phone_pose, 'darkorange'),
            #     'Real Phone Goal': (real_goal_phone_pose, 'bisque')
            # }
            # # plot_poses(poses_to_plot)
            
            time.sleep(1)

        # Create separate dictionaries for desired and actual poses
        # desired_poses = {}
        # actual_poses = {}
        # for i, (desired_arm, desired_ee, actual_arm, actual_ee) in enumerate(zip(desired_arm_poses, desired_ee_poses, actual_arm_poses, actual_ee_poses)):
        #     desired_poses[f'Desired Arm {i}'] = (desired_arm, 'red')
        #     desired_poses[f'Desired EE {i}'] = (desired_ee, 'lightcoral')
        #     actual_poses[f'Actual Arm {i}'] = (actual_arm, 'blue')
        #     actual_poses[f'Actual EE {i}'] = (actual_ee, 'lightblue')
        
        # # Plot desired and actual poses separately
        # plot_poses(desired_poses)
        # plot_poses(actual_poses)

    except KeyboardInterrupt:
        print("\nExiting gracefully...")
    finally:
        cv2.destroyAllWindows()
        env.close()
        print("Connection closed")

if __name__ == "__main__":
    main()