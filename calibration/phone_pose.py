import sys
sys.path.append('.')

import cv2
from robot import Robot
from robot_wrapper import RobotEnv, plotter, robot_arm_chain, raw_joints_to_plottable
import numpy as np
import copy
from typing import Dict
import json
import matplotlib.pyplot as plt


def main():
    try:
        env = RobotEnv(track_phone=True)
        env.connect()

        phone_marker_pose = json.load(open('calibration/marker_pose.json'))['T_marker2base']

        while True:
            obs = env.get_observation()
            live_phone_pose = env.tracker.full_pose.copy()
            # Rotation matrix to convert from +y up to +z up coordinate system
            R_convert = np.array([
                [1, 0, 0, 0],
                [0, 0, -1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1]
            ])
            live_phone_pose = R_convert @ live_phone_pose

            T_phone2marker = np.eye(4)
            T_phone2marker[:3, :3] = np.array([
                [0, 1, 0],
                [1, 0, 0],
                [0, 0, -1],
            ])
            
            live_phone_pose = live_phone_pose @ T_phone2marker
            live_phone_pose[:3, 3] += live_phone_pose[:3, 1] * 0.06
            live_phone_pose[:3, 3] -= live_phone_pose[:3, 0] * 0.02

            base_pose = live_phone_pose @ np.linalg.inv(phone_marker_pose)

            plotter.update_plot(robot_arm_chain, raw_joints_to_plottable(obs.joints), plot_id="follower",
                        target_matrix=obs.ee_pose, other_frames=[live_phone_pose, base_pose])
            plt.draw()
            plt.pause(0.001)  # Small pause to allow for interaction
            
    except KeyboardInterrupt:
        print("\nExiting gracefully...")
    finally:
        cv2.destroyAllWindows()
        env.close()
        print("Connection closed")

if __name__ == "__main__":
    main()