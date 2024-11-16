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


def plot_poses(ee_pose, goal_ee_pose, arm_pose, goal_arm_pose, phone_pose, goal_phone_pose, real_phone_pose, real_goal_phone_pose):
    """Plot the current and goal poses for ee, arm, and phone in 3D."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot poses as points with arrows for orientation
    poses = {
        'EE Current': (ee_pose, 'red'),
        'EE Goal': (goal_ee_pose, 'lightcoral'),
        'Arm Current': (arm_pose, 'blue'),
        'Arm Goal': (goal_arm_pose, 'lightblue'),
        'Phone Current': (phone_pose, 'green'),
        'Phone Goal': (goal_phone_pose, 'lightgreen'),
        'Real Phone': (real_phone_pose, 'darkorange'),
        'Real Phone Goal': (real_goal_phone_pose, 'bisque')
    }
    
    # Calculate plot limits
    all_positions = np.vstack([pose[:3, 3] for pose, _ in poses.values()])
    min_pos = np.min(all_positions, axis=0)
    max_pos = np.max(all_positions, axis=0)
    center = (min_pos + max_pos) / 2
    max_range = np.max(max_pos - min_pos)
    
    # RGB colors for xyz axes
    axis_colors = ['red', 'green', 'blue']
    
    for name, (pose, dot_color) in poses.items():
        # Plot position
        pos = pose[:3, 3]
        ax.scatter(pos[0], pos[1], pos[2], c=dot_color, marker='o', label=name)
        
        # Plot orientation arrows (using rotation matrix columns)
        arrow_length = max_range * 0.1  # Scale arrow length relative to plot size
        for i, axis in enumerate(['x', 'y', 'z']):
            direction = pose[:3, i] * arrow_length
            ax.quiver(pos[0], pos[1], pos[2],
                     direction[0], direction[1], direction[2],
                     color=axis_colors[i], alpha=0.8)
    
    # Set equal aspect ratio by setting axis limits
    for dim in [0, 1, 2]:
        mid = center[dim]
        ax.set_xlim(mid - max_range/2, mid + max_range/2)
        ax.set_ylim(mid - max_range/2, mid + max_range/2)
        ax.set_zlim(mid - max_range/2, mid + max_range/2)
    
    # Set labels and title
    ax.set_xlabel('X (red)')
    ax.set_ylabel('Y (green)')
    ax.set_zlabel('Z (blue)')
    ax.set_title('Robot Poses Visualization')
    ax.legend()
    
    # Make the plot more viewable
    ax.grid(True)
    
    plt.show(block=True)

def main():
    try:
        env = RobotEnv(track_phone=True)
        env.connect()
        env.home_joints()

        # obs = env.get_observation()
        # print(obs.ee_pose)

        arm_pose, ee_pose, phone_pose = env.get_world_pose()
        # print(arm_pose)
        # print(ee_pose)
        goal_ee_pose = ee_pose.copy()
        # Create rotation matrix for 45 degrees around z-axis
        rot_z = R.from_euler('y', 45, degrees=True).as_matrix()
        # Apply rotation to the orientation part of the pose matrix
        goal_ee_pose[:3, :3] = goal_ee_pose[:3, :3] @ rot_z

        # Calculate the fixed transform from arm to ee
        arm_to_ee_pose = np.linalg.inv(arm_pose) @ ee_pose
        # Calculate new arm pose that would put ee at goal
        goal_arm_pose = goal_ee_pose @ np.linalg.inv(arm_to_ee_pose)

        arm_to_phone = np.linalg.inv(arm_pose) @ phone_pose
        goal_phone_pose = goal_arm_pose @ arm_to_phone

        real_phone_pose = env.tracker.full_pose
        # Calculate the transformation from phone_pose to real_phone_pose
        phone_to_real = np.linalg.inv(phone_pose) @ real_phone_pose
        # Apply the same transformation to goal_phone_pose
        real_goal_phone_pose = goal_phone_pose @ phone_to_real

        goal_xyt = env.tracker.calculate_xyt(real_goal_phone_pose)

        base_pose = env.tracker.get_latest_position()

        print('ee_pose', ee_pose, '->', goal_ee_pose)
        print('arm_pose', arm_pose, '->', goal_arm_pose)
        print('phone_pose', phone_pose, '->', goal_phone_pose)
        print('base_pose', base_pose, '->', goal_xyt)
        
        # Add this line to visualize the poses
        plot_poses(ee_pose, goal_ee_pose, arm_pose, goal_arm_pose, phone_pose, goal_phone_pose, real_phone_pose, real_goal_phone_pose)
        
        done = False
        while not done:
            done = env.move_base_to(goal_xyt['x'], goal_xyt['y'], goal_xyt['yaw'])
            print(done)
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nExiting gracefully...")
    finally:
        cv2.destroyAllWindows()
        env.close()
        print("Connection closed")

if __name__ == "__main__":
    main()