import time
from robot import Robot
from robot_wrapper import RobotEnv
from simple_pid import PID
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def calculate_offset_poses(poses, offset):
    """Calculate poses with x/y offset based on yaw angle"""
    offset_poses = poses.copy()
    offset_poses[:, 0] += offset * np.cos(poses[:, 2])  # x += offset * cos(yaw)
    offset_poses[:, 1] += offset * np.sin(poses[:, 2])  # y += offset * sin(yaw)
    return offset_poses

def objective_function(offset, poses):
    """Function to minimize - variance of x and y positions"""
    offset_poses = calculate_offset_poses(poses, offset)
    # Calculate variance of x and y coordinates and sum them
    var_x = np.var(offset_poses[:, 0])
    var_y = np.var(offset_poses[:, 1])
    return var_x + var_y

def main():
    try:
        env = RobotEnv()
        env.connect()
        env.home_joints()
        poses = []

        # for yaw in np.linspace(0, np.pi*2.0, 10):
        #     done = False
        #     while not done:
        #         done = env.move_base_to(0, 0, yaw)
        #         time.sleep(0.1)
        #     time.sleep(1)
        # env.stop_base()
        # exit()

        for _ in range(200):
            env.send_base([0, 0, 0.2])
            time.sleep(0.1)
            # env.stop_base()
            pose = env.tracker.get_latest_position()
            poses.append([pose['x'], pose['y'], pose['yaw']])

        # Convert poses list to numpy array for easier plotting
        poses = np.array(poses)
        env.stop_base()
        
        # Find optimal offset
        result = minimize(objective_function, x0=0.0, args=(poses,))
        optimal_offset = result.x[0]
        print(f"Optimal offset: {optimal_offset:.3f} meters")
        
        # Calculate offset-corrected poses
        offset_poses = calculate_offset_poses(poses, optimal_offset)
        
        # Create single comparison plot
        plt.figure(figsize=(8, 8))
        plt.scatter(poses[:, 0], poses[:, 1], c='blue', marker='o', label='Original', alpha=0.6)
        plt.scatter(offset_poses[:, 0], offset_poses[:, 1], c='red', marker='o', label='Offset-corrected', alpha=0.6)
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title(f'Trajectory Comparison\n(optimal offset = {optimal_offset:.3f}m)')
        plt.grid(True)
        plt.axis('equal')
        plt.legend()
        plt.tight_layout()
        plt.show(block=True)
        
    except KeyboardInterrupt:
        print("\nExiting gracefully...")
    finally:
        plt.close()  # Close the plot window
        env.close()
        print("Connection closed")

if __name__ == "__main__":
    main()
