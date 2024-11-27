import time
from robot import Robot
from robot_wrapper import RobotEnv
from simple_pid import PID
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def calculate_offset_poses(poses, offsets):
    """Calculate poses with longitudinal and lateral offsets based on yaw angle"""
    long_offset, lat_offset = offsets
    offset_poses = poses.copy()
    # Longitudinal offset (forward/backward)
    offset_poses[:, 0] += long_offset * np.cos(poses[:, 2])
    offset_poses[:, 1] += long_offset * np.sin(poses[:, 2])
    # Lateral offset (left/right)
    offset_poses[:, 0] += lat_offset * np.cos(poses[:, 2] + np.pi/2)
    offset_poses[:, 1] += lat_offset * np.sin(poses[:, 2] + np.pi/2)
    return offset_poses

def objective_function(offsets, poses):
    """Function to minimize - variance of x and y positions"""
    offset_poses = calculate_offset_poses(poses, offsets)
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

        for _ in range(3):
            for _ in range(50):
                env.send_base([0, 0, 0.2])
                time.sleep(0.1)
                # env.stop_base()
                pose = env.tracker.get_latest_position()
                poses.append([pose['x'], pose['y'], pose['yaw']])
            for _ in range(50):
                env.send_base([0, 0, -0.2])
                time.sleep(0.1)
                # env.stop_base()
                pose = env.tracker.get_latest_position()
                poses.append([pose['x'], pose['y'], pose['yaw']])

        # Convert poses list to numpy array for easier plotting
        poses = np.array(poses)
        env.stop_base()
        
        # Find optimal offsets (longitudinal and lateral)
        result = minimize(objective_function, x0=[0.0, 0.0], args=(poses,))
        optimal_long_offset, optimal_lat_offset = result.x
        print(f"Optimal longitudinal offset: {optimal_long_offset:.3f} meters")
        print(f"Optimal lateral offset: {optimal_lat_offset:.3f} meters")
        
        # Calculate offset-corrected poses
        offset_poses = calculate_offset_poses(poses, [optimal_long_offset, optimal_lat_offset])
        
        # Create single comparison plot
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Create color gradients based on time
        time_colors = np.linspace(0, 1, len(poses))
        
        # Plot original trajectory with arrows
        ax.quiver(poses[:, 0], poses[:, 1], 
                  np.cos(poses[:, 2]), np.sin(poses[:, 2]),
                  time_colors, cmap='Blues', 
                  scale=30, alpha=0.6, label='Original')
        
        # Plot offset-corrected trajectory with arrows
        ax.quiver(offset_poses[:, 0], offset_poses[:, 1],
                  np.cos(offset_poses[:, 2]), np.sin(offset_poses[:, 2]),
                  time_colors, cmap='Reds',
                  scale=30, alpha=0.6, label='Offset-corrected')
        
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title(f'Trajectory Comparison\n(long={optimal_long_offset:.3f}m, lat={optimal_lat_offset:.3f}m)')
        ax.grid(True)
        ax.axis('equal')
        ax.legend()
        
        # Fix colorbar by setting array data
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(0, len(poses)))
        sm.set_array(time_colors)
        plt.colorbar(sm, ax=ax, label='Time Progress')
        
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
