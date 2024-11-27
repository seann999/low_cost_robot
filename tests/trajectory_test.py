import time
from robot import Robot
from robot_wrapper import RobotEnv
import numpy as np
import math
import csv
from estimate_velocity import VelocityMatcher
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from matplotlib.animation import FuncAnimation, FFMpegWriter


def main():
    try:
        env = RobotEnv()
        env.connect()
        env.home_joints()
        env.move_base_to_wait(0, 0, 0, pos_tol=0.01, yaw_tol=3)
        time.sleep(1)

        # Add waypoints (time, x, y, yaw)
        # env.base_trajectory.add_waypoint(0.0, 0.0, 0.0, 0.0)
        # env.base_trajectory.add_waypoint(2.0, 0.2, 0.0, 0.0)
        # env.base_trajectory.add_waypoint(4.0, 0.2, 0.2, -3.14*0.5)

        # env.arm_trajectory.add_waypoint(0.0, [2088, 2071, 1773, 3058, 2078, 2890])
        # env.arm_trajectory.add_waypoint(5.0, [2088, 2071, 1973, 3258, 2278, 2890])
        obs = env.get_observation()
        env.add_ee_waypoint(0.0, obs.ee_pose, 0.0)
        x2 = obs.ee_pose.copy()
        # x2[2, 3] -= 0.05
        # x2[0, 3] += 0.3
        env.add_ee_waypoint(30.0, x2, 0.0)
        # x3 = x2.copy()
        # x3[2, 3] -= 0.05
        # x3[1, 3] += 0.2
        # env.add_ee_waypoint(6.0, x3, 0)

        start_time = time.time()

        while time.time() - start_time < 60: #30.0:
            current_time = time.time()
            # print(current_time)
            env.move_base_trajectory(time.time() - start_time)
            env.move_arm_trajectory(time.time() - start_time)

            sleep_time = 1/50 - (time.time() - current_time)
            if sleep_time > 0:
                time.sleep(sleep_time)

        env.stop_base()

        waypoints = list(zip(env.base_trajectory.x_points, env.base_trajectory.y_points))  # Convert trajectory waypoints
        # env.trajectory_tracker.create_animation(waypoints=waypoints)
        env.create_animation(waypoints=waypoints)

        print('done')
    except KeyboardInterrupt:
        print("\nExiting gracefully...")
    finally:
        env.close()
        print("Connection closed")

if __name__ == "__main__":
    main()
