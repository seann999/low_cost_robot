import time
from robot import Robot
from robot_wrapper import RobotEnv
import numpy as np
import math
import csv
from estimate_velocity import VelocityMatcher


def main():
    try:
        env = RobotEnv()
        env.connect()
        env.home_joints()
        env.move_base_to_wait(0, 0, 0, pos_tol=0.01, yaw_tol=3)
        time.sleep(1)

        matcher = VelocityMatcher('robot_movement_data.pkl', k=1)

        def set_vel(vx, vy, vyaw):
            vel, _ = matcher.find_nearest_input(vx, vy, vyaw)
            # env.send_base(vel.tolist())
            print(vel)

        set_vel(0, 0.2, 0)
        # time.sleep(1)
        # set_vel(0.3, 0, 0)
        # time.sleep(1)
        # set_vel(0, -0.3, 0)
        # time.sleep(1)
        # set_vel(-0.3, 0, 0)
        # time.sleep(1)
        # set_vel(0, 0, 0)
        # time.sleep(1)
        # for i in range(10):
        #     rotate_speed = np.pi * 0.05 * i
        #     print(rotate_speed)
        #     set_vel(0, 0, rotate_speed)
        #     time.sleep(1)
            # set_vel(0, 0, -np.pi * 0.5)
            # time.sleep(1)
        # set_vel(0, 0, 0)
        env.stop_base()

    except KeyboardInterrupt:
        print("\nExiting gracefully...")
    finally:
        # env.close()
        print("Connection closed")

if __name__ == "__main__":
    main()
