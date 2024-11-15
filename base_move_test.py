import time
from robot import Robot
from robot_wrapper import RobotEnv, joints_to_pose, pose_to_joints
from simple_pid import PID
import numpy as np
import math


def main():
    try:
        env = RobotEnv()
        env.connect()

        current = env.tracker.get_latest_position()
        goal_dir = 0
        goal_x = 0
        goal_y = 0

        pid_yaw = PID(1.0, 0.3, 0.2, setpoint=goal_dir, output_limits=(-0.3, 0.3))
        pid_distance = PID(10.0, 5.0, 5.0, setpoint=0, output_limits=(-90, 0))

        while True:
            current_pose = env.tracker.get_latest_position()

            pid_yaw.setpoint = 0 # np.sin(time.time())
            control_yaw = pid_yaw(current_pose['yaw'])
            print(current_pose['yaw'], '->', pid_yaw.setpoint)
            # env.send_base([0, 90, -control])
            # print(control)

            curr_x = current_pose['x']
            curr_y = current_pose['y']
            diff_x = goal_x - curr_x
            diff_y = goal_y - curr_y
            distance = math.sqrt(diff_x**2 + diff_y**2)
            pid_distance.setpoint = 0
            control_speed = -pid_distance(distance)

            direction = math.degrees(math.atan2(diff_y, diff_x)) % 360 + 90
            # print(curr_x, curr_y, goal_x, goal_y)
            print(distance, control_speed)
            # print(control_speed, direction, -control_yaw)
            env.send_base([control_speed, direction, -control_yaw])

            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nExiting gracefully...")
    except Exception as e:
        print(f"Error during robot movement: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        # Optionally add traceback
        import traceback
        print(f"Traceback:\n{traceback.format_exc()}")
    finally:
        env.close()
        print("Connection closed")

if __name__ == "__main__":
    main()
