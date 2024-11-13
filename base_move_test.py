import time
from robot import Robot
from robot_wrapper import RobotEnv, joints_to_pose, pose_to_joints
from simple_pid import PID
import numpy as np


def main():
    try:
        env = RobotEnv()
        env.connect()

        current = env.tracker.get_latest_position()
        goal_dir = 0 # current['yaw'] - np.pi / 2
        goal_x = 0 # current['x'] + np.cos(goal_dir) * 0.5
        goal_y = 0 # current['y'] + np.sin(goal_dir) * 0.5

        pid = PID(1.0, 0.3, 0.2, setpoint=goal_dir, output_limits=(-0.3, 0.3))

        while True:
            pid.setpoint = np.sin(time.time())
            current_pose = env.tracker.get_latest_position()
            control = pid(current_pose['yaw'])
            print(current_pose['yaw'], '->', pid.setpoint)
            env.send_base([0, 90, -control])
            print(control)

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
