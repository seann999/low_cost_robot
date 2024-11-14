import time
from robot import Robot
from robot_wrapper import RobotEnv, joints_to_pose, pose_to_joints
from simple_pid import PID
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def main():
    try:
        env = RobotEnv()
        env.connect()

        # Set up the plot
        plt.ion()  # Enable interactive mode
        fig, ax = plt.subplots()
        robot_point, = ax.plot([], [], 'bo', label='Robot')  # Blue dot for robot
        goal_point, = ax.plot([], [], 'ro', label='Goal')    # Red dot for goal
        direction_arrow = ax.quiver([], [], [], [], color='b', scale=5)
        
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.grid(True)
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.legend()

        goal_x = 0
        goal_y = 0
        
        # Plot the goal position
        goal_point.set_data([goal_x], [goal_y])

        while True:
            current_pose = env.tracker.get_latest_position()

            curr_x = current_pose['x']
            curr_y = current_pose['y']
            curr_yaw = current_pose['yaw']
            
            # Update robot position on plot
            robot_point.set_data([curr_x], [curr_y])
            
            # Update direction arrow
            # yaw is already in radians, so no conversion needed
            dx = 0.2 * math.cos(curr_yaw)  # 0.2 is the arrow length
            dy = 0.2 * math.sin(curr_yaw)
            direction_arrow.set_offsets([[curr_x, curr_y]])
            direction_arrow.set_UVC(dx, dy)
            
            fig.canvas.draw()
            fig.canvas.flush_events()

            # Calculate distance to goal
            dx = goal_x - curr_x
            dy = goal_y - curr_y
            distance = math.sqrt(dx*dx + dy*dy)
            
            # Calculate angle to goal in world space (in radians)
            goal_angle = math.atan2(dy, dx)
            
            # Convert to robot-relative angle
            # Subtract current yaw (already in radians) and convert to degrees
            relative_angle = math.degrees(goal_angle - curr_yaw)
            
            # Normalize angle to [-180, 180]
            relative_angle = ((relative_angle + 180) % 360) - 180
            
            # Convert to robot's direction system (0 is left, 90 is forward)
            direction = relative_angle + 90
            
            # Calculate speed based on distance
            # Using a simple mapping: closer = slower
            if distance < 0.05:  # Very close to goal
                speed = 0
            else:
                # speed = min(90, max(20, distance * 50))  # Scale with distance, but keep within [20, 90]
                speed = 90
            
            env.send_base([speed, direction, 0])

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
        plt.close()  # Close the plot window
        env.close()
        print("Connection closed")

if __name__ == "__main__":
    main()
