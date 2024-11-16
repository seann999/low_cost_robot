import time
from robot import Robot
from robot_wrapper import RobotEnv
from simple_pid import PID
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def main():
    try:
        env = RobotEnv()
        env.connect()
        env.home_joints()

        # Set up the plot
        plt.ion()  # Enable interactive mode
        fig, ax = plt.subplots()
        robot_point, = ax.plot([], [], 'bo', label='Robot')  # Blue dot for robot
        goal_point, = ax.plot([], [], 'ro', label='Goal')    # Red dot for goal
        direction_arrow = ax.quiver([], [], [], [], color='b', scale=5)
        goal_arrow = ax.quiver([], [], [], [], color='r', scale=5)  # Add goal direction arrow

        # Add text annotations right here, after setting up the plot but before the goal position
        distance_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
        yaw_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)

        input("Press Enter to continue...")

        current_pose = env.tracker.get_latest_position()

        goal_x = current_pose['x']
        goal_y = current_pose['y']
        goal_yaw = current_pose['yaw']
        
        # Plot the goal position
        goal_point.set_data([goal_x], [goal_y])

        ax.set_xlim(goal_x - 1, goal_x + 1)
        ax.set_ylim(goal_y - 1, goal_y + 1)
        ax.grid(True)
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.legend()

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
            
            # Update goal direction arrow
            goal_dx = 0.2 * math.cos(goal_yaw)
            goal_dy = 0.2 * math.sin(goal_yaw)
            goal_arrow.set_offsets([[goal_x, goal_y]])
            goal_arrow.set_UVC(goal_dx, goal_dy)

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
            
            # Calculate rotation command (yaw control)
            # Calculate the shortest angle difference between current and goal yaw
            yaw_diff = goal_yaw - curr_yaw
            yaw_diff = math.atan2(math.sin(yaw_diff), math.cos(yaw_diff))  # Normalize to [-pi, pi]
            
            # Convert to rotation command (-0.3 to 0.3)
            rotation_speed = np.clip(yaw_diff * 0.3, -0.3, 0.3)
            
            # Calculate speed based on distance
            if distance < 0.01:  # Very close to goal
                speed = 0
                # When stopped, focus on final orientation
                if abs(yaw_diff) > np.deg2rad(3):
                    rotation_speed = np.clip(yaw_diff * 0.3, -0.3, 0.3)
                    if abs(rotation_speed) < 0.2:
                        rotation_speed = 0.2 * np.sign(rotation_speed)
                else:
                    rotation_speed = 0
            else:
                # speed = 90
                speed = min(90, max(40, distance * 500))

            # Add the text updates right here, after calculating distance and yaw_diff but before sending commands
            distance_text.set_text(f'Distance to goal: {distance:.3f} m')
            yaw_text.set_text(f'Yaw difference: {math.degrees(yaw_diff):.1f}°')
            
            print(speed, direction, rotation_speed)
            
            env.send_base([speed, direction, -rotation_speed])

            # Add these two lines to update the plot
            plt.draw()
            plt.pause(0.001)  # Small pause to allow the plot to update

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
