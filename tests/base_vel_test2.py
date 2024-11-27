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

        # Create trajectory
        traj = Trajectory()

        # Add waypoints (time, x, y, yaw)
        traj.add_waypoint(0.0, 0.0, 0.0, 0.0)
        traj.add_waypoint(2.0, 0.2, 0.0, 0.0)
        traj.add_waypoint(4.0, 0.2, 0.2, 0.0)
        traj.add_waypoint(5.0, 0, 0.2, -3.14*0.5)
        traj.add_waypoint(7.0, 0.2, 0.2, -3.14)
        traj.add_waypoint(11.0, 0.2, 0.2, -3.14*1.5)
        traj.add_waypoint(15.0, 0.2, 0.2, -3.14*2.0)
        traj.add_waypoint(25.0, 0, 0, -3.14*2.0)

        # Create lists to store ALL trajectory data
        x_points = []
        y_points = []
        time_points = []
        curr_vy_points = []
        goal_vy_points = []
        yaw_points = []
        # Add lists for desired trajectory
        goal_x = []
        goal_y = []
        goal_yaw = []
        
        start_time = time.time()

        command_speed = 20
        command_rotation = 0

        while time.time() - start_time < 26.0:
            current_time = time.time()
            
            base_state = env.tracker.get_latest_state()

            curr_x = base_state['x']
            curr_y = base_state['y']
            
            curr_vx = base_state['vx']
            curr_vy = base_state['vy']
            curr_yaw = base_state['yaw']
            curr_vyaw = base_state['vyaw']

            goal = traj.get_state(time.time() - start_time)

            K_pos_gain = 1.0
            K_rot_gain = 1.0
            diff_x = goal['x'] - curr_x
            diff_y = goal['y'] - curr_y
            diff_yaw = goal['yaw'] - curr_yaw
            # print(diff_yaw, goal['yaw'], curr_yaw)
            diff_yaw = math.atan2(math.sin(diff_yaw), math.cos(diff_yaw))

            goal_vx = goal['vx'] + diff_x * K_pos_gain
            goal_vy = goal['vy'] + diff_y * K_pos_gain
            goal_vyaw = goal['vyaw'] + diff_yaw * K_rot_gain
            # print('v', goal_vyaw, goal['vyaw'], diff_yaw)
            
            goal_angle = math.atan2(goal_vy, goal_vx)
            relative_angle = math.degrees(goal_angle - curr_yaw)
            # Normalize angle to [-180, 180]
            relative_angle = ((relative_angle + 180) % 360) - 180
            # Convert to robot's direction system (0 is left, 90 is forward)
            command_direction = relative_angle + 90

            curr_speed = math.sqrt(curr_vx*curr_vx + curr_vy*curr_vy)
            goal_speed = math.sqrt(goal_vx*goal_vx + goal_vy*goal_vy)
            diff_speed = goal_speed - curr_speed
            command_speed += diff_speed * 1.0
            command_speed = np.clip(command_speed, 0, 90)

            # Normalize yaw difference to [-pi, pi] for shortest rotation
            diff_yaw = goal_vyaw - curr_vyaw
            command_rotation += diff_yaw * 0.01
            command_rotation = np.clip(command_rotation, -0.3, 0.3)

            # Store ALL position and goal data
            x_points.append(curr_x)
            y_points.append(curr_y)
            time_points.append(time.time() - start_time)
            curr_vy_points.append(curr_vy)
            goal_vy_points.append(goal_vy)
            yaw_points.append(curr_yaw)
            # Store goal trajectory
            goal_x.append(goal['x'])
            goal_y.append(goal['y'])
            goal_yaw.append(goal['yaw'])

            env.send_base([command_speed, command_direction, -command_rotation])

            sleep_time = 1/50 - (time.time() - current_time)
            if sleep_time > 0:
                time.sleep(sleep_time)

        env.stop_base()

        print('done')

        # Extract waypoints from trajectory for plotting
        waypoint_x = traj.x_points
        waypoint_y = traj.y_points

        # Replace the static plotting code with animation
        fig, ax = plt.subplots(figsize=(10, 10))
        
        def animate(frame):
            if frame % 10 == 0:  # Print progress every 10 frames
                print(f"Rendering frame {frame}/{num_frames} ({(frame/num_frames)*100:.1f}%)")
            
            ax.clear()
            
            # Plot full desired trajectory as reference (faded)
            ax.plot(goal_x, goal_y, 'r-', label='Desired Path', linewidth=1, alpha=0.1)
            
            # Calculate end_idx based on total points and desired duration
            end_idx = int((frame / num_frames) * len(x_points))
            if end_idx > 0:
                ax.plot(x_points[:end_idx], y_points[:end_idx], 'b-', 
                       label='Actual Path', linewidth=1, alpha=0.3)
                
                # Add direction arrows along both trajectories
                arrow_spacing = 10  # Show an arrow every N points
                for i in range(0, end_idx, arrow_spacing):
                    # Actual trajectory arrows
                    arrow_length = 0.005
                    dx = arrow_length * math.cos(yaw_points[i])
                    dy = arrow_length * math.sin(yaw_points[i])
                    ax.arrow(x_points[i], y_points[i], dx, dy,
                            head_width=0.002, head_length=0.002, fc='b', ec='b', alpha=0.5)
                    
                    # Desired trajectory arrows
                    dx = arrow_length * math.cos(goal_yaw[i])
                    dy = arrow_length * math.sin(goal_yaw[i])
                    ax.arrow(goal_x[i], goal_y[i], dx, dy,
                            head_width=0.002, head_length=0.002, fc='r', ec='r', alpha=0.5)
                
                # Plot current position arrows (larger)
                if end_idx < len(x_points):
                    arrow_length = 0.02
                    # Current position arrow (blue)
                    dx = arrow_length * math.cos(yaw_points[end_idx-1])
                    dy = arrow_length * math.sin(yaw_points[end_idx-1])
                    ax.arrow(x_points[end_idx-1], y_points[end_idx-1], dx, dy,
                            head_width=0.01, head_length=0.01, fc='b', ec='b')
                    
                    # Desired position arrow (red)
                    dx = arrow_length * math.cos(goal_yaw[end_idx-1])
                    dy = arrow_length * math.sin(goal_yaw[end_idx-1])
                    ax.arrow(goal_x[end_idx-1], goal_y[end_idx-1], dx, dy,
                            head_width=0.01, head_length=0.01, fc='r', ec='r')
            
            # Plot waypoints
            ax.scatter(waypoint_x, waypoint_y, color='green', s=100, label='Waypoints')
            
            ax.set_xlabel('X Position (m)')
            ax.set_ylabel('Y Position (m)')
            ax.set_title(f'Robot Trajectory (t={frame/50:.2f}s)')
            ax.grid(True)
            ax.axis('equal')
            ax.legend()
            
            # Set consistent axis limits
            ax.set_xlim(min(goal_x)-0.1, max(goal_x)+0.1)
            ax.set_ylim(min(goal_y)-0.1, max(goal_y)+0.1)
        
        # Create animation
        # Set num_frames based on desired animation duration and frame rate
        desired_fps = 50  # frames per second for smooth animation
        animation_duration = 26.0  # match your actual robot movement duration
        num_frames = int(desired_fps * animation_duration)
        print(f"\nStarting animation creation...")
        print(f"Total frames to render: {num_frames}")
        print(f"Expected duration: {animation_duration} seconds at {desired_fps} FPS")

        anim = FuncAnimation(fig, animate, frames=num_frames, 
                            interval=1000/desired_fps, repeat=False)

        # Save as MP4
        print(f"\nSaving animation to MP4...")
        writer = FFMpegWriter(fps=50, bitrate=2000)
        anim.save('trajectory.mp4', writer=writer)
        print("Animation saved successfully!")
        
        plt.show()
        
    except KeyboardInterrupt:
        print("\nExiting gracefully...")
    finally:
        env.close()
        print("Connection closed")

if __name__ == "__main__":
    main()
