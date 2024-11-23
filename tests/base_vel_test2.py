import time
from robot import Robot
from robot_wrapper import RobotEnv
import numpy as np
import math
import csv
from estimate_velocity import VelocityMatcher
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

class Trajectory:
    def __init__(self):
        self.times = []
        self.x_points = []
        self.y_points = []
        self.yaw_points = []
    
    def add_waypoint(self, t, x, y, yaw):
        """Add a waypoint at time t with position (x, y) and yaw angle"""
        # Insert maintaining time order
        insert_idx = np.searchsorted(self.times, t)
        self.times.insert(insert_idx, t)
        self.x_points.insert(insert_idx, x)
        self.y_points.insert(insert_idx, y)
        self.yaw_points.insert(insert_idx, yaw)
    
    def get_state(self, t):
        """
        Get interpolated state at time t using simple linear interpolation
        between the two nearest waypoints
        """
        if len(self.times) < 2:
            raise ValueError("Need at least 2 waypoints")
            
        # If beyond last waypoint, return last position with zero velocity
        if t >= self.times[-1]:
            return {
                'x': self.x_points[-1],
                'y': self.y_points[-1],
                'yaw': self.yaw_points[-1],
                'vx': 0.0,
                'vy': 0.0,
                'vyaw': 0.0
            }
            
        # Find the two waypoints we're between
        next_idx = np.searchsorted(self.times, t)
        prev_idx = next_idx - 1
        
        # Get time segment and do linear interpolation
        t0, t1 = self.times[prev_idx], self.times[next_idx]
        alpha = (t - t0) / (t1 - t0)  # Interpolation factor (0 to 1)
        
        # Interpolate positions
        x = self.x_points[prev_idx] + alpha * (self.x_points[next_idx] - self.x_points[prev_idx])
        y = self.y_points[prev_idx] + alpha * (self.y_points[next_idx] - self.y_points[prev_idx])
        yaw = self.yaw_points[prev_idx] + alpha * (self.yaw_points[next_idx] - self.yaw_points[prev_idx])
        
        # Calculate velocities (constant between waypoints)
        dt = t1 - t0
        vx = (self.x_points[next_idx] - self.x_points[prev_idx]) / dt
        vy = (self.y_points[next_idx] - self.y_points[prev_idx]) / dt
        vyaw = (self.yaw_points[next_idx] - self.yaw_points[prev_idx]) / dt
        
        return {
            'x': x,
            'y': y,
            'yaw': yaw,
            'vx': vx,
            'vy': vy,
            'vyaw': vyaw
        }

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
        traj.add_waypoint(5.0, 0.2, 0.2, -3.14*0.5)
        traj.add_waypoint(7.0, 0.2, 0.2, -3.14)
        traj.add_waypoint(11.0, 0.2, 0.2, -3.14*1.5)
        traj.add_waypoint(15.0, 0.2, 0.2, 3.14)
        traj.add_waypoint(25.0, 0, 0, 0)

        # Plot desired trajectory before execution
        num_steps = 600
        dt = 1/50
        sample_times = np.array([i * dt for i in range(num_steps)])
        goal_x = np.zeros(num_steps)
        goal_y = np.zeros(num_steps)
        
        for i in range(num_steps):
            state = traj.get_state(sample_times[i])
            goal_x[i] = state['x']
            goal_y[i] = state['y']

        # plt.figure(figsize=(8, 8))
        # plt.plot(goal_x, goal_y, 'r-', label='Desired Path', linewidth=2)
        # plt.scatter([0.0, 0.2, 0.2, 0.2], [0.0, 0.0, 0.2, 0.2], 
        #            color='blue', s=100, label='Waypoints')
        # plt.xlabel('X Position (m)')
        # plt.ylabel('Y Position (m)')
        # plt.title('Desired Robot Trajectory')
        # plt.grid(True)
        # plt.axis('equal')
        # plt.legend()
        # plt.show()

        # Add lists to store position data
        x_points = []
        y_points = []
        time_points = []
        curr_vy_points = []
        goal_vy_points = []
        start_time = time.time()

        command_speed = 20
        command_rotation = 0


        for i in range(50 * 30):
            current_time = time.time()
            # dt = current_time - prev_time
            goal = traj.get_state(i/50)
            goal_vx = goal['vx']
            goal_vy = goal['vy']
            goal_vyaw = goal['vyaw']
            # print(goal_vx, goal_vy, goal_vyaw)
            
            base_state = env.tracker.get_latest_state()

            curr_x = base_state['x']
            curr_y = base_state['y']
            
            curr_vx = base_state['vx']
            curr_vy = base_state['vy']
            curr_yaw = base_state['yaw']
            curr_vyaw = base_state['vyaw']
            
            goal_angle = math.atan2(goal_vy, goal_vx)
            relative_angle = math.degrees(goal_angle - curr_yaw)
            # Normalize angle to [-180, 180]
            relative_angle = ((relative_angle + 180) % 360) - 180
            # Convert to robot's direction system (0 is left, 90 is forward)
            command_direction = relative_angle + 90

            curr_speed = math.sqrt(curr_vx*curr_vx + curr_vy*curr_vy)
            goal_speed = math.sqrt(goal_vx*goal_vx + goal_vy*goal_vy)
            diff_speed = goal_speed - curr_speed
            command_speed += diff_speed * 10.0
            command_speed = np.clip(command_speed, 0, 90)
            # print(goal_speed, curr_speed, diff_speed, command_speed)

            diff_vyaw = goal_vyaw - curr_vyaw
            command_rotation += diff_vyaw * 0.01
            command_rotation = np.clip(command_rotation, -0.3, 0.3)

            # Store position data
            x_points.append(curr_x)
            y_points.append(curr_y)
            time_points.append(time.time() - start_time)
            curr_vy_points.append(curr_vy)
            goal_vy_points.append(goal_vy)

            env.send_base([command_speed, command_direction, -command_rotation])
            time.sleep(1/50)

        env.stop_base()
        
        # Create figure with two subplots
        fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(15, 6))
        ax2 = ax1.twinx()  # Create second y-axis sharing same x-axis

        # First subplot - velocities and command speed
        line1 = ax1.plot(time_points, curr_vy_points, 'b-', label='Current Vy')
        line2 = ax1.plot(time_points, goal_vy_points, 'g-', label='Goal Vy')
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Velocity (m/s)', color='b')
        ax1.tick_params(axis='y', labelcolor='b')

        # line3 = ax2.plot(time_points, command_speed_points, 'r-', label='Command Speed')
        # ax2.set_ylabel('Command Speed', color='r')
        # ax2.tick_params(axis='y', labelcolor='r')

        # lines = line1 + line2 + line3
        # labels = [l.get_label() for l in lines]
        # ax1.legend(lines, labels, loc='upper right')
        # ax1.set_title('Y-axis Velocity and Command Speed')
        # ax1.grid(True)

        # # Second subplot - X-Y trajectory
        # ax3.plot(x_points, y_points, 'k-', label='Robot Path')
        # ax3.set_xlabel('X Position (m)')
        # ax3.set_ylabel('Y Position (m)')
        # ax3.set_title('Robot Trajectory')
        # ax3.grid(True)
        # ax3.axis('equal')  # Make the plot aspect ratio 1:1
        # ax3.legend()

        # plt.tight_layout()
        # plt.show()
        
    except KeyboardInterrupt:
        print("\nExiting gracefully...")
    finally:
        env.close()
        print("Connection closed")

if __name__ == "__main__":
    main()
