import time
from robot import Robot
from robot_wrapper import RobotEnv
import numpy as np
import math
import matplotlib.pyplot as plt
import csv
import pickle
import argparse


def collect_data(env, speed, direction, yaw, duration=1):
    hz = 50
    data = []
    base = env.tracker.get_latest_position()
    # print(base)
    assert abs(base['x'] - 0) < 0.03 and abs(base['y'] - 0) < 0.03

    for _ in range(int(duration * hz)):
        data.append(env.tracker.get_latest_position())
        env.send_base([speed, direction, yaw])
        time.sleep(1/hz)
    
    env.stop_base()

    # end_base_pose = env.tracker.get_latest_position()
    # moved_distance = math.sqrt((end_base_pose['x'] - start_base_pose['x'])**2 + (end_base_pose['y'] - start_base_pose['y'])**2)
    # return moved_distance

    return data


def plot_from_pkl(filename='robot_movement_data.pkl'):
    # Load pickle data
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    
    data_input = data['data_input']  # List of [speed, direction]
    data_result = data['data_result']  # List of trajectory dictionaries
    
    plt.figure(figsize=(12, 8))
    
    # Create color list for different speeds
    unique_speeds = sorted(set(input[0] for input in data_input))
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'cyan', 'magenta']
    # Repeat colors if we have more speeds than colors
    speed_colors = colors * (len(unique_speeds) // len(colors) + 1)
    
    # Plot each trajectory
    for (speed, direction, yaw), trajectory in zip(data_input, data_result):
        # Extract x and y coordinates from trajectory
        x_coords = [pose['x'] for pose in trajectory]
        y_coords = [pose['y'] for pose in trajectory]
        
        # Get color based on speed
        color_idx = unique_speeds.index(speed)
        
        # Plot trajectory
        plt.plot(x_coords, y_coords, '-', 
                color=speed_colors[color_idx],
                label=f'Speed: {speed}, Direction: {direction}°')
        
        # Mark start point with a green circle
        plt.plot(x_coords[0], y_coords[0], 'go', markersize=8, label='_nolegend_')
        
        # Mark end point with a red square
        plt.plot(x_coords[-1], y_coords[-1], 'rs', markersize=8, label='_nolegend_')
    
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('Robot Trajectories')
    plt.grid(True)
    plt.axis('equal')  # Equal aspect ratio
    
    # Add legend with unique entries only
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    # plt.legend(by_label.values(), by_label.keys(), 
    #           bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()  # Adjust layout to prevent legend cutoff
    plt.savefig('robot_trajectories.png', bbox_inches='tight')
    plt.show()

def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description='Robot movement data collection and analysis')
    parser.add_argument('operation', choices=['collect', 'plot', 'collect_duration', 'plot_duration', 'test_yaw'],
                        help='Operation to perform')
    args = parser.parse_args()
    
    try:
        # Replace input() with args.operation
        operation = args.operation
        
        if operation == 'collect':
            env = RobotEnv()
            env.connect()
            env.home_joints()
            env.move_base_to_wait(0, 0, 0, pos_tol=0.01, yaw_tol=3)
            time.sleep(1)

            # load existing data if it exists
            try:
                with open('robot_movement_data.pkl', 'rb') as f:
                    data = pickle.load(f)
                    data_input = data['data_input']
                    data_result = data['data_result']
            except FileNotFoundError:
                data_input = []
                data_result = []

            # for d in np.linspace(0, 0.3, 11):
            #     print(d)
            #     collect_data(env, 0, 90, d)
            
            for yaw in [-0.1, -0.2, -0.3]:# [0, 0.1, 0.2, 0.3]:
                for direction in [0, 45, 90, 135, 180, 225, 270, 315]: # [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]:
                    for speed in [0, 20, 40, 60, 80, 90]: # list(range(20)) + list(range(20, 91, 10)):
                        print(speed, direction, yaw)
                        data_input.append([speed, direction, yaw])
                        result = collect_data(env, speed, direction, yaw)
                        data_result.append(result)

                        env.move_base_to_wait(0, 0, 0)
                        base = env.get_base_pose()
                        # print(base)
                        assert abs(base['x'] - 0) < 0.03 and abs(base['y'] - 0) < 0.03
                        time.sleep(1)
                    
            # Save data to pickle
            with open('robot_movement_data.pkl', 'wb') as f:
                pickle.dump(dict(data_input=data_input, data_result=data_result), f)
            print("Data saved to robot_movement_data.pkl")
            
        elif operation == 'plot':
            plot_from_pkl()
        elif operation == 'collect_duration':
            env = RobotEnv()
            env.connect()

            data_duration = []
            data_distance = []

            # for duration in np.linspace(0, 1, 11).tolist() + list(range(2, 6)):
            for duration in np.linspace(0, 1, 51).tolist():
                input(f'start? {duration}')
                distance = collect_data(env, 90, 90, 0, duration)
                data_duration.append(duration)
                data_distance.append(distance)
                input('reset?')
                distance = collect_data(env, -90, 90, 0, duration)
                data_duration.append(duration)
                data_distance.append(distance)

            save_data_to_csv(data_duration, data_distance, 'robot_movement_data_duration.csv')
            print("Data saved to robot_movement_data_duration.csv")
        elif operation == 'plot_duration':
            plot_from_csv('robot_movement_data_duration.csv')
        elif operation == 'test_yaw':
            env = RobotEnv()
            env.connect()

            for yaw in np.linspace(0.1, 0.3, 31):
                input(yaw)
                env.send_base([0, 0, yaw])
                time.sleep(1)
                env.send_base([0, 0, 0])
        else:
            print("Invalid operation selected")
            
    except KeyboardInterrupt:
        print("\nExiting gracefully...")
    except Exception as e:
        print(f"Error during robot movement: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        print(f"Traceback:\n{traceback.format_exc()}")
    finally:
        if operation == 'collect':
            env.close()
            print("Connection closed")

if __name__ == "__main__":
    main()
