import sys
sys.path.append('.')

import cv2
from robot import Robot
from robot_wrapper import RobotEnv, plotter, robot_arm_chain, raw_joints_to_plottable
import numpy as np
import copy
from typing import Dict
import json
import matplotlib.pyplot as plt

plt.ion()  # Enable interactive mode

def main():
    try:
        env = RobotEnv(track_phone=False)
        env.connect()

        camera_matrix = np.array(json.load(open('calibration/camera_params.json'))['camera_matrix'])
        dist_coeffs = np.array(json.load(open('calibration/camera_params.json'))['dist_coeffs'])
        
        # Set up ArUco dictionary and parameters
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        aruco_params = cv2.aruco.DetectorParameters()
        aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
        
        # Define marker size in meters (you need to set this)
        marker_size = 0.0275  # example: 5cm marker
        
        # Create a figure once at the start
        fig = plt.figure(figsize=(10, 10))
        plt.show()
        
        while True:
            obs = env.get_observation()
            
            if obs.image is not None:
                # Detect ArUco markers
                corners, ids, rejected = detector.detectMarkers(obs.image)
                
                if ids is not None and 0 in ids:
                    # Draw detected markers
                    cv2.aruco.drawDetectedMarkers(obs.image, corners, ids)
                    
                    # Get index of marker with ID 0
                    marker_idx = list(ids.flatten()).index(0)
                    marker_corners = corners[marker_idx]
                    
                    # Undistort the corner points using fisheye model
                    undistorted_corners = cv2.fisheye.undistortPoints(
                        marker_corners, 
                        camera_matrix, 
                        dist_coeffs, 
                        P=camera_matrix
                    )
                    
                    # Estimate pose using undistorted points
                    # Convert marker corners to the format expected by solvePnP
                    object_points = np.array([
                        [-marker_size/2, marker_size/2, 0],
                        [marker_size/2, marker_size/2, 0],
                        [marker_size/2, -marker_size/2, 0],
                        [-marker_size/2, -marker_size/2, 0]
                    ], dtype=np.float32)
                    
                    success, rvec, tvec = cv2.solvePnP(
                        object_points,
                        undistorted_corners,
                        camera_matrix,
                        np.zeros((1,5)),  # Using zero distortion since we already undistorted
                        flags=cv2.SOLVEPNP_IPPE_SQUARE
                    )
                    
                    # Draw axis for each marker
                    cv2.drawFrameAxes(obs.image, camera_matrix, dist_coeffs, 
                                    rvec, tvec, marker_size/2)
                    
                    # Print pose information
                    print(f"Translation vector: {tvec.flatten()}")
                    print(f"Rotation vector: {rvec.flatten()}")
                    
                    # Convert rvec to rotation matrix
                    R_marker2cam, _ = cv2.Rodrigues(rvec)
                    # Create homogeneous transformation matrix from marker to camera
                    T_marker2cam = np.eye(4)
                    T_marker2cam[:3, :3] = R_marker2cam
                    T_marker2cam[:3, 3] = tvec.flatten()
                    
                    # Load camera to end-effector transformation
                    with open('calibration/camera_to_ee.json', 'r') as f:
                        T_cam2gripper = np.array(json.load(f)['T_cam2gripper'])
                    
                    # Get end-effector pose in base frame
                    T_gripper2base = obs.ee_pose
                    
                    # Calculate transformations:
                    # marker -> camera -> gripper -> base
                    T_marker2base = T_gripper2base @ T_cam2gripper @ T_marker2cam
                    
                    print("Marker pose in base frame:")
                    print(T_marker2base)

                    # Update plot with interactive capabilities
                    plotter.update_plot(robot_arm_chain, raw_joints_to_plottable(obs.joints), plot_id="follower",
                        target_matrix=obs.ee_pose, other_frames=[T_marker2base, np.eye(4)])
                    plt.draw()
                    plt.pause(0.001)  # Small pause to allow for interaction
                    

                cv2.imshow('Follower Camera', obs.image)
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    print("Plot interaction mode - press Enter to continue detection")
                    plt.show(block=True)  # This will block until the plot window is closed
                elif key == ord('y'):
                    # Save T_marker2base to JSON
                    save_data = {
                        'T_marker2base': T_marker2base.tolist()
                    }
                    with open('calibration/marker_pose.json', 'w') as f:
                        json.dump(save_data, f, indent=4)
                    print("Saved marker pose to calibration/marker_pose.json")
                    break
            
            print("Follower position:", obs.joints)
            
    except KeyboardInterrupt:
        print("\nExiting gracefully...")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        cv2.destroyAllWindows()
        env.close()
        print("Connection closed")

if __name__ == "__main__":
    main()