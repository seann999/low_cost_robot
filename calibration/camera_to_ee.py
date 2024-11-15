import sys, json, argparse
sys.path.append('.')

import cv2
import numpy as np
from robot_wrapper import RobotEnv
from scipy.spatial.transform import Rotation as R
from pathlib import Path
import time

def find_checkerboard(image, board_size):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
    ret, corners = cv2.findChessboardCorners(gray, board_size, flags)
    if ret:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return ret, corners

def load_camera_params(file_path):
    with open(file_path, 'r') as f:
        params = json.load(f)
    camera_matrix = np.array(params['camera_matrix'], dtype=np.float64)
    dist_coeffs = np.array(params['dist_coeffs'], dtype=np.float64)
    return camera_matrix, dist_coeffs

def collect_calibration_data(args):
    env = RobotEnv(track_phone=False)
    env.connect()

    # Setup checkerboard parameters
    board_size = (10, 7)
    square_size = 0.022  # meters
    
    # Load camera parameters
    camera_matrix, dist_coeffs = load_camera_params(args.camera_params)
    
    # Lists to store data
    objpoints = []
    imgpoints = []
    robot_poses = []

    # Create 3D points
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1,2) * square_size

    try:
        # Define poses to visit (similar to your original script but more structured)
        poses = []
        for height in [0.03, 0.1, 0.15]:
            for pitch in [120, 90, 60]:
                for roll in [-45, 0, 45]:
                    posemat = np.eye(4)
                    posemat[:3, :3] = R.from_euler('xyz', [pitch, roll, 0], degrees=True).as_matrix()
                    posemat[0, 3] = 0
                    posemat[1, 3] = 0.15
                    posemat[2, 3] = height
                    poses.append(posemat)

        for pose in poses:
            env.move_to_pose(pose, duration=2)
            time.sleep(2)
            
            obs = env.get_observation()
            image = obs.image
            ee_pose = obs.ee_pose
            
            display = image.copy()
            ret, corners = find_checkerboard(image, board_size)
            
            if ret:
                cv2.drawChessboardCorners(display, board_size, corners, ret)
            
            cv2.imshow('Camera Feed', display)
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('y') and ret:  # Accept this pose
                objpoints.append(objp.tolist())
                imgpoints.append(corners.tolist())
                robot_poses.append(ee_pose.tolist())
                print(f"Captured pose {len(robot_poses)}")
            elif key == ord('q'):  # Quit
                break

        # Save collected data
        data = {
            'objpoints': objpoints,
            'imgpoints': imgpoints,
            'robot_poses': robot_poses
        }
        
        with open(args.data_file, 'w') as f:
            json.dump(data, f)
        print(f"Saved {len(robot_poses)} poses to {args.data_file}")

    finally:
        cv2.destroyAllWindows()
        env.close()

def estimate_calibration(args):
    # Load collected data
    with open(args.data_file, 'r') as f:
        data = json.load(f)
    
    camera_matrix, dist_coeffs = load_camera_params(args.camera_params)
    
    objpoints = [np.array(x) for x in data['objpoints']]
    imgpoints = [np.array(x) for x in data['imgpoints']]
    robot_transforms = [np.array(x) for x in data['robot_poses']]

    # Calculate camera poses
    rvecs = []
    tvecs = []
    for obj, img in zip(objpoints, imgpoints):
        _, rvec, tvec = cv2.fisheye.solvePnP(
            obj.reshape(-1, 1, 3).astype(np.float64),
            img.reshape(-1, 1, 2).astype(np.float64),
            camera_matrix,
            dist_coeffs
        )
        rvecs.append(rvec)
        tvecs.append(tvec)

    # Perform hand-eye calibration
    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        [T[:3, :3] for T in robot_transforms],
        [T[:3, 3] for T in robot_transforms],
        rvecs,
        tvecs,
        method=cv2.CALIB_HAND_EYE_TSAI
    )

    # Create 4x4 homogeneous transformation matrix
    T_cam2gripper = np.eye(4)
    T_cam2gripper[:3, :3] = R_cam2gripper
    T_cam2gripper[:3, 3] = t_cam2gripper.flatten()
    
    # Save results
    result = {
        'T_cam2gripper': T_cam2gripper.tolist()
    }
    
    with open(args.output_file, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Camera-to-gripper transformation saved to {args.output_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['collect', 'estimate'], required=True)
    parser.add_argument('--data-file', default='calibration/hand_eye_data.json')
    parser.add_argument('--output-file', default='calibration/camera_to_ee.json')
    parser.add_argument('--camera-params', default='calibration/camera_params.json')
    args = parser.parse_args()

    if args.mode == 'collect':
        collect_calibration_data(args)
    else:
        estimate_calibration(args)

if __name__ == "__main__":
    main()