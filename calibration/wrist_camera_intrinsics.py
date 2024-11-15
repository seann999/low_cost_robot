import sys, json, argparse
sys.path.append('.')

import cv2
import numpy as np
from robot_wrapper import RobotEnv
from scipy.spatial.transform import Rotation as R
from pathlib import Path

def find_checkerboard(image, board_size):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
    ret, corners = cv2.findChessboardCorners(gray, board_size, flags)
    if ret:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return ret, corners

def collect_images(args):
    env = RobotEnv(track_phone=False)
    env.connect()

    # Setup checkerboard parameters
    board_size = (10, 7)  # Modify based on your checkerboard
    square_size = 0.022
    
    # Lists to store object and image points
    objpoints = []
    imgpoints = []
    images = []

    try:
        while True:
            image = env.get_observation().image
            display = image.copy()
            
            ret, corners = find_checkerboard(image, board_size)
            if ret:
                cv2.drawChessboardCorners(display, board_size, corners, ret)
            
            cv2.imshow('Camera Feed', display)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('c') and ret:  # Capture
                objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
                objp[:,:2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1,2) * square_size
                objpoints.append(objp)
                imgpoints.append(corners)
                images.append(image)
                print(f"Captured image {len(images)}")
            elif key == ord('q'):  # Quit
                break

        # Save collected data
        data = {
            'objpoints': [x.tolist() for x in objpoints],
            'imgpoints': [x.tolist() for x in imgpoints],
            'images': [cv2.imencode('.jpg', img)[1].tolist() for img in images]
        }
        
        with open(args.data_file, 'w') as f:
            json.dump(data, f)
        print(f"Saved {len(images)} images to {args.data_file}")

    finally:
        cv2.destroyAllWindows()
        env.close()

def estimate_parameters(args):
    # Load collected data
    with open(args.data_file, 'r') as f:
        data = json.load(f)
    
    objpoints = [np.array(x) for x in data['objpoints']]
    imgpoints = [np.array(x) for x in data['imgpoints']]
    images = [cv2.imdecode(np.array(x, dtype=np.uint8), cv2.IMREAD_COLOR) 
             for x in data['images']]
    
    # Convert object points to the correct format - ensure 3 channels
    objpoints = [x.reshape(-1, 1, 3).astype(np.float32) for x in objpoints]
    imgpoints = [x.reshape(-1, 1, 2).astype(np.float32) for x in imgpoints]
    
    # Initialize camera matrix and distortion coefficients
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))  # Fisheye uses 4 distortion coefficients
    
    # Calibrate fisheye camera
    flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    
    ret, mtx, dist, rvecs, tvecs = cv2.fisheye.calibrate(
        objpoints,
        imgpoints,
        images[0].shape[:2][::-1],
        K,
        D,
        flags=flags,
        criteria=criteria
    )
    
    # Save parameters
    params = {
        'camera_matrix': mtx.tolist(),
        'dist_coeffs': dist.flatten().tolist(),
        'calibration_error': ret  # Include RMS error
    }
    
    with open(args.output_file, 'w') as f:
        json.dump(params, f, indent=2)
    print(f"Fisheye camera parameters saved to {args.output_file}")
    print(f"Calibration RMS error: {ret}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['collect', 'estimate'], required=True)
    parser.add_argument('--data-file', default='calibration/calibration_data.json')
    parser.add_argument('--output-file', default='calibration/camera_params.json')
    args = parser.parse_args()

    if args.mode == 'collect':
        collect_images(args)
    else:
        estimate_parameters(args)

if __name__ == "__main__":
    main()