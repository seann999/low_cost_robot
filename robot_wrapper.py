import socket
import json
import cv2
import numpy as np
import threading
from typing import Tuple, Optional
from dataclasses import dataclass
import time
import math

import ikpy
from ikpy.chain import Chain
from ikpy.link import OriginLink, URDFLink
ikpy.inverse_kinematics.ORIENTATION_COEFF = 0.01
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation as R

from phone import PhoneTracker

from umi.common.pose_util import mat_to_pose10d, rot6d_to_mat
import json


camera2gripper = json.load(open('calibration/camera_to_ee.json'))['T_cam2gripper']


def modify_urdf_for_ikpy(urdf_file):
    tree = ET.parse(urdf_file)
    root = tree.getroot()
    
    for joint in root.findall(".//joint"):
        joint_type = joint.get("type")
        if joint_type is None:
            # If 'type' is not an attribute, try to find it as a child element
            type_elem = joint.find("type")
            if type_elem is not None:
                joint_type = type_elem.text
        
        if joint_type == "continuous":
            if "type" in joint.attrib:
                joint.attrib["type"] = "revolute"
            else:
                # If 'type' was a child element, modify it
                type_elem = joint.find("type")
                if type_elem is not None:
                    type_elem.text = "revolute"
            
            # Add limit element if it doesn't exist
            if joint.find("limit") is None:
                limit = ET.SubElement(joint, "limit")
                limit.set("lower", "-3.14159")  # -pi
                limit.set("upper", "3.14159")   # pi
    
    modified_urdf = "modified_" + urdf_file
    tree.write(modified_urdf)
    return modified_urdf


modified_urdf = modify_urdf_for_ikpy("low_cost_robot.urdf")
# Load the modified URDF file
robot_arm_chain = Chain.from_urdf_file(modified_urdf)

from plotter import KinematicsPlotter
plotter = KinematicsPlotter()
plotter.initialize_plot(plot_id="follower")


def pose7d_to_matrix(pose):
    """Convert a pose (position and orientation) to a 4x4 transformation matrix."""
    position = pose[:3]
    orientation = R.from_quat(pose[3:7]).as_matrix()
    matrix = np.eye(4)
    matrix[:3, :3] = orientation
    matrix[:3, 3] = position
    return matrix


def matrix_to_pose7d(matrix):
    position = matrix[:3, 3]
    orientation = R.from_matrix(matrix[:3, :3]).as_quat()
    return np.concatenate([position, orientation])


def cam_move_to_ee(cam_mat, pos, rot6d):
    # cam_mat = ee_mat @ camera2gripper

    trans = np.eye(4)
    trans[:3, 3] = pos
    trans[:3, :3] = rot6d_to_mat(rot6d)
    cam_move_to = cam_mat @ trans
    ee_move_to = cam_move_to @ np.linalg.inv(camera2gripper)

    return cam_move_to, ee_move_to


def joints_to_posemat(joints, camera_frame=False):
    raw_state = np.concatenate([[0], joints[:-1]])
    joints_float = (np.array(raw_state) - 2048) / 2048 * np.pi
    pose_matrix = robot_arm_chain.forward_kinematics(joints_float)

    if camera_frame:
        pose_matrix = pose_matrix @ camera2gripper

    return pose_matrix


def raw_joints_to_plottable(joints):
    position_angles = (np.array(joints) - 2048) / 2048 * np.pi
    position_angles = np.concatenate([[0], position_angles[:-1]])
    return position_angles


def pose7d_to_joints(pose, gripper_joint, initial_position=None):
    pose_matrix = pose7d_to_matrix(pose)

    if initial_position is not None:
        initial_position_angles = (np.array(initial_position) - 2048) / 2048 * np.pi
        initial_position_angles = np.concatenate([[0], initial_position_angles[:-1]])
    else:
        initial_position_angles = None

    # plotter.update_plot(robot_arm_chain, initial_position_angles, plot_id="leader", target_matrix=joints_to_pose(initial_position))

    joints = robot_arm_chain.inverse_kinematics(
        target_position=pose_matrix[:3, -1],
        target_orientation=pose_matrix[:3, :3],
        orientation_mode="all",
        initial_position=initial_position_angles)

    plotter.update_plot(robot_arm_chain, joints, plot_id="follower", target_matrix=pose_matrix)

    joints = np.array(joints) / np.pi * 2048 + 2048
    joints[:-1] = joints[1:]
    joints[-1] = gripper_joint
    joints = joints.astype(np.int32).tolist()

    return joints

@dataclass
class Observation:
    image: Optional[np.ndarray]
    joints: Optional[np.ndarray]
    ee_pose: Optional[np.ndarray]
    cam_pose: Optional[np.ndarray]
    
class RobotEnv:
    def __init__(self, host: str = '192.168.0.231', port: int = 5000, track_phone: bool = True):
        self.host = host
        self.port = port
        self.client_socket = None
        self.running = False
        self.latest_observation = None
        self._lock = threading.Lock()
        self.tracker = PhoneTracker(port=5555, enable_visualization=False) if track_phone else None
        self.track_phone = track_phone

        self.phone_marker_pose = json.load(open('calibration/marker_pose.json'))['T_marker2base']
        
    def _receive_sized_message(self, sock):
        """Helper method to receive a size-prefixed message."""
        size_bytes = sock.recv(4)
        if not size_bytes:
            raise ConnectionError("Connection closed by server")
        size = int.from_bytes(size_bytes, byteorder='big')
        
        data = b''
        while len(data) < size:
            chunk = sock.recv(min(size - len(data), 4096))
            if not chunk:
                raise ConnectionError("Connection closed while receiving data")
            data += chunk
        return data
        
    def connect(self):
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((self.host, self.port))
        self.running = True
        self.update_thread = threading.Thread(target=self._update_loop)
        self.update_thread.daemon = True
        self.update_thread.start()

        while self.get_observation() is None:
            time.sleep(0.1)

        if self.track_phone:
            print("Waiting for phone to initialize...")
            while not self.tracker.received_first_message:
                time.sleep(0.1)
            print("Phone initialized")
        
    def _update_loop(self):
        while self.running:
            try:
                # Receive position data
                position_data = self._receive_sized_message(self.client_socket)
                header = json.loads(position_data.decode())
                follower_joints = header['position']
                
                # Receive image data
                image_data = self._receive_sized_message(self.client_socket)
                
                # Process image
                img_np = np.frombuffer(image_data, dtype=np.uint8)
                frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
                
                if frame is not None:
                    ee_pose = joints_to_posemat(follower_joints, camera_frame=False)
                    cam_pose = joints_to_posemat(follower_joints, camera_frame=True)
                    
                    # Update latest observation thread-safely
                    with self._lock:
                        self.latest_observation = Observation(frame, follower_joints, ee_pose, cam_pose)
                        
            except (ConnectionError, json.JSONDecodeError) as e:
                print(f"Connection error in update loop: {e}")
                self._attempt_reconnect()
            except Exception as e:
                print(f"Error in update loop: {e}")
                self._attempt_reconnect()
                
    def _attempt_reconnect(self):
        """Helper method to handle reconnection."""
        while self.running:
            try:
                print("Attempting to reconnect...")
                if self.client_socket:
                    self.client_socket.close()
                self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.client_socket.connect((self.host, self.port))
                print("Reconnected successfully")
                break
            except Exception as e:
                print(f"Reconnection failed: {e}")
                time.sleep(2)  # Wait before retrying
                
    def get_observation(self) -> Observation:
        with self._lock:
            return self.latest_observation

    def move_to_pose(self, pose_matrix, duration=3.0):
        current_joints = self.get_observation().joints
        pose7d = matrix_to_pose7d(pose_matrix)
        joints = pose7d_to_joints(pose7d, current_joints[-1], initial_position=current_joints)
        self.move_to_joints(joints, duration)

    def move_to_joints(self, joints, duration=3.0):
        """Move robot from current joints to target joints over specified time.
        
        Args:
            joints: Target joint positions
            time: Duration of movement in seconds
        """
        # Wait for valid observation
        while True:
            current_obs = self.get_observation()
            if current_obs.joints is not None:
                break
            print("Waiting for valid joint observation...")
            time.sleep(0.1)

        if duration == 0:
            self.send_joints(joints)
            return
            
        start_joints = np.array(current_obs.joints)
        target_joints = np.array(joints)
        
        # Calculate number of steps for 10Hz
        hz = 50
        steps = int(duration * hz)
        step_time = 1.0 / hz
        
        # Linear interpolation between current and target joints
        for i in range(steps + 1):
            t = i / steps  # Interpolation factor (0 to 1)
            interpolated_joints = start_joints + t * (target_joints - start_joints)
            
            # Convert to integers
            interpolated_joints = np.round(interpolated_joints).astype(np.int32)
            
            # Send interpolated joints
            self.send_joints(interpolated_joints.tolist())
            
            # Sleep to maintain 10Hz
            time.sleep(step_time)
    
    def _send_message(self, data: dict):
        """Send a length-prefixed JSON message."""
        if self.client_socket:
            try:
                json_data = json.dumps(data).encode()
                length = len(json_data)
                # Send size header first
                self.client_socket.sendall(length.to_bytes(4, byteorder='big'))
                # Then send the actual data
                self.client_socket.sendall(json_data)
            except Exception as e:
                print(f"Error sending message: {e}")
                self._attempt_reconnect()
    
    def send_joints(self, position: float):
        """Send joint positions to the robot."""
        self._send_message({"position": position})

    def send_base(self, base):
        self._send_message({"base": base})

    def send_action(self, joints, base):
        self._send_message({
            "position": joints,
            "base": base
        })

    def stop_base(self):
        self.send_base([0, 0, 0])
            
    def close(self):
        self.stop_base()
        self.running = False
        if self.client_socket:
            self.client_socket.close()

    def get_world_pose(self):
        obs = self.get_observation()
        live_phone_pose = self.tracker.full_pose.copy()

        T_phone2marker = np.eye(4)
        T_phone2marker[:3, :3] = np.array([
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, -1],
        ])
        
        live_phone_pose = live_phone_pose @ T_phone2marker
        live_phone_pose[:3, 3] += live_phone_pose[:3, 1] * 0.06
        live_phone_pose[:3, 3] -= live_phone_pose[:3, 0] * 0.02

        world_base_pose = live_phone_pose @ np.linalg.inv(self.phone_marker_pose)
        world_ee_pose = world_base_pose @ obs.ee_pose

        return world_base_pose, world_ee_pose, live_phone_pose

    def move_base_to(self, goal_x, goal_y, goal_yaw):
        current_pose = self.tracker.get_latest_position()

        curr_x = current_pose['x']
        curr_y = current_pose['y']
        curr_yaw = current_pose['yaw']

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
                self.send_base([speed, direction, -rotation_speed])
                return True
        else:
            # speed = 90
            speed = min(90, max(40, distance * 500))
        
        self.send_base([speed, direction, -rotation_speed])
        return False

    def home_joints(self):
        posemat = np.eye(4)
        posemat[:3, :3] = R.from_euler('xyz', [90, 0, 0], degrees=True).as_matrix()
        posemat[0, 3] = 0
        posemat[1, 3] = 0.15
        posemat[2, 3] = 0.1
        self.move_to_pose(posemat, duration=2)