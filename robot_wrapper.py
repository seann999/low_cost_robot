import socket
import json
import cv2
import numpy as np
import threading
from typing import Tuple, Optional
from dataclasses import dataclass
import time

import ikpy
from ikpy.chain import Chain
from ikpy.link import OriginLink, URDFLink
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation as R


camera2gripper = np.array([
    [
        0.9997668505539576,
        -0.019970828271681517,
        -0.008210392899446018, 0
    ],
    [
        -0.021388860005435672,
        -0.8638202079680002,
        -0.503345969462147, 0
    ],
    [
        0.0029599326154731358,
        0.5034042255725152,
        -0.864045962015128, 0
    ],
    [
        0.00018299471587136708,
        0.06175676142391705,
        -0.00888269351076889, 1
    ]
]).T


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


def pose_to_matrix(pose):
    """Convert a pose (position and orientation) to a 4x4 transformation matrix."""
    position = pose[:3]
    orientation = R.from_quat(pose[3:]).as_matrix()
    matrix = np.eye(4)
    matrix[:3, :3] = orientation
    matrix[:3, 3] = position
    return matrix


def joints_to_pose(joints):
    raw_state = np.concatenate([[0], joints[:-1]])
    joints_float = (np.array(raw_state) - 2048) / 2048 * np.pi
    pose_matrix = robot_arm_chain.forward_kinematics(joints_float)
    rot_quat = R.from_matrix(pose_matrix[:3, :3]).as_quat()

    pose = np.concatenate([pose_matrix[:3, -1], rot_quat, joints[-1:]])
    return pose


def pose_to_joints(pose, initial_position=None):
    if initial_position is not None:
        initial_position = (np.array(initial_position) - 2048) / 2048 * np.pi
        initial_position = np.concatenate([[0], initial_position[:-1]])

    pose_matrix = pose_to_matrix(pose[:7])
    joints = robot_arm_chain.inverse_kinematics(
        target_position=pose_matrix[:3, -1],
        target_orientation=pose_matrix[:3, :3],
        orientation_mode="all",
        initial_position=initial_position)

    joints = np.array(joints) / np.pi * 2048 + 2048
    joints[:-1] = joints[1:]
    joints[-1] = pose[-1]
    joints = joints.astype(np.int32).tolist()

    return joints

@dataclass
class Observation:
    image: Optional[np.ndarray]
    ee_pose: Optional[np.ndarray]
    ee_joints: Optional[np.ndarray]
    
class RobotEnv:
    def __init__(self, host: str = '192.168.0.231', port: int = 5000):
        self.host = host
        self.port = port
        self.client_socket = None
        self.running = False
        self.latest_observation = Observation(None, None, None)
        self._lock = threading.Lock()
        
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
                    follower_pose = joints_to_pose(follower_joints)
                    
                    # Update latest observation thread-safely
                    with self._lock:
                        self.latest_observation = Observation(frame, follower_pose, follower_joints)
                        
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
            
    def close(self):
        self.running = False
        if self.client_socket:
            self.client_socket.close()