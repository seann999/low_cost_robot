import base64
import time
import eventlet
import socketio
# from zmq_subscriber import decode_data, DataPacket

import base64
import struct
import numpy as np
from scipy.spatial.transform import Rotation
import cv2
import threading
import logging

class DataPacket:
    def __init__(self, transform_matrix: np.ndarray, timestamp):
        self.transform_matrix = transform_matrix.copy()
        self.timestamp = timestamp

    def __str__(self):
        return f"Translation: {self.transform_matrix[:3, 3]}, Timestamp: {self.timestamp:.3f}"

def decode_data(encoded_str):
    # Decode the base64 string to bytes
    data_bytes = base64.b64decode(encoded_str)
    
    transform_matrix = np.zeros((4, 4))
    # Unpack transform matrix (16 floats)
    for i in range(4):
        for j in range(4):
            transform_matrix[i, j] = struct.unpack('f', data_bytes[4 * (4 * i + j):4 * (4 * i + j + 1)])[0]
    # The transform matrix is stored in column-major order in swift, so we need to transpose it in python
    transform_matrix = transform_matrix.T
    
    # Unpack timestamp (1 double)
    timestamp = struct.unpack('d', data_bytes[64:72])[0]
    
    return DataPacket(transform_matrix, timestamp)

class PhoneTracker:
    def __init__(self, port=5555, enable_visualization=False):
        self.sio = socketio.Server()
        self.app = socketio.WSGIApp(self.sio)
        self.port = port
        self.enable_visualization = enable_visualization
        
        # State variables
        self.latest_position = {'x': 0, 'y': 0, 'yaw': 0}
        self.prev_position = {'x': 0, 'y': 0, 'yaw': 0}

        self.package_cnt = 0
        self.received_first_message = False
        self.full_pose = None
        
        # Visualization variables (if enabled)
        if enable_visualization:
            self.window_name = 'Phone Tracking (Top-Down View)'
            self.image_size = (600, 800)
            self.plot_height = 150
            self.history_length = 10000
            self.position_history = []
            self.angle_history = []
            self.initialized = False
            
        # Setup socket.io events
        self.sio.on('connect')(self.on_connect)
        self.sio.on('disconnect')(self.on_disconnect)
        self.sio.on('update')(self.handle_message)
        
        # Start server in a separate thread
        self.server_thread = threading.Thread(target=self._run_server)
        self.server_thread.daemon = True
        self.server_thread.start()

    def on_connect(self, sid, environ):
        print("Client connected", sid)

    def on_disconnect(self, sid):
        print("Client disconnected", sid)

    def calculate_xyt(self, pose_matrix):
        """
        Calculate position and orientation from transform matrix.
        
        Args:
            pose_matrix (np.ndarray): 4x4 transformation matrix
            
        Returns:
            dict: Contains x, y, and yaw values
        """
        # Extract position and rotation
        position = pose_matrix[:3, 3]
        rotation_matrix = pose_matrix[:3, :3]
        
        # Calculate vectors
        up_vector = np.array([0, 0, 1])
        right_vector = rotation_matrix[:3, 0]
        forward_vector = np.cross(right_vector, up_vector)
        phone_to_center_distance_long = 0.06
        phone_to_center_distance_lat = 0
        yaw = float(np.arctan2(forward_vector[1], forward_vector[0]))
        
        return {
            'x': float(position[0] + np.cos(yaw) * phone_to_center_distance_long + np.sin(yaw) * phone_to_center_distance_lat),
            'y': float(position[1] + np.sin(yaw) * phone_to_center_distance_long - np.cos(yaw) * phone_to_center_distance_lat),
            'yaw': yaw
        }

    def handle_message(self, sid, data):
        self.received_first_message = True
        structured_data = decode_data(data)
        # Rotation matrix to convert from +y up to +z up coordinate system
        
        raw_pose = structured_data.transform_matrix
        raw_pose[:3, 3] += raw_pose[:3, 0] * 0.03
        raw_pose[:3, 3] -= raw_pose[:3, 1] * 0.02
        R_convert = np.array([
            [1, 0, 0, 0],
            [0, 0, -1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ])
        self.full_pose = R_convert @ raw_pose

        self.prev_position = self.latest_position.copy()
        
        # Update latest position using the new function
        self.latest_position = self.calculate_xyt(self.full_pose)
        self.latest_position['timestamp'] = structured_data.timestamp
        
        if self.enable_visualization:
            if not self.initialized:
                self._init_visualization()
            self._update_visualization(self.latest_position)
        
        self.package_cnt += 1

    # def get_latest_position(self):
    #     """Returns the latest x, y, and yaw values of the phone along with initialization status."""
    #     return {**self.latest_position, 'initialized': self.received_first_message}

    def get_latest_state(self):
        state = {**self.latest_position, 'initialized': self.received_first_message}
        
        # Calculate velocities if we have both positions and timestamps
        if self.received_first_message and 'timestamp' in self.latest_position and 'timestamp' in self.prev_position:
            dt = self.latest_position['timestamp'] - self.prev_position['timestamp']
            if dt > 0:  # Avoid division by zero
                state['vx'] = (self.latest_position['x'] - self.prev_position['x']) / dt
                state['vy'] = (self.latest_position['y'] - self.prev_position['y']) / dt
                # Calculate angular velocity (handle wraparound for yaw)
                dyaw = self.latest_position['yaw'] - self.prev_position['yaw']
                # Normalize angular difference to [-pi, pi]
                dyaw = (dyaw + np.pi) % (2 * np.pi) - np.pi
                state['vyaw'] = dyaw / dt
            else:
                state['vx'] = state['vy'] = state['vyaw'] = 0.0
        else:
            state['vx'] = state['vy'] = state['vyaw'] = 0.0
            
        return state

    def _run_server(self):
        """Runs the socketio server in a separate thread."""
        logging.getLogger('engineio').setLevel(logging.ERROR)
        logging.getLogger('socketio').setLevel(logging.ERROR)
        logging.getLogger('werkzeug').setLevel(logging.ERROR)
        
        eventlet.wsgi.server(eventlet.listen(('', self.port)), self.app, log_output=False)

    def _init_visualization(self):
        if not self.initialized:
            cv2.namedWindow(self.window_name)
            self.initialized = True

    def _update_visualization(self, position):
        # Create a blank image
        image = np.zeros((self.image_size[0], self.image_size[1], 3), dtype=np.uint8)
        
        # Update history
        self.position_history.append([position['x'], position['y']])
        self.angle_history.append(position['yaw'])
        if len(self.position_history) > self.history_length:
            self.position_history.pop(0)
            self.angle_history.pop(0)
        
        # Convert histories to numpy arrays
        pos_array = np.array(self.position_history)
        ang_array = np.array(self.angle_history)
        
        # Draw top-down view in the upper half
        top_view_size = 300
        center_x = self.image_size[1] // 2
        center_y = top_view_size // 2
        
        # Draw coordinate grid
        cv2.line(image, (center_x, 0), (center_x, top_view_size), (50, 50, 50), 1)
        cv2.line(image, (center_x - 150, center_y), (center_x + 150, center_y), (50, 50, 50), 1)
        
        # Draw current position and orientation
        if len(pos_array) > 0:
            scale = 100
            pos_x = int(center_x + pos_array[-1, 0] * scale)
            pos_y = int(center_y - pos_array[-1, 1] * scale)
            
            # Draw position trail
            for i in range(len(pos_array) - 1):
                pt1 = (int(center_x + pos_array[i, 0] * scale), 
                      int(center_y - pos_array[i, 1] * scale))
                pt2 = (int(center_x + pos_array[i+1, 0] * scale), 
                      int(center_y - pos_array[i+1, 1] * scale))
                cv2.line(image, pt1, pt2, (255, 255, 0), 1)
            
            # Draw phone orientation (arrow)
            angle_rad = ang_array[-1]
            arrow_length = 30
            end_x = pos_x + int(arrow_length * np.cos(angle_rad))
            end_y = pos_y - int(arrow_length * np.sin(angle_rad))
            cv2.arrowedLine(image, (pos_x, pos_y), (end_x, end_y), (0, 0, 255), 2)
        
        # Draw plots
        self._draw_plot(image, pos_array[:, 0], top_view_size + 50, 'X Position')
        self._draw_plot(image, ang_array, top_view_size + 50 + self.plot_height, 'Y Rotation')
        
        # Display the image
        cv2.imshow(self.window_name, image)
        cv2.waitKey(1)

    def _draw_plot(self, image, data, y_offset, label):
        cv2.putText(image, label, (10, y_offset + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.line(image, (0, y_offset + self.plot_height//2), 
                (self.image_size[1], y_offset + self.plot_height//2), (50, 50, 50), 1)
        
        if len(data) > 1:
            pts = np.zeros((len(data), 2), dtype=np.int32)
            pts[:, 0] = np.linspace(0, self.image_size[1], len(data))
            scale = 50 if label == 'X Position' else 0.5
            scaled_vals = data * scale
            pts[:, 1] = y_offset + self.plot_height//2 - scaled_vals
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(image, [pts], False, (0, 255, 0), 1)

if __name__ == '__main__':
    np.set_printoptions(precision=4, suppress=True)
    tracker = PhoneTracker(port=5555, enable_visualization=True)

    try:
        while True:
            position = tracker.get_latest_state()
            print(f"X: {position['x']:.3f}, Y: {position['y']:.3f}, Yaw: {position['yaw']:.3f}")
            print(f"vX: {position['vx']:.3f}, vY: {position['vy']:.3f}, vYaw: {position['vyaw']:.3f}")
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Shutting down...")
