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


# Create a Socket.IO server
sio = socketio.Server()

# Create a WSGI app
app = socketio.WSGIApp(sio)

# Event handler for new connections
@sio.event
def connect(sid, environ):
    print("Client connected", sid)

# Event handler for disconnections
@sio.event
def disconnect(sid):
    print("Client disconnected", sid)

prev_time = 0
package_cnt = 0
# Event handler for messages on 'update' channel
@sio.on('update')
def handle_message(sid, data):
    # Assuming data is base64-encoded from the client
    global prev_time, package_cnt
    structured_data = decode_data(data)
    
    # Extract rotation matrix and convert to euler angles
    rotation_matrix = structured_data.transform_matrix[:3, :3]
    euler_angles = Rotation.from_matrix(rotation_matrix).as_euler('xyz', degrees=True)
    
    print(f"{structured_data}, fps: {1/(structured_data.timestamp - prev_time):.2f}")
    print(f"Euler angles (xyz): {euler_angles}")
    
    # Update visualization
    if not initialized:
        init_visualization()
    update_visualization(structured_data.transform_matrix)
    
    prev_time = structured_data.timestamp
    package_cnt += 1
    # Process data here as needed

# Update visualization globals
window_name = 'Phone Tracking (Top-Down View)'
image_size = (600, 800)  # height, width
plot_height = 150
history_length = 10000
position_history = []
angle_history = []
initialized = False

def init_visualization():
    global initialized, window_name
    if not initialized:
        cv2.namedWindow(window_name)
        initialized = True

def update_visualization(transform_matrix):
    global window_name, image_size, position_history, angle_history
    
    # Create a blank image
    image = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)
    
    # Extract position and rotation
    position = transform_matrix[:3, 3]
    rotation_matrix = transform_matrix[:3, :3]
    up_vector = np.array([0, 1, 0])
    right_vector = rotation_matrix[:3, 0]
    forward_vector = np.cross(right_vector, up_vector)
    x_pos = position[2] # z -> x
    y_pos = position[0] # x -> y
    angle = np.arctan2(forward_vector[0], forward_vector[2])
    
    # Update history (only storing x, z position and y rotation)
    position_history.append([x_pos, y_pos])  # x and z only
    # angle_history.append(euler_angles[1])  # y rotation only
    angle_history.append(angle)
    if len(position_history) > history_length:
        position_history.pop(0)
        angle_history.pop(0)
    
    # Convert histories to numpy arrays
    pos_array = np.array(position_history)
    ang_array = np.array(angle_history)
    
    # Draw top-down view in the upper half
    top_view_size = 300
    center_x = image_size[1] // 2
    center_y = top_view_size // 2
    
    # Draw coordinate grid
    cv2.line(image, (center_x, 0), (center_x, top_view_size), (50, 50, 50), 1)  # vertical line
    cv2.line(image, (center_x - 150, center_y), (center_x + 150, center_y), (50, 50, 50), 1)  # horizontal line
    
    # Draw current position and orientation
    if len(pos_array) > 0:
        # Scale position for visualization (adjust scale factor as needed)
        scale = 100
        pos_x = int(center_x + pos_array[-1, 0] * scale)
        pos_z = int(center_y - pos_array[-1, 1] * scale)
        
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
        end_y = pos_z - int(arrow_length * np.sin(angle_rad))
        cv2.arrowedLine(image, (pos_x, pos_z), (end_x, end_y), (0, 0, 255), 2)
    
    # Draw plots in the lower half
    # X position plot
    y_offset = top_view_size + 50
    cv2.putText(image, 'X Position', (10, y_offset + 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.line(image, (0, y_offset + plot_height//2), 
            (image_size[1], y_offset + plot_height//2), (50, 50, 50), 1)
    if len(pos_array) > 1:
        pts = np.zeros((len(pos_array), 2), dtype=np.int32)
        pts[:, 0] = np.linspace(0, image_size[1], len(pos_array))
        scaled_vals = pos_array[:, 0] * 50
        pts[:, 1] = y_offset + plot_height//2 - scaled_vals
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(image, [pts], False, (0, 255, 0), 1)
    
    # Y rotation plot
    y_offset = top_view_size + 50 + plot_height
    cv2.putText(image, 'Y Rotation', (10, y_offset + 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.line(image, (0, y_offset + plot_height//2), 
            (image_size[1], y_offset + plot_height//2), (50, 50, 50), 1)
    if len(ang_array) > 1:
        pts = np.zeros((len(ang_array), 2), dtype=np.int32)
        pts[:, 0] = np.linspace(0, image_size[1], len(ang_array))
        scaled_vals = ang_array * 0.5
        pts[:, 1] = y_offset + plot_height//2 - scaled_vals
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(image, [pts], False, (0, 255, 0), 1)
    
    # Display the image
    cv2.imshow(window_name, image)
    cv2.waitKey(1)

# Run the server
if __name__ == '__main__':
    np.set_printoptions(precision=4, suppress=True)
    eventlet.wsgi.server(eventlet.listen(('', 5555)), app)
