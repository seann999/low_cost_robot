from robot import Robot
import socket
import json
import cv2
import numpy as np
import time
import threading
from queue import Queue
import sys
sys.path.append('/home/pi/TurboPi/')
import HiwonderSDK.mecanum as mecanum


chassis = mecanum.MecanumChassis()
follower = Robot(device_name='/dev/ttyACM0')
follower._enable_torque()

# Set up server details
HOST = '0.0.0.0'  # Listen on all network interfaces
PORT = 5000       # Port to listen on

# Create a socket object
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(1)  # Listen for incoming connections

print(f"Server listening on {HOST}:{PORT}...")

def accept_connection():
    conn, addr = server_socket.accept()
    print(f"Connected by {addr}")
    conn.setblocking(1)  # Make sure connection is blocking by default
    return conn, addr

# Move webcam setup here, before the main loop
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

def receive_position_data(conn, follower, running):
    while running[0]:
        try:
            # First receive the size header (4 bytes)
            size_data = conn.recv(4, socket.MSG_WAITALL)
            if not size_data:  # Connection closed
                break
                
            msg_size = int.from_bytes(size_data, byteorder='big')
            # Then receive the actual message
            data = conn.recv(msg_size, socket.MSG_WAITALL)
            if not data:  # Connection closed
                break
                
            json_data = json.loads(data.decode())
            if 'position' in json_data:
                follower.set_goal_pos(json_data['position'])
            if 'base' in json_data:
                magnitude, direction, skew = json_data['base']
                chassis.set_velocity(magnitude, direction, skew)
                
        except (BlockingIOError, socket.error, ConnectionResetError) as e:
            print(f"Position receive error: {e}")
            continue
            
    print("Position receive thread ended")

try:
    conn, addr = accept_connection()
    running = [True]  # Using list to allow modification in threads
    
    # Start position receiving thread
    position_thread = threading.Thread(
        target=receive_position_data, 
        args=(conn, follower, running)
    )
    position_thread.start()
    
    while True:
        try:
            # Main thread handles only camera and position updates
            current_pos = follower.read_position()
            ret, frame = cap.read()
            
            if ret:
                # Compress frame to JPEG
                _, img_encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                img_bytes = img_encoded.tobytes()
                
                # Create response packet
                response = {
                    'position': current_pos,
                }
                response_json = json.dumps(response).encode()
                
                try:
                    # Send position data with size header
                    conn.send(len(response_json).to_bytes(4, byteorder='big'))
                    conn.send(response_json)
                    
                    # Send image data with size header
                    conn.send(len(img_bytes).to_bytes(4, byteorder='big'))
                    conn.send(img_bytes)
                except (BrokenPipeError, ConnectionResetError):
                    print("Client disconnected. Waiting for new connection...")
                    conn.close()
                    # Stop the position thread
                    running[0] = False
                    position_thread.join()
                    # Get new connection
                    conn, addr = accept_connection()
                    # Restart position thread
                    running[0] = True
                    position_thread = threading.Thread(
                        target=receive_position_data, 
                        args=(conn, follower, running)
                    )
                    position_thread.start()
            
            # Reduced sleep time
            time.sleep(0.01)  # ~100fps max
            
        except Exception as e:
            print(f"Error in main loop: {e}")
            if isinstance(e, (ConnectionResetError, BrokenPipeError)):
                print("Client disconnected. Waiting for new connection...")
                conn.close()
                # Stop the position thread
                running[0] = False
                position_thread.join()
                # Get new connection
                conn, addr = accept_connection()
                # Restart position thread
                running[0] = True
                position_thread = threading.Thread(
                    target=receive_position_data, 
                    args=(conn, follower, running)
                )
                position_thread.start()

finally:
    running[0] = False  # Stop the position thread
    if 'position_thread' in locals():
        position_thread.join()
    cap.release()
    if 'conn' in locals():
        conn.close()
    server_socket.close()
