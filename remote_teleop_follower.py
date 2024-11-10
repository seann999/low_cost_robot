from robot import Robot
import socket
import json
import cv2
import numpy as np
import time

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

try:
    conn, addr = accept_connection()
    while True:
        try:
            # Try to receive data without blocking
            conn.setblocking(0)
            try:
                # First receive the size header (4 bytes)
                size_data = conn.recv(4)
                if size_data:
                    msg_size = int.from_bytes(size_data, byteorder='big')
                    # Then receive the actual message
                    data = conn.recv(msg_size)
                    if data:
                        json_data = json.loads(data.decode())
                        if 'position' in json_data:
                            follower.set_goal_pos(json_data['position'])
            except (BlockingIOError, socket.error):
                # No data available, continue with sending updates
                pass

            # Always send current state and image
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
                    conn, addr = accept_connection()
            
            # Add a small sleep to prevent overwhelming the network
            time.sleep(0.03)  # ~30fps
            
        except Exception as e:
            print(f"Error: {e}")
            # If connection is lost, try to reconnect
            if isinstance(e, (ConnectionResetError, BrokenPipeError)):
                print("Client disconnected. Waiting for new connection...")
                conn.close()
                conn, addr = accept_connection()

finally:
    cap.release()
    if 'conn' in locals():
        conn.close()
    server_socket.close()
