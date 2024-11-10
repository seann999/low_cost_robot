from robot import Robot
import socket
import json
import cv2
import numpy as np

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

# Add webcam setup
cap = cv2.VideoCapture(0)  # 0 is usually the built-in webcam
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

try:
    while True:
        print("Waiting for client connection...")
        conn, addr = server_socket.accept()
        print(f"Connected by {addr}")
        
        try:
            while True:
                data = conn.recv(1024)  # Receive up to 1024 bytes
                if not data:
                    print(f"Client {addr} disconnected")
                    break
                try:
                    # Decode and load JSON
                    json_data = json.loads(data.decode())
                    # print("Received JSON data:", json_data)
                    action = json_data['position']
                    # action = follower.read_position()
                    # print(follower.read_position())
                    follower.set_goal_pos(action)
                    
                    # Read current position
                    current_pos = follower.read_position()
                    
                    # Capture and send frame
                    ret, frame = cap.read()
                    if ret:
                        # Compress frame to JPEG
                        _, img_encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                        # Convert to bytes and get size
                        img_bytes = img_encoded.tobytes()
                        size = len(img_bytes)
                        
                        # Create response packet with both position and image
                        response = {
                            'position': current_pos,
                            'image_size': size
                        }
                        
                        # Send JSON header first
                        header = json.dumps(response).encode()
                        header_size = len(header)
                        
                        # Send header size, header, then image data
                        conn.send(header_size.to_bytes(4, byteorder='big'))
                        conn.send(header)
                        conn.send(img_bytes)
                        
                except json.decoder.JSONDecodeError:
                    print("Received invalid JSON data")
        except socket.error:
            print(f"Lost connection to client {addr}")
        finally:
            conn.close()
except KeyboardInterrupt:
    print("\nServer shutting down...")
finally:
    cap.release()
    server_socket.close()
