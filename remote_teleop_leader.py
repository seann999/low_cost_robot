from robot import Robot
import socket
import json
import sys
import cv2
import numpy as np

# Server details
HOST = '192.168.0.231'
PORT = 5000

try:
    # Initialize robot
    leader = Robot(device_name='/dev/ttyACM0')
    leader.set_trigger_torque()

    # Create and connect socket
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((HOST, PORT))

    try:
        while True:
            # Send position data
            data_to_send = {
                "position": leader.read_position()
            }
            json_data = json.dumps(data_to_send).encode()
            client_socket.sendall(json_data)

            # Receive response header size
            header_size_bytes = client_socket.recv(4)
            header_size = int.from_bytes(header_size_bytes, byteorder='big')

            # Receive JSON header
            header_data = client_socket.recv(header_size)
            header = json.loads(header_data)

            # Get follower position and image size
            follower_position = header['position']
            image_size = header['image_size']

            # Receive image data
            img_data = b''
            while len(img_data) < image_size:
                chunk = client_socket.recv(min(image_size - len(img_data), 4096))
                if not chunk:
                    break
                img_data += chunk

            # Decode and display image
            img_np = np.frombuffer(img_data, dtype=np.uint8)
            frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
            
            # Display the frame
            cv2.imshow('Follower Camera', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
                break

            print("Leader position:", data_to_send["position"])
            print("Follower position:", follower_position)

    except KeyboardInterrupt:
        print("\nExiting gracefully...")
    
    except Exception as e:
        print(f"An error occurred: {e}")

finally:
    # Cleanup
    cv2.destroyAllWindows()
    client_socket.close()
    print("Connection closed")