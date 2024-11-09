from robot import Robot
import socket
import json
import sys

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
            # JSON data to send
            data_to_send = {
                "position": leader.read_position()
            }

            # Encode and send data
            json_data = json.dumps(data_to_send).encode()
            client_socket.sendall(json_data)
            print("Sent JSON data:", data_to_send)

    except KeyboardInterrupt:
        print("\nExiting gracefully...")
    
    except Exception as e:
        print(f"An error occurred: {e}")

finally:
    # Ensure socket is closed
    client_socket.close()
    print("Connection closed")