from robot import Robot
import socket
import json

# Set up server details
HOST = '0.0.0.0'  # Listen on all network interfaces
PORT = 5000       # Port to listen on

# Create a socket object
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(1)  # Listen for incoming connections

follower = Robot(device_name='/dev/ttyACM0')


try:
    while True:
        data = conn.recv(1024)  # Receive up to 1024 bytes
        if not data:
            break
        # Decode and load JSON
        json_data = json.loads(data.decode())
        print("Received JSON data:", json_data)
        # follower.set_goal_pos(leader.read_position())
finally:
    conn.close()
    server_socket.close()
