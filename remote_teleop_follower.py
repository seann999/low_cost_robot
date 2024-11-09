from robot import Robot
import socket
import json

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

# Accept a connection
conn, addr = server_socket.accept()
print(f"Connected by {addr}")


try:
    while True:
        data = conn.recv(1024)  # Receive up to 1024 bytes
        if not data:
            break
        try:
            # Decode and load JSON
            json_data = json.loads(data.decode())
            # print("Received JSON data:", json_data)
            action = json_data['position']
            # action = follower.read_position()
            # print(follower.read_position())
            follower.set_goal_pos(action)
        except json.decoder.JSONDecodeError:
            print("Received invalid JSON data")
finally:
    conn.close()
    server_socket.close()
