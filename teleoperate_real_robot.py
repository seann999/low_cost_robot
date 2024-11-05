from robot import Robot
from dynamixel import Dynamixel


import ikpy
from ikpy.chain import Chain
from ikpy.link import OriginLink, URDFLink
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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


# Modify the URDF file
modified_urdf = modify_urdf_for_ikpy("low_cost_robot.urdf")
# Load the modified URDF file
robot_arm_chain = Chain.from_urdf_file(modified_urdf)

# leader_dynamixel = Dynamixel.Config(baudrate=1_000_000, device_name='/dev/ttyACM0').instantiate()
# follower_dynamixel = Dynamixel.Config(baudrate=1_000_000, device_name='/dev/ttyACM1').instantiate()
follower = Robot('/dev/ttyACM1')
leader = Robot('/dev/ttyACM0')
leader.set_trigger_torque()

# Set up the plot
plt.ion()  # Turn on interactive mode
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

while True:
    raw_state = leader.read_position()
    print("raw state", raw_state)
    state = [0] + raw_state[:-1]
    joints = (np.array(state) - 2048) / 2048 * np.pi
    
    # Clear the previous plot
    ax.clear()
    
    # Plot the new position
    robot_arm_chain.plot(joints, ax, show=False)
    
    # Set consistent axis limits
    ax.set_xlim(-0.2, 0.2)
    ax.set_ylim(-0.2, 0.2)
    ax.set_zlim(-0.2, 0.2)
    
    # Update the plot
    plt.draw()
    plt.pause(0.01)  # Small pause to allow the plot to update

    print("actual angles:", joints)
    pose = robot_arm_chain.forward_kinematics(joints)
    print("End effector position:", pose)
    # print("Raw state:", state)

    joints = robot_arm_chain.inverse_kinematics(
        target_position=pose[:3, -1],
        target_orientation=pose[:3, :3],
        orientation_mode="all")
    joints = np.array(joints) / np.pi * 2048 + 2048
    joints[:-1] = joints[1:]
    joints[-1] = raw_state[-1]
    joints = np.round(joints).astype(np.int32)
    print("s", state)
    print(joints)
    follower.set_goal_pos(joints)

    if plt.waitforbuttonpress(timeout=0.01):
        break

plt.ioff()  # Turn off interactive mode
plt.show()