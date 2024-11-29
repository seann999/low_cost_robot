#!/usr/bin/python3
# coding=utf8
import sys
import time
import signal
import math
from evdev import InputDevice, list_devices
import time
from robot_wrapper import RobotEnv
import numpy as np
from copy import deepcopy
from scipy.spatial.transform import Rotation as R


# Add this near the top of the file, after imports
BUTTON_MAP = {
    'A': 304,
    'B': 305,
    'X': 307,
    'Y': 308,
    'LB': 310,
    'RB': 311,
    'start': 315,
    'left_joystick': 317,
    'right_joystick': 318,
}

DPAD_AXES = {
    'horizontal': 16,  # Left/Right on D-pad
    'vertical': 17,    # Up/Down on D-pad
}

def is_button_clicked(button_name, flip=False):
    """
    Returns True if button was just pressed (transition from 0 to 1)
    button_name: string from BUTTON_MAP keys
    """
    if button_name not in BUTTON_MAP:
        return False
    button_code = BUTTON_MAP[button_name]

    if flip:
        return (button_states.get(button_code, 0) == 0 and 
                previous_button_states.get(button_code, 0) == 1)
    else:
        return (button_states.get(button_code, 0) == 1 and 
                previous_button_states.get(button_code, 0) == 0)

def is_button_pressed(button_name):
    """
    Returns True if button is pressed (value > 0)
    """
    if button_name not in BUTTON_MAP:
        return False
    button_code = BUTTON_MAP[button_name]
    return button_states.get(button_code, 0) > 0

def get_dpad_state():
    """
    Returns the D-pad state as (horizontal, vertical) where:
    horizontal: -1 (left), 0 (center), 1 (right)
    vertical: -1 (up), 0 (center), 1 (down)
    """
    horizontal = axis_states.get(DPAD_AXES['horizontal'], 0)
    vertical = axis_states.get(DPAD_AXES['vertical'], 0)
    return horizontal, vertical

def is_dpad_pressed(direction):
    """
    Check if a specific D-pad direction is pressed
    direction: 'up', 'down', 'left', or 'right'
    """
    h, v = get_dpad_state()
    if direction == 'up': return v == -1
    if direction == 'down': return v == 1
    if direction == 'left': return h == -1
    if direction == 'right': return h == 1
    return False

devices = [InputDevice(path) for path in list_devices()]

for device in devices:
    if 'Logitech' in device.name:
        gamepad = device
        break

# Print device info
print(f"Gamepad detected: {gamepad.name}")

# Initialize state variables
button_states = {}
previous_button_states = {}
axis_states = {}

# Non-blocking event reading
gamepad.grab()  # Take exclusive control of the device


if __name__ == '__main__':
    env = RobotEnv()
    env.connect()
    env.home_joints()
    # env.start_trajectory_executor()

    def toggle_gripper():
        joints = env.get_observation().joints
        gripper_deg = np.rad2deg(env.get_gripper_rad())

        if gripper_deg > 20:
            joints[-1] = env.gripper_rad_to_joint(0)
            print('close gripper')
        else:
            joints[-1] = env.gripper_rad_to_joint(np.deg2rad(55))
            print('open gripper')
            
        env.send_joints(joints)

    def move_joint(joint_id, direction):
        joints = env.get_observation().joints
        joints[joint_id] += direction
        env.send_joints(joints)

    # origin_base = None
    target_ee_pose = None

    while True:
        previous_button_states = button_states.copy()
        try:
            # Read all pending events
            events = gamepad.read()
            for event in events:
                if event.type == 1:  # Button event
                    # Store previous state before updating
                    button_states[event.code] = event.value
                elif event.type == 3:  # Axis event
                    axis_states[event.code] = event.value
        except BlockingIOError:
            # No events available, just continue
            pass
        except KeyboardInterrupt:
            chassis.set_velocity(0,0,0)  # 关闭所有电机
            # Release the device on Ctrl+C
            gamepad.ungrab()
            break

        if is_button_clicked('right_joystick'):
            toggle_gripper()
        # axis_states.get(5, 0) / 255.

        # print('previous_button_states', previous_button_states)
        # print('button_states', button_states)
        # print('axis_states', axis_states)

        left_x = float(axis_states.get(0, 0)) / 32768.0
        left_y = float(axis_states.get(1, 0)) / 32768.0
        right_x = float(axis_states.get(3, 0)) / 32768.0
        right_y = float(axis_states.get(4, 0)) / 32768.0

        if is_button_clicked('RB'):
            obs = deepcopy(env.get_observation())
            # origin_base = obs.arm_base_pose
            target_ee_pose = obs.ee_pose
            # print('origin_base', origin_base)
        elif is_button_clicked('RB', flip=True):
            # origin_base = None
            target_ee_pose = None
            # print('origin_base', origin_base)

        if is_button_pressed('RB'):
            obs = deepcopy(env.get_observation())
            origin_base = obs.arm_base_pose

            gripper_rad = env.get_gripper_rad()
            dt = 1/50
            rot_x, rot_y, rot_z = 0, 0, 0
            if is_dpad_pressed('left'): rot_z = -dt * 100
            elif is_dpad_pressed('right'): rot_z = dt * 100
            if is_dpad_pressed('up'): rot_x = dt * 100
            elif is_dpad_pressed('down'): rot_x = -dt * 100
            if is_button_pressed('X'): rot_y = -dt * 100
            elif is_button_pressed('B'): rot_y = dt * 100

            # Create rotation in origin_base frame
            rot_xyz = R.from_euler('xyz', [rot_x, rot_y, rot_z], degrees=True).as_matrix()

            # Get current position
            current_pos = target_ee_pose[:3, 3]
            
            # Translate to origin_base, rotate, translate back
            translated_pose = target_ee_pose.copy()
            translated_pose[:3, 3] -= current_pos  # Translate to origin
            # translated_pose[:3, :3] = origin_base[:3, :3] @ rot_xyz
            target_ee_pose[:3, :3] = origin_base[:3, :3] @ rot_xyz @ origin_base[:3, :3].T @ target_ee_pose[:3, :3]
            # target_ee_pose[:3, :3] = translated_pose[:3, :3]  # Apply rotation

            # Translation in origin_base frame
            speed = 0.2
            velocities = np.array([speed * left_x, -speed * left_y, -speed * right_y]) * dt
            target_ee_pose[:3, 3] += origin_base[:3, :3] @ velocities

            env.add_ee_waypoint(time.time() + dt, target_ee_pose, gripper_rad=gripper_rad)

            if not is_button_pressed('LB'):
                env.move_base_trajectory(time.time())
            env.move_arm_trajectory(time.time())
            time.sleep(dt)
            continue
        elif axis_states.get(5, 0) > 0:
            if is_dpad_pressed('up'):
                move_joint(2, 30)
            if is_dpad_pressed('down'):
                move_joint(2, -30)
            if is_button_pressed('Y'):
                move_joint(3, 30)
            if is_button_pressed('A'):
                move_joint(3, -30)
            if is_button_pressed('X'):
                move_joint(4, -30)
            if is_button_pressed('B'):
                move_joint(4, 30)
        else:
            if is_dpad_pressed('up'):
                move_joint(1, -30)
            if is_dpad_pressed('down'):
                move_joint(1, 30)
            if is_dpad_pressed('left'):
                move_joint(0, 30)
            if is_dpad_pressed('right'):
                move_joint(0, -30)

        if is_button_pressed('LB'):
            direction = math.degrees(math.atan2(-left_y, left_x)) % 360  # Convert to degrees and normalize to 0-360
            magnitude = min(math.sqrt(left_x**2 + left_y**2), 1)
            env.send_base([magnitude * 90, direction, right_x * 0.3])
        else:
            env.send_base([0, 0, 0])

        if is_button_pressed('start'):
            env.home_joints()

        time.sleep(1/50)
