from robot import Robot
from robot_env import RobotEnv

# Initialize the robot and environment
follower = Robot(device_name='/dev/ttyACM0')
follower._enable_torque()
env = RobotEnv()

# Connect to the environment
env.connect()

try:
    while True:
        # Get the latest observation from the environment
        observation = env.get_observation()
        
        # If we have valid joint positions, set them as the goal
        if observation.ee_joints is not None:
            follower.set_goal_pos(observation.ee_joints)
            
finally:
    # Clean up
    env.close()
