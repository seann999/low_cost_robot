import cv2
from robot import Robot
from robot_wrapper import RobotEnv, joints_to_pose, pose_to_joints


def main():
    try:
        # Initialize robot and environment
        leader = Robot(device_name='/dev/ttyACM0')
        leader.set_trigger_torque()
        
        env = RobotEnv()
        env.connect()
        
        while True:
            # Get leader position and send to follower
            leader_position = leader.read_position()

            # leader_pose = joints_to_pose(leader_position)
            # print(leader_pose)
            # leader_position_rec = pose_to_joints(leader_pose, leader_position)
            # print(leader_position_rec)

            env.send_joints(leader_position)
            
            # Get latest observation
            obs = env.get_observation()
            
            # Display frame if available
            if obs.image is not None:
                cv2.imshow('Follower Camera', obs.image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            print("Leader position:", leader_position)
            print("Follower position:", obs.joints)
            
    except KeyboardInterrupt:
        print("\nExiting gracefully...")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        cv2.destroyAllWindows()
        env.close()
        print("Connection closed")

if __name__ == "__main__":
    main()