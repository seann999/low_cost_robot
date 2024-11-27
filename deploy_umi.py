import torch
import cv2
import time
import numpy as np
import dill
import hydra
import os
import zarr
import math
from copy import deepcopy
from scipy.spatial.transform import Rotation as R

from robot import Robot
from robot_wrapper import RobotEnv, joints_to_posemat, pose7d_to_joints, cam_move_to_ee, plotter, robot_arm_chain, camera2gripper

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from umi.common.pose_util import pose_to_mat, mat_to_pose, mat_to_pose10d, rot6d_to_mat
from umi.common.interpolation_util import get_interp1d, PoseInterpolator
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, JpegXl
from umi.common.cv_util import draw_predefined_mask
from visualize_zarr import draw_actions

register_codecs()


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def load_checkpoint(ckpt_path):
    if not ckpt_path.endswith('.ckpt'):
        ckpt_path = os.path.join(ckpt_path, 'checkpoints', 'latest.ckpt')
    payload = torch.load(open(ckpt_path, 'rb'), map_location='cpu', pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    policy.num_inference_steps = 16 # DDIM inference iterations
    obs_pose_rep = cfg.task.pose_repr.obs_pose_repr
    action_pose_repr = cfg.task.pose_repr.action_pose_repr
    print('obs_pose_rep', obs_pose_rep)
    print('action_pose_repr', action_pose_repr)

    policy.eval().to(device)

    return policy


def process_observation(observation):
    # Create a copy of the observation
    obs = deepcopy(observation)
    
    # Process the copied observation
    image = cv2.resize(obs.image, (224, 224))
    img = draw_predefined_mask(image, color=(0,0,0), 
                    mirror=False, gripper=True, finger=False)
    obs.image = image / 255.0
    
    return obs


def get_obs_dict(history):
    camera0_rgb = np.stack([np.transpose(history[i].image, (2, 0, 1)) for i in [-2, -1]])

    robot0_matrices = np.stack([history[i].cam_pose for i in [-2, -1]])

    base_matrix = robot0_matrices[-1]
    robot0_matrices = np.linalg.inv(base_matrix) @ robot0_matrices
    robot0_poses = np.stack([mat_to_pose10d(mat) for mat in robot0_matrices])[:, :9]

    robot0_widths = (np.stack([history[i].joints[-1] for i in [-2, -1]]) / 2048 - 1.0) * np.pi
    robot0_widths = robot0_widths[:, None]

    return dict(
        camera0_rgb=camera0_rgb,
        robot0_eef_pos=robot0_poses[:, :3],
        robot0_eef_rot_axis_angle=robot0_poses[:, 3:9],
        robot0_eef_rot_axis_angle_wrt_start=robot0_poses[:, 3:9],
        robot0_gripper_width=robot0_widths,
    )


import matplotlib.pyplot as plt


def execute_actions(env, history, action_seq):
    # print('>>', history[-1].joints)

    joint_commands = []
    matrices = []
    offsets = []

    for i, action in enumerate(action_seq[:4]):
        new_cam_mat, new_ee_mat = cam_move_to_ee(history[-1].cam_pose, action[:3], action[3:9])

        x_offset = new_ee_mat[0, 3]
        y_offset = new_ee_mat[1, 3] - history[-1].ee_pose[1, 3]
        offsets.append([x_offset, y_offset])
        new_ee_mat[0, 3] = 0
        new_ee_mat[1, 3] = history[-1].ee_pose[1, 3]

        new_cam_mat = new_ee_mat @ camera2gripper

        move_pos = new_ee_mat[:3, 3]
        move_rot = R.from_matrix(new_ee_mat[:3, :3]).as_quat()
        new_ee_pose = np.concatenate([move_pos, move_rot])
        
        # if np.rad2deg(action[9]) < 45:
        #     print('gripper close')
        #     gripper_offset = np.deg2rad(20)
        # else:
        #     print('gripper open')
        gripper_offset = np.deg2rad(5)
        gripper_units = (action_seq[i+1, 9] - gripper_offset) / np.pi
        gripper_joint = int(gripper_units * 2048 + 2048)

        try:
            joints = pose_to_joints(new_ee_pose, gripper_joint, history[-1].joints)
        except Exception as e:
            print(e)
            return

        joint_commands.append(joints)
        matrices.append(new_ee_mat)
        matrices.append(new_cam_mat)

    offsets = np.array(offsets)
    deltas = np.concatenate([offsets[:1], offsets[1:] - offsets[:-1]])

    # Create a figure for plotting deltas
    # plt.figure(figsize=(8, 6))
    # plt.scatter(offsets[:, 0], offsets[:, 1], marker='o')

    # current = history[-1].ee_pose[:3, 3]
    # plt.scatter(current[0], current[1], marker='*')
    # plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    # plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    # plt.xlabel('Step')
    # plt.ylabel('Delta (m)')
    # plt.title('X-Y Deltas Between Steps')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # index = 0
    # while True:
    #     i = index % len(outs)
    #     index += 1

    #     plot_joints = outs[i]
    #     plot_joints = (np.array(plot_joints) - 2048) / 2048 * np.pi
    #     plot_joints = np.concatenate([[0], plot_joints[:-1]])

    #     plotter.update_plot(robot_arm_chain, plot_joints, plot_id="follower", other_frames=matrices)# , target_matrix=matrices[i])
    #     plt.pause(0.01)

    #     k = cv2.waitKey(1)

    #     if k == ord('n'):
    #         break
    #     elif k == ord('s'):
    #         outs = []
    #         break
    #     elif k == ord('q'):
    #         raise KeyboardInterrupt

    base_commands = []
    missed_deltas = []
    for i in range(len(deltas)):
        if len(missed_deltas) > 0:
            for j in range(len(missed_deltas)):
                deltas[i] += missed_deltas[j]
            missed_deltas = []

        delta_x, delta_y = deltas[i]

        direction = math.degrees(math.atan2(delta_y, delta_x)) % 360  # Convert to degrees and normalize to 0-360
        magnitude = min(math.sqrt(delta_x**2 + delta_y**2), 1) * 2000 # 4000
        if magnitude < 20:
            magnitude = 0
            missed_deltas.append(deltas[i].copy())
        base_commands.append([magnitude, direction, 0])

    for i in range(len(joint_commands)):
        magnitude, direction, skew = base_commands[i]
        print(magnitude)
        env.send_action(joint_commands[i], base_commands[i])
        # time.sleep(0.1)

        # env.send_action(joint_commands[i], [0, 0, 0])
        time.sleep(0.1)

    env.stop_base()


def main():
    ckpt_path = '/home/sean/projects/universal_manipulation_interface/data/outputs/2024.11.07/03.07.41_train_diffusion_unet_timm_umi/checkpoints/latest.ckpt'
    policy = load_checkpoint(ckpt_path)

    frequency = 10
    history = []

    try:
        env = RobotEnv()
        env.connect()

        for _ in range(10):
            history.append(process_observation(env.get_observation()))

        with torch.no_grad():
            policy.reset()
            obs_dict_np = get_obs_dict(history)
            obs_dict = dict_apply(obs_dict_np, 
                lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
            result = policy.predict_action(obs_dict)
            action = result['action_pred'][0].detach().to('cpu').numpy()
        
        while True:
            # Get leader position and send to follower
            # leader_position = leader.read_position()

            # leader_pose = joints_to_pose(leader_position)
            # print(leader_pose)
            # leader_position_rec = pose_to_joints(leader_pose, leader_position)
            # print(leader_position_rec)

            # env.send_joints(leader_position)
            
            # Get latest observation
            obs = env.get_observation()
            history.append(process_observation(obs))
            # time.sleep(0.1)

            obs_dict_np = get_obs_dict(history)

            # T C H W -> T H W C
            obs_image = np.hstack(np.transpose(obs_dict_np['camera0_rgb'], (0, 2, 3, 1)))
            cv2.imshow('RGB', obs_image)

            obs_dict = dict_apply(obs_dict_np, 
                lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
            result = policy.predict_action(obs_dict)
            action_seq = result['action_pred'][0].detach().to('cpu').numpy()

            image = draw_actions(history[-1].image.copy(), action_seq[:, :3], action_seq[0:1, -1])
            cv2.imshow('Camera', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            execute_actions(env, history, action_seq)
            
    except KeyboardInterrupt:
        print("\nExiting gracefully...")
    except Exception as e:
        import traceback
        print(f"\nError Details:")
        print(f"Type: {type(e).__name__}")
        print(f"Message: {str(e)}")
        print("\nFull traceback:")
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()
        env.close()
        print("Connection closed")

    # time.sleep(1/frequency)

if __name__ == "__main__":
    main()