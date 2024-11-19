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
from robot_wrapper import RobotEnv, cam_move_to_ee, plotter, robot_arm_chain, camera2gripper

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
    n_obs = 1
    indices = [-1]

    camera0_rgb = np.stack([np.transpose(history[i].image, (2, 0, 1)) for i in indices])

    robot0_matrices = np.stack([history[i].cam_pose for i in indices])

    base_matrix = robot0_matrices[-1]
    robot0_matrices = np.linalg.inv(base_matrix) @ robot0_matrices
    robot0_poses = np.stack([mat_to_pose10d(mat) for mat in robot0_matrices])[:, :9]

    robot0_widths = (np.stack([history[i].joints[-1] for i in indices]) / 2048 - 1.0) * np.pi
    robot0_widths = robot0_widths[:, None]

    return dict(
        camera0_rgb=camera0_rgb,
        robot0_eef_pos=robot0_poses[:, :3],
        robot0_eef_rot_axis_angle=robot0_poses[:, 3:9],
        robot0_eef_rot_axis_angle_wrt_start=robot0_poses[:, 3:9],
        robot0_gripper_width=robot0_widths,
    )


def execute_actions(env, history, action_seq):
    poses = dict()

    # print(action_seq.shape)
    # print([np.rad2deg(act[9]) for act in action_seq])

    for i, action in enumerate(action_seq[:8]):
        # if i < 4:
        new_cam_mat, new_ee_mat = cam_move_to_ee(history[-1].cam_pose, action[:3], action[3:9])
        gripper_offset = np.deg2rad(0)
        gripper_rad = action[9] - gripper_offset
        # print('gripper_rad', np.rad2deg(gripper_rad))
        # for _ in range(5):
        #     goal_arm_base_pose = env.move_ee_to(new_ee_mat, gripper_rad=gripper_rad, wait_base=False)
        #     time.sleep(0.02)
        goal_arm_base_pose = env.move_ee_to(new_ee_mat, gripper_rad=gripper_rad, wait_base=True)
        env.stop_base()
        time.sleep(0.1)

        goal_ee_pose = new_ee_mat
        actual_base_pose, actual_ee_pose, _ = env.get_world_pose()
        poses[f'desired_arm_{i}'] = (goal_arm_base_pose.copy(), 'lightcoral')
        poses[f'desired_ee_{i}'] = (goal_ee_pose.copy(), 'lightcoral')
        poses[f'actual_ee_{i}'] = (actual_ee_pose, 'lightcoral')
        poses[f'actual_arm_{i}'] = (actual_base_pose, 'lightcoral')

    # plotter.plot_poses(poses)

    env.stop_base()


def main():
    ckpt_path = '/home/sean/projects/universal_manipulation_interface/data/outputs/2024.11.18/04.17.35_train_diffusion_unet_timm_umi/checkpoints/latest.ckpt'
    policy = load_checkpoint(ckpt_path)

    frequency = 10
    history = []
    running = False
    execute_once = False  # New flag for single execution

    try:
        env = RobotEnv()
        env.connect()
        env.home_joints()

        home_base = env.get_base_pose()

        def refresh():
            history.clear()
            for _ in range(10):
                obs = deepcopy(env.get_observation())
                history.append(process_observation(obs))

        refresh()

        while True:
            obs = deepcopy(env.get_observation())
            history.append(process_observation(obs))

            obs_dict_np = get_obs_dict(history)
            obs_image = np.hstack(np.transpose(obs_dict_np['camera0_rgb'], (0, 2, 3, 1)))
            cv2.imshow('RGB', obs_image)

            # Get prediction regardless of running state
            obs_dict = dict_apply(obs_dict_np, 
                lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
            result = policy.predict_action(obs_dict)
            action_seq = result['action_pred'][0].detach().to('cpu').numpy()

            if running or execute_once:
                print('obs gripper width', np.rad2deg(obs_dict_np['robot0_gripper_width']))
                image = draw_actions(history[-1].image.copy(), action_seq[:, :3], action_seq[:, -1])
                cv2.imshow('Camera', image)

                execute_actions(env, history, action_seq)
                execute_once = False  # Reset the execute_once flag after execution
            else:
                cv2.imshow('Camera', history[-1].image)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                running = not running  # Toggle running state
                print("Running:" if running else "Stopped")

                if running:
                    home_base = env.get_base_pose()
            elif key == ord('r') and not running:
                env.home_joints()
                env.move_base_to_wait(home_base['x'], home_base['y'], home_base['yaw'])
                refresh()
            elif key == ord('e') and not running:
                execute_once = True
                print("Executing single action sequence...")
            
            time.sleep(1/frequency)

    except KeyboardInterrupt:
        print("\nExiting gracefully...")
    finally:
        cv2.destroyAllWindows()
        env.close()
        print("Connection closed")

if __name__ == "__main__":
    main()