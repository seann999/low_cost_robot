from ikpy.utils import geometry, plot as plot_utils
from ikpy.utils.plot import plot_frame
import numpy as np
import matplotlib.pyplot as plt


camera2gripper = np.array([
    [
        0.9997668505539576,
        -0.019970828271681517,
        -0.008210392899446018, 0.00018299471587136708,
    ],
    [
        -0.021388860005435672,
        -0.8638202079680002,
        -0.503345969462147, 0.06175676142391705,
    ],
    [
        0.0029599326154731358,
        0.5034042255725152,
        -0.864045962015128, -0.00888269351076889,
    ],
    [
        0, 0, 0, 1
    ]
])


class KinematicsPlotter:
    def __init__(self):
        self.figures = {}
        self.axes = {}

    def raw_plot_chain(self, chain, joints, ax, name="chain", other_frames=()):
        """Plot the chain with its rotation and translation axes"""
        nodes = []
        rotation_axes = []
        translation_axes = []

        transformation_matrixes = chain.forward_kinematics(joints, full_kinematics=True)

        # Get nodes and orientations
        for (index, link) in enumerate(chain.links):
            (node, orientation) = geometry.from_transformation_matrix(transformation_matrixes[index])
            nodes.append(node)

            # Handle rotation axes
            if link.has_rotation:
                rotation_axis = link.get_rotation_axis()
                if index == 0:
                    rotation_axes.append((node, rotation_axis))
                else:
                    rotation_axes.append((node, geometry.homogeneous_to_cartesian_vectors(
                        np.dot(transformation_matrixes[index - 1], rotation_axis))))

            # Handle translation axes
            if link.has_translation:
                translation_axis = link.get_translation_axis()
                if index == 0:
                    translation_axes.append((node, translation_axis))
                else:
                    translation_axes.append((node, geometry.homogeneous_to_cartesian_vectors(
                        np.dot(transformation_matrixes[index - 1], translation_axis))))

        # Plot chain and nodes
        lines = ax.plot([x[0] for x in nodes], [x[1] for x in nodes], 
                       [x[2] for x in nodes], linewidth=5, label=name)
        ax.scatter([x[0] for x in nodes], [x[1] for x in nodes], 
                  [x[2] for x in nodes], s=55, c=lines[0].get_color())

        # Plot rotation and translation axes
        for node, axe in rotation_axes:
            ax.plot([node[0], axe[0]], [node[1], axe[1]], [node[2], axe[2]], 
                   c=lines[0].get_color())
        
        for node, axe in translation_axes:
            ax.plot([node[0], axe[0]], [node[1], axe[1]], [node[2], axe[2]], 
                   c=lines[0].get_color(), linestyle='dotted', linewidth=2.5)

        # Plot frames
        camera_frame = np.dot(transformation_matrixes[-1], camera2gripper)
        plot_frame(camera_frame, ax, length=chain.links[-1].length)

        for frame in other_frames:
            plot_frame(frame, ax, length=chain.links[-1].length)

    def plot_chain(self, chain, joints, ax=None, target_matrix=None, show=False, other_frames=()):
        """Plot the kinematic chain with optional target"""
        if ax is None:
            _, ax = plot_utils.init_3d_figure()
            
        self.raw_plot_chain(chain, joints, ax, name=chain.name, other_frames=other_frames)

        if target_matrix is not None:
            plot_frame(target_matrix, ax, length=chain.links[-1].length)
            
        if show:
            plot_utils.show_figure()

    def initialize_plot(self, plot_id=0):
        """Initialize a new plot with the given ID"""
        self.figures[plot_id], self.axes[plot_id] = plot_utils.init_3d_figure()
        plt.ion()
        plt.show()

    def update_plot(self, robot_arm_chain, joint_angles, plot_id, plot_new=True, target_matrix=None, **kwargs):
        """Update the plot with new joint angles"""
        if plot_id not in self.figures:
            self.initialize_plot(plot_id)

        if plot_new:
            self.axes[plot_id].clear()

        self.plot_chain(robot_arm_chain, joint_angles, self.axes[plot_id], 
                       target_matrix=target_matrix, **kwargs)

        if plot_new:
            self._set_equal_aspect_ratio(plot_id)
            plt.draw()
            plt.pause(0.001)

    def _set_equal_aspect_ratio(self, plot_id):
        """Helper method to set equal aspect ratio for 3D plot"""
        ax = self.axes[plot_id]
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()
        
        x_range = x_limits[1] - x_limits[0]
        y_range = y_limits[1] - y_limits[0]
        z_range = z_limits[1] - z_limits[0]
        
        max_range = max(x_range, y_range, z_range)
        
        x_middle = np.mean(x_limits)
        y_middle = np.mean(y_limits)
        z_middle = np.mean(z_limits)
        
        ax.set_xlim3d([x_middle - max_range / 2, x_middle + max_range / 2])
        ax.set_ylim3d([y_middle - max_range / 2, y_middle + max_range / 2])
        ax.set_zlim3d([z_middle - max_range / 2, z_middle + max_range / 2])