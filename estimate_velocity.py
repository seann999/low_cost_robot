import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

def calculate_velocities(data_input, data_result, plot=False):
    """
    Calculate velocities for each trajectory in data_result
    
    Args:
        data_input: numpy array of input parameters
        data_result: list of trajectory results
        plot: boolean to enable/disable plotting (default: False)
    
    Returns:
        numpy array of shape (n, 3) containing [vx, vy, vyaw] for each entry
    """
    velocities = []
    
    for traj_data in data_result:
        traj = traj_data[25:]  # Skip first 25 points as in original code
        
        # Get initial pose
        initial_pose = traj[0]
        initial_x = initial_pose['x']
        initial_y = initial_pose['y']
        initial_yaw = initial_pose['yaw']
        
        # Transform trajectory relative to first point
        transformed_traj = []
        for pose in traj:
            dx = pose['x'] - initial_x
            dy = pose['y'] - initial_y
            rotated_x = dx * np.cos(-initial_yaw) - dy * np.sin(-initial_yaw)
            rotated_y = dx * np.sin(-initial_yaw) + dy * np.cos(-initial_yaw)
            dyaw = (pose['yaw'] - initial_yaw + np.pi) % (2 * np.pi) - np.pi
            
            transformed_traj.append({
                'x': rotated_x,
                'y': rotated_y,
                'yaw': dyaw,
                'timestamp': pose['timestamp']
            })
        
        # Calculate velocities
        time_length = transformed_traj[-1]['timestamp'] - transformed_traj[0]['timestamp']
        final_pose = transformed_traj[-1]
        vx = final_pose['x'] / time_length
        vy = final_pose['y'] / time_length
        vyaw = final_pose['yaw'] / time_length
        
        velocities.append([vx, vy, vyaw])
        
        if plot:
            plot_trajectories(traj, transformed_traj)
    
    return np.array(velocities)

def plot_trajectories(traj, transformed_traj):
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot original trajectory
    x_orig = [pose['x'] for pose in traj]
    y_orig = [pose['y'] for pose in traj]
    yaw_orig = [pose['yaw'] for pose in traj]

    ax1.plot(x_orig, y_orig, 'b.-', label='Original Path')
    # Add first point marker
    ax1.plot(x_orig[0], y_orig[0], 'go', markersize=10, label='Start Point')
    # Plot yaw arrows every 5 points
    for i in range(0, len(traj), 5):
        arrow_len = 0.005  # Adjust this value based on your scale
        dx = arrow_len * np.cos(yaw_orig[i])
        dy = arrow_len * np.sin(yaw_orig[i])
        ax1.arrow(x_orig[i], y_orig[i], dx, dy, head_width=0.003, head_length=0.001, fc='r', ec='r')
    ax1.set_title('Original Trajectory')
    ax1.set_aspect('equal')
    ax1.grid(True)

    # Plot transformed trajectory
    x_trans = [pose['x'] for pose in transformed_traj]
    y_trans = [pose['y'] for pose in transformed_traj]
    yaw_trans = [pose['yaw'] for pose in transformed_traj]

    ax2.plot(x_trans, y_trans, 'b.-', label='Transformed Path')
    # Add first point marker
    ax2.plot(x_trans[0], y_trans[0], 'go', markersize=10, label='Start Point')
    # Plot yaw arrows every 5 points
    for i in range(0, len(transformed_traj), 5):
        arrow_len = 0.005  # Adjust this value based on your scale
        dx = arrow_len * np.cos(yaw_trans[i])
        dy = arrow_len * np.sin(yaw_trans[i])
        ax2.arrow(x_trans[i], y_trans[i], dx, dy, head_width=0.003, head_length=0.001, fc='r', ec='r')
    ax2.set_title('Transformed Trajectory')
    ax2.set_aspect('equal')
    ax2.grid(True)

    # Add legends to both plots
    ax1.legend()
    ax2.legend()

    plt.tight_layout()
    plt.show()

class VelocityMatcher:
    def __init__(self, filename, k=1, weights=[1.0, 1.0, 1.0]):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            data_input = np.array(data['data_input'])
            data_result = data['data_result']

        self.data_input = data_input
        self.velocities = calculate_velocities(data_input, data_result)
        self.weights = np.array(weights)
        self.k = k
        
        # Initialize nearest neighbor model with weighted metric
        self.nn_model = NearestNeighbors(n_neighbors=k, 
                                        metric='pyfunc',
                                        metric_params={'func': lambda x, y: weighted_euclidean(x, y, self.weights)})
        self.nn_model.fit(self.velocities)
    
    def find_nearest_input(self, target_vx, target_vy, target_vyaw):
        """
        Find interpolated input parameters for desired velocities using K nearest neighbors
        
        Args:
            target_vx: desired x velocity
            target_vy: desired y velocity
            target_vyaw: desired yaw velocity
            
        Returns:
            tuple: (interpolated_input_parameters, actual_velocities)
        """
        target = np.array([[target_vx, target_vy, target_vyaw]])
        distances, indices = self.nn_model.kneighbors(target)
        
        # Convert distances to weights using inverse distance weighting
        weights = 1 / (distances + 1e-6)  # Add small epsilon to avoid division by zero
        weights = weights / np.sum(weights)  # Normalize weights
        
        # Interpolate input parameters
        interpolated_input = np.average(self.data_input[indices[0]], weights=weights[0], axis=0)
        
        # Calculate the weighted average of actual velocities
        interpolated_velocities = np.average(self.velocities[indices[0]], weights=weights[0], axis=0)
        
        return interpolated_input, interpolated_velocities

# Add this function outside the class
def weighted_euclidean(x, y, w):
    """Custom weighted euclidean distance metric"""
    return np.sqrt(np.sum(w * (x - y) ** 2))

# Example usage:
if __name__ == "__main__":
    weights = [1.0, 1.0, 1.0]
    matcher = VelocityMatcher('robot_movement_data.pkl', k=1, weights=weights)
    
    # Example: Find nearest neighbor for desired velocity
    target_vx, target_vy, target_vyaw = 0, 0.15, 0
    matched_input, actual_velocities = matcher.find_nearest_input(target_vx, target_vy, target_vyaw)
    
    print(f"Target velocities [vx, vy, vyaw]: [{target_vx}, {target_vy}, {target_vyaw}]")
    print(f"Nearest neighbor velocities: {actual_velocities}")
    print(f"Corresponding input parameters: {matched_input}")
    
    velocities = matcher.velocities
    # Plot vx vs vy with orientation arrows
    plt.figure(figsize=(8, 8))
    arrow_length = 0.05  # Adjust this value to change arrow size
    plt.quiver(velocities[:, 0], velocities[:, 1],  # positions
               arrow_length * np.cos(velocities[:, 2]), arrow_length * np.sin(velocities[:, 2]),  # directions
               scale=1, scale_units='xy', angles='xy',  # scale arrows relative to plot units
               color='red', alpha=0.5, width=0.005)  # styling
    plt.xlabel('Velocity X (m/s)')
    plt.ylabel('Velocity Y (m/s)')
    plt.title('Velocity Distribution (X vs Y) with Yaw Orientation')
    plt.grid(True)
    plt.axis('equal')
    plt.show()
