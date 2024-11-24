import numpy as np
import math


class JointTrajectory:
    def __init__(self):
        self.times = []
        self.positions = []  # List of position vectors
        self._size = None    # Size of position vectors, set on first waypoint
    
    def add_waypoint(self, t, position):
        """Add a waypoint at time t with position vector"""
        # Validate position vector
        position = np.array(position)
        if self._size is None:
            self._size = len(position)
        elif len(position) != self._size:
            raise ValueError(f"Position vector must have size {self._size}")
            
        # Find where to insert the new waypoint
        insert_idx = np.searchsorted(self.times, t)
        
        # Remove any waypoints that occur after this time
        self.times = self.times[:insert_idx]
        self.positions = self.positions[:insert_idx]
        
        # Add the new waypoint
        self.times.append(t)
        self.positions.append(position)
    
    def get_state(self, t):
        """
        Get interpolated state at time t using simple linear interpolation
        between the two nearest waypoints
        """
        if len(self.times) < 2:
            raise ValueError("Need at least 2 waypoints")
            
        # If beyond last waypoint, return last position with zero velocity
        if t >= self.times[-1]:
            return {
                'position': self.positions[-1],
                'velocity': np.zeros(self._size)
            }
            
        # Find the two waypoints we're between
        next_idx = np.searchsorted(self.times, t)
        prev_idx = next_idx - 1
        
        # Get time segment and do linear interpolation
        t0, t1 = self.times[prev_idx], self.times[next_idx]
        alpha = (t - t0) / (t1 - t0)  # Interpolation factor (0 to 1)
        
        # Interpolate positions
        pos0, pos1 = self.positions[prev_idx], self.positions[next_idx]
        position = pos0 + alpha * (pos1 - pos0)
        
        # Calculate velocities (constant between waypoints)
        dt = t1 - t0
        velocity = (pos1 - pos0) / dt
        
        return {
            'position': position,
            'velocity': velocity
        }


class BaseTrajectory:
    def __init__(self):
        self.times = []
        self.x_points = []
        self.y_points = []
        self.yaw_points = []
    
    def add_waypoint(self, t, x, y, yaw):
        """Add a waypoint at time t with position (x, y) and yaw angle"""
        # Find where to insert the new waypoint
        insert_idx = np.searchsorted(self.times, t)
        
        # Remove any waypoints that occur after this time
        self.times = self.times[:insert_idx]
        self.x_points = self.x_points[:insert_idx]
        self.y_points = self.y_points[:insert_idx]
        self.yaw_points = self.yaw_points[:insert_idx]
        
        # Add the new waypoint
        self.times.append(t)
        self.x_points.append(x)
        self.y_points.append(y)
        self.yaw_points.append(yaw)
    
    def get_state(self, t):
        """
        Get interpolated state at time t using simple linear interpolation
        between the two nearest waypoints
        """
        if len(self.times) < 2:
            raise ValueError("Need at least 2 waypoints")
            
        # If beyond last waypoint, return last position with zero velocity
        if t >= self.times[-1]:
            return {
                'x': self.x_points[-1],
                'y': self.y_points[-1],
                'yaw': self.yaw_points[-1],
                'vx': 0.0,
                'vy': 0.0,
                'vyaw': 0.0
            }
            
        # Find the two waypoints we're between
        next_idx = np.searchsorted(self.times, t)
        prev_idx = next_idx - 1
        
        # Get time segment and do linear interpolation
        t0, t1 = self.times[prev_idx], self.times[next_idx]
        alpha = (t - t0) / (t1 - t0)  # Interpolation factor (0 to 1)
        
        # Interpolate positions
        x = self.x_points[prev_idx] + alpha * (self.x_points[next_idx] - self.x_points[prev_idx])
        y = self.y_points[prev_idx] + alpha * (self.y_points[next_idx] - self.y_points[prev_idx])
        
        # Calculate shortest angular distance for yaw
        diff_yaw = self.yaw_points[next_idx] - self.yaw_points[prev_idx]
        diff_yaw = math.atan2(math.sin(diff_yaw), math.cos(diff_yaw))  # Normalize to [-pi, pi]
        yaw = self.yaw_points[prev_idx] + alpha * diff_yaw
        
        # Calculate velocities (constant between waypoints)
        dt = t1 - t0
        vx = (self.x_points[next_idx] - self.x_points[prev_idx]) / dt
        vy = (self.y_points[next_idx] - self.y_points[prev_idx]) / dt
        vyaw = (self.yaw_points[next_idx] - self.yaw_points[prev_idx]) / dt
        
        return {
            'x': x,
            'y': y,
            'yaw': yaw,
            'vx': vx,
            'vy': vy,
            'vyaw': vyaw
        }