import gymnasium as gym
import numpy as np
import mujoco
from gymnasium import spaces


class RoverNavigationEnv(gym.Env):
    """
    Custom Gymnasium environment for a differential-drive rover navigating an obstacle course.
    Goal: Reach position (8, 0) while avoiding obstacles.
    """
    
    def __init__(self, render_mode=None):
        super().__init__()
        
        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path('./models/rover.xml')
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        
        # Environment parameters
        self.goal_pos = np.array([8.0, 0.0])
        self.max_steps = 500
        self.step_count = 0
        
        # Action space: [left_motor, right_motor] in range [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        # Observation space: [chassis_x, chassis_y, chassis_z, 
        #                     chassis_vx, chassis_vy, chassis_vz,
        #                     quat_x, quat_y, quat_z, quat_w,
        #                     angvel_x, angvel_y, angvel_z,
        #                     left_joint_pos, right_joint_pos,
        #                     left_joint_vel, right_joint_vel,
        #                     goal_rel_x, goal_rel_y, goal_distance]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32
        )
        
    def _get_observation(self):
        """Extract observation from MuJoCo data"""
        # Get chassis position
        chassis_pos = self.data.body('chassis').xpos.copy()
        
        # Get chassis velocity
        chassis_linvel = self.data.body('chassis').cvel[:3].copy()
        
        # Get chassis orientation (quaternion)
        chassis_quat = self.data.body('chassis').xquat.copy()
        
        # Get chassis angular velocity
        chassis_angvel = self.data.body('chassis').cvel[3:].copy()
        
        # Get joint positions and velocities
        left_joint_pos = self.data.joint('left_motor').qpos[0]
        right_joint_pos = self.data.joint('right_motor').qpos[0]
        left_joint_vel = self.data.joint('left_motor').qvel[0]
        right_joint_vel = self.data.joint('right_motor').qvel[0]
        
        # Calculate goal-relative position
        goal_rel = self.goal_pos - chassis_pos[:2]
        goal_distance = np.linalg.norm(goal_rel)
        
        # Assemble observation
        obs = np.concatenate([
            chassis_pos,                           # 3
            chassis_linvel,                        # 3
            chassis_quat,                          # 4
            chassis_angvel,                        # 3
            [left_joint_pos, right_joint_pos],    # 2
            [left_joint_vel, right_joint_vel],    # 2
            goal_rel,                              # 2
            [goal_distance]                        # 1
        ]).astype(np.float32)
        
        return obs
    
    def _get_reward(self):
        """Calculate reward based on progress toward goal"""
        chassis_pos = self.data.body('chassis').xpos[:2].copy()
        
        # Distance to goal
        distance_to_goal = np.linalg.norm(self.goal_pos - chassis_pos)
        
        # Reward components
        # 1. Progress toward goal (negative distance, scaled)
        progress_reward = -distance_to_goal * 0.1
        
        # 2. Goal reached bonus
        goal_bonus = 0
        if distance_to_goal < 0.5:
            goal_bonus = 10.0
        
        # 3. Penalty for being out of bounds
        if np.abs(chassis_pos[0]) > 15 or np.abs(chassis_pos[1]) > 10:
            out_of_bounds_penalty = -5.0
        else:
            out_of_bounds_penalty = 0
        
        # 4. Small penalty per step (encourage efficiency)
        step_penalty = -0.01
        
        total_reward = progress_reward + goal_bonus + out_of_bounds_penalty + step_penalty
        return total_reward
    
    def reset(self, seed=None, options=None):
        """Reset environment"""
        super().reset(seed=seed)
        
        # Reset MuJoCo data
        mujoco.mj_resetData(self.model, self.data)
        
        # Set initial chassis position
        self.data.body('chassis').xpos[:] = [0, 0, 0.15]
        self.data.body('chassis').xvel[:] = 0
        
        # Reset step counter
        self.step_count = 0
        
        # Forward simulation to settle
        mujoco.mj_forward(self.model, self.data)
        
        obs = self._get_observation()
        info = {}
        
        return obs, info
    
    def step(self, action):
        """Execute one step"""
        # Apply action to motors
        self.data.ctrl[0] = action[0]  # left motor
        self.data.ctrl[1] = action[1]  # right motor
        
        # Step simulation
        mujoco.mj_step(self.model, self.data)
        
        # Get observation and reward
        obs = self._get_observation()
        reward = self._get_reward()
        
        # Check termination conditions
        self.step_count += 1
        terminated = False
        truncated = self.step_count >= self.max_steps
        
        # Check if fallen over (chassis z < 0.05 means flipped)
        if self.data.body('chassis').xpos[2] < 0.05:
            terminated = True
            reward -= 5.0
        
        # Check if out of bounds
        if np.abs(self.data.body('chassis').xpos[0]) > 15 or np.abs(self.data.body('chassis').xpos[1]) > 10:
            terminated = True
        
        # Check if reached goal
        chassis_pos = self.data.body('chassis').xpos[:2]
        if np.linalg.norm(self.goal_pos - chassis_pos) < 0.5:
            terminated = True
        
        info = {}
        
        return obs, reward, terminated, truncated, info
    
    def render(self):
        """Render environment (optional)"""
        if self.render_mode == "human":
            # You would need mujoco-python-viewer for this
            pass
    
    def close(self):
        """Clean up"""
        pass


if __name__ == "__main__":
    # Quick test
    env = RoverNavigationEnv()
    obs, info = env.reset()
    
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Run a few random steps
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {step}: Reward = {reward:.3f}, Distance to goal = {obs[-1]:.3f}")
        
        if terminated or truncated:
            break
    
    env.close()
    print("Test complete!")