import gymnasium as gym
import numpy as np
import mujoco 
from gymnasium import spaces

class RoverEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self.model = mujoco.MjModel.from_xml_path("models/rover.xml")

        self.data = mujoco.MjData(self.model)

        self.goal_pos = np.array([8.0, 0.0])
        self.max_steps = 500
        self.step_count = 0

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        self.observation_space = spaces.Box(low= -np.inf, high=np.inf, shape=(20,), dtype=np.float32)

    def reset(self):
        """Reset the environment to a new state"""
        mujoco.mj_resetData(self.model, self.data)
        self.data.body('chassis').xpos[:] = [0, 0, 0.15]
        self.step_count = 0
        obs = self._get_observation()
        info = {}
        return obs, info 
    
    def step(self, action):
        """ Execute one timestep within the environment"""
        self.data.ctrl[0] = action[0]
        self.data.ctrl[1] = action[1]

        mujoco.mj_step(self.model, self.data)
        self.step_count += 1
        obs = self._get_observation()
        reward = self._get_reward()
        self.step_count += 1

        terminated = False
        if self.data.body('chassis').xpos[2] < 0.05:
            terminated = True
            reward -= 5 

        distance_to_goal = np.linalg.norm(self.goal_pos - self.data.body('chassis').xpos[:2])
        if distance_to_goal < 0.5:
            terminated = True 

        if np.abs(self.data.body('chassis').xpos[0]) > 15 or np.abs(self.data.body('chassis').xpos[1]) > 10:
            terminated = True

        truncated = self.step_count >= self.max_steps

        info = {}

        return obs, reward, terminated, truncated, info
        
    def _get_observation(self):
        """ Extract observation from Mujoco data """
        chassis_pos = self.data.body('chassis').xpos.copy()

        chassis_linvel = self.data.body('chassis').cvel[:3].copy()

        chassis_quat = self.data.body('chassis').xquat.copy()

        chassis_angvel = self.data.body('chassis').cvel[3:].copy()

        left_joint_pos = self.data.joint('left_motor').qpos[0]
        right_joint_pos = self.data.joint('right_motor').qpos[0]
        left_joint_vel = self.data.joint('left_motor').qvel[0]
        right_joint_vel = self.data.joint('right_motor').qvel[0]

        goal_rel = self.goal_pos - chassis_pos[:2]
        goal_distance = np.linalg.norm(goal_rel)

        obs = np.concatenate([
            chassis_pos,
            chassis_linvel,
            chassis_quat,
            chassis_angvel,
            [left_joint_pos, right_joint_pos],
            [left_joint_vel, right_joint_vel],
            goal_rel,
            [goal_distance]
        ]).astype(np.float32)

        return obs 

        








env = gym.make("CartPole-v1", render_mode="human")

observation, info = env.reset()

print(f"Starting observation: {observation}")

episode_over = False
total_reward = 0

while not episode_over:

    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    total_reward += reward 
    episoder_over = terminated or truncated 

print(f"Epsiode finished! Total reward: {total_reward }")
env.close()

