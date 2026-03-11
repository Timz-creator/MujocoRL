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

