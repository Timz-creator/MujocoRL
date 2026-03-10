import gymnasium as gym
import numpy as np
import mujoco 

class RoverEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self.model = mujoco.MjModel.from_xml_path("models/rover.xml")

        self.data = mujoco.MjData(self.model)
        
        
    






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

