import gymnasium as gym 

env = gym.make('Humanoid-v4')
observation, info = env.reset()

for step in range(100):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    print(f"Step {step}, Reward: {reward}")
    
    if terminated or truncated:
        break
env.close()