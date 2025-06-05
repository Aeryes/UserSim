from stable_baselines3 import PPO
from src.app.agent.browser_gym_env import BrowserGymEnv

env = BrowserGymEnv(use_llm=True)
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)
model.save("ppo_browser_agent")
