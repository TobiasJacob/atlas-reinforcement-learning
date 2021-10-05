import os
import numpy as np

import gym
from atlasrl.robots.AtlasBulletEnv import AtlasBulletEnv
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import datetime

log_dir = f"runs/{datetime.datetime.now()}"
os.makedirs(log_dir, exist_ok=True)
env = AtlasBulletEnv(render=True)

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
model.learn(total_timesteps=400000)
model.save(f"ModelTrained.torch")


obs = env.reset()
while True:
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=False)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()

env.close()
