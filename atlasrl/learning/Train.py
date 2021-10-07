import os
import numpy as np

import gym
from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.policies import ActorCriticPolicy, get_policy_from_name
from atlasrl.robots.AtlasBulletEnv import AtlasBulletEnv
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import datetime
from time import sleep

from atlasrl.robots.AtlasBulletVecEnv import AtlasBulletVecEnv
from stable_baselines3.common.vec_env import SubprocVecEnv

def getBullentEnv(index: int):
    return lambda: AtlasBulletEnv(render=index == 0)

if __name__ == "__main__":
    log_dir = f"runs/{datetime.datetime.now()}"
    os.makedirs(log_dir, exist_ok=True)
    env = SubprocVecEnv([getBullentEnv(i) for i in range(16)]) 
    # env = AtlasBulletEnv(render=True)

    # TODO: Run with use_sde=False, policy_kwargs={"log_std_init": -2.5}, 
    model = PPO("MlpPolicy", env, learning_rate=1e-3, n_epochs=4, n_steps=256, verbose=1, tensorboard_log=log_dir)
    # model.load(f"ModelTrained.torch")
    # model.policy.log_std.requires_grad = False # Prevent training of std
    model.learn(total_timesteps=800000)
    model.save(f"ModelTrained.torch")


    obs = env.reset()
    while True:
        for i in range(1000):
            action, _states = model.predict(obs, deterministic=False)
            obs, reward, done, info = env.step(action)
            env.render()
            if done:
                obs = env.reset()
                break
            sleep(env.timeDelta)
        sleep(1)

    env.close()
