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
from stable_baselines3.common.vec_env import VecCheckNan

def getBullentEnv(index: int):
    return lambda: AtlasBulletEnv(render=index == 0)

if __name__ == "__main__":
    # env = AtlasBulletEnv(render=True)
    if True: # Set True for training
        log_dir = f"runs/{datetime.datetime.now()}"
        os.makedirs(log_dir, exist_ok=True)
        env = SubprocVecEnv([getBullentEnv(i) for i in range(16)]) 
        env = VecCheckNan(env, raise_exception=True)
        if False: # Use a pre-trained model
            model = PPO.load("runs/2021-10-09 12:02:23.349342/ModelTrained100M.torch")
            model.env = env
            model.learn(total_timesteps=1000000)
            model.save(f"{log_dir}/ModelTrained1M.torch")
        for i in range(0, 100):
            # TODO: Run with use_sde=False, policy_kwargs={"log_std_init": -2.5}, 
            if i == 0:
                model = PPO(
                    "MlpPolicy",
                    env,
                    learning_rate=3e-4,
                    n_epochs=8,
                    n_steps=2048,
                    batch_size=512,
                    gae_lambda=0.95,
                    gamma=0.95,
                    verbose=1,
                    use_sde=False,
                    policy_kwargs={"log_std_init": -2.5},
                    tensorboard_log=log_dir
                )
            # model.policy.log_std.requires_grad = False # Prevent training of std
            model.learn(total_timesteps=1000000, reset_num_timesteps=False)
            model.save(f"{log_dir}/ModelTrained{i + 1}M.torch")

    model = PPO.load(f"runs/2021-10-09 12:02:23.349342/ModelTrained30M.torch")
    env = getBullentEnv(0)()

    obs = env.reset()
    while True:
        for i in range(1000):
            action, _states = model.predict(obs, deterministic=False)
            obs, reward, done, info = env.step(action)
            if done:
                obs = env.reset()
                break
            sleep(env.timeDelta)
        sleep(1)

    env.close()
