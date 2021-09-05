import gym

from stable_baselines3 import PPO
from atlasrl.robots.AtlasBulletEnv import AtlasBulletEnv

env = AtlasBulletEnv(render=True)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

for epoch in range(20):
    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()

env.close()
