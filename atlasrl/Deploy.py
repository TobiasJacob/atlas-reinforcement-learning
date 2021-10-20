from stable_baselines3 import PPO
from time import sleep

from atlasrl.robots.AtlasRemoteEnv import AtlasRemoteEnv


if __name__ == "__main__":
    # env = AtlasBulletEnv(render=True)
    model = PPO.load(f"runs/reward2-2021-10-18 11:58:03.398274/ModelTrained115M.torch")
    env = AtlasRemoteEnv()

    obs = env.reset()
    while True:
        for i in range(1000):
            action, _states = model.predict(obs, deterministic=False)
            obs, reward, done, info = env.step(action)
            if done:
                obs = env.reset()
                break
            sleep(env.timeDelta * 30)
        sleep(1)

    env.close()
