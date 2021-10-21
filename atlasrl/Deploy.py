from datetime import datetime
from stable_baselines3 import PPO
from time import sleep

from atlasrl.robots.AtlasRemoteEnv import AtlasRemoteEnv


if __name__ == "__main__":
    # env = AtlasBulletEnv(render=True)
    env = AtlasRemoteEnv()
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=5e-5,
        n_epochs=4,
        n_steps=2048,
        batch_size=512,
        gae_lambda=0.95,
        gamma=0.95,
        verbose=1,
        use_sde=False,
        policy_kwargs={"log_std_init": -2.5},
        tensorboard_log="runs/" + str(datetime.now())
    )
    # model = PPO.load(f"runs/reward2-2021-10-18 11:58:03.398274/ModelTrained115M.torch")

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
