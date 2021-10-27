from datetime import datetime
import numpy as np
import quaternion
from stable_baselines3 import PPO
from time import sleep
from atlasrl.robots.AtlasBulletEnv import AtlasBulletEnv

from atlasrl.robots.AtlasRemoteEnv import AtlasRemoteEnv


if __name__ == "__main__":
    # env = AtlasBulletEnv(render=False)
    envDC = AtlasBulletEnv(render=True)
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
    model = PPO.load(f"runs/reward2-2021-10-23 14:29:42.544031/ModelTrained33M.torch")

    obs = env.reset()
    obsDC = envDC.reset(randomStartPosition=False)
    print(obs, obsDC)
    while True:
        for i in range(1000):
            action, _states = model.predict(obs, deterministic=False)
            obs, reward, done, info = env.step(action)
            # obsDC, _, _, _ = envDC.step(action)
            Z = obs[0]
            orn = quaternion.as_float_array(info["orn"])
            jointAngles = obs[16:46]
            jointSpeeds = obs[46:76]
            envDC._p.resetBasePositionAndOrientation(envDC.atlas, np.array([0, 0, Z]), np.array([*orn[1:4], orn[0]]))
            for i in range(30):
                envDC._p.resetJointState(envDC.atlas, i, jointAngles[i])
            print(orn)
            if done:
                obs = env.reset()
                break
            sleep(1/30)

    env.close()
