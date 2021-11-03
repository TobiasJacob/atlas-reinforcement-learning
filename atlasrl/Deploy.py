from datetime import datetime
import numpy as np
import quaternion
from stable_baselines3 import PPO
from time import sleep
from atlasrl.robots.AtlasBulletEnv import AtlasBulletEnv

from atlasrl.robots.AtlasRemoteEnv import AtlasRemoteEnv
from atlasrl.robots.Constants import parameterNames


if __name__ == "__main__":
    # env = AtlasBulletEnv(render=True)
    envDC = AtlasBulletEnv(render=True, simStepsPerControlStep=1000)
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
    model = PPO.load(f"runs/littleRandomization-2021-11-02 01:18:17.536564/ModelTrained24M.torch")

    obs = env.reset()
    obsDC = envDC.reset(randomStartPosition=False)
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=False)
        # action[parameterNames.index("back_bky")] += -0.16
        # action[parameterNames.index("back_bkx")] += 0.08
        obs, reward, done, info = env.step(action)
        # Parse obs
        Z = obs[0]
        orn = quaternion.as_float_array(info["orn"])
        jointAngles = obs[16:46]
        jointSpeeds = obs[46:76]
        baseSpeed, baseOrn = obs[7:10], obs[13:16]
        # Sync DCEnv
        if i < 1000:
            envDC._p.resetBasePositionAndOrientation(envDC.atlas, np.array([0, 0, Z + 0.03]), np.array([*orn[1:4], orn[0]]))
            envDC._p.resetBaseVelocity(envDC.atlas, baseSpeed, baseOrn)
            for i in range(30):
                envDC._p.resetJointState(envDC.atlas, i, jointAngles[i], jointSpeeds[i])
        obsDC, _, _, _ = envDC.step(action)
        # obsDC = envDC.getObservation()[0]
        # envDC.time = env.time
        # Compare observation
        print(np.abs(obs - obsDC).max())
        if np.abs(obs - obsDC).max() > 1e-0:
            print(np.abs(obs - obsDC) > 1e-0)
            names = ["Z",
                    "vecX", "vecX", "vecX",
                    "vecY", "vecY", "vecY",
                    "posSpeed", "posSpeed", "posSpeed",
                    "desiredBaseSpeed", "desiredBaseSpeed", "desiredBaseSpeed",
                    "ornSpeed", "ornSpeed", "ornSpeed"
                    ] + ["jointAngles"] * 30 + ["jointSpeeds"] * 30 + ["desiredAngles"] * 30 + ["desiredJointSpeeds"] * 30
            for j in range(len(obs)):
                print(j, names[j], np.abs(obs[j] - obsDC[j]) > 1e-0, obs[j], obsDC[j])
            print(env.time, envDC.time)
            sleep(1)

    env.close()
