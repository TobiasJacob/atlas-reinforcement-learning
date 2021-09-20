import pytest
from atlasrl.robots.AtlasBulletEnv import AtlasBulletEnv
import numpy as np
from time import sleep

@pytest.mark.skip(reason="Requires GUI")
def test_AtlasBulletEnv():
    env = AtlasBulletEnv(render=True)

    for episode in range(10): 
        obs = env.reset()
        for step in range(5000):
            action = env.action_space.sample()
            # for _ in np.arange(0, 1, env.timeDelta):
            obs, reward, done, info = env.step(action)
            env.render("human")
            sleep(env.timeDelta)
    env.close()

if __name__ == "__main__":
	test_AtlasBulletEnv()
