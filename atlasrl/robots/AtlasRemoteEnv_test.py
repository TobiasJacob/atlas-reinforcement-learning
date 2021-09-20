import pytest
from atlasrl.robots.AtlasRemoteEnv import AtlasRemoteEnv

@pytest.mark.skip(reason="Requires ihmc-open-robotics-software running")
def test_AtlasBulletEnv():
	env = AtlasRemoteEnv()

	env.reset()
	for step in range(5000):
		action = env.action_space.sample()
		obs, reward, done, info = env.step(action)
		env.render("human")

if __name__ == "__main__":
	test_AtlasBulletEnv()
