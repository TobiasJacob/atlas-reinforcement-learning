from time import sleep
import numpy as np
import pytest
from atlasrl.robots.AtlasRemoteEnv import AtlasRemoteEnv, parameterNames

@pytest.mark.skip(reason="Requires ihmc-open-robotics-software running")
def test_AtlasBulletEnv():
	env = AtlasRemoteEnv()

	(initAngles, _) = env.reset()
	centerOfMassFrame = None
	for step in range(5000):
		action = env.action_space.sample()
		delta = np.zeros_like(initAngles)
		if centerOfMassFrame is not None:
			gain = 0.000
			delta[parameterNames.index("l_leg_aky")] = centerOfMassFrame[0] * gain
			delta[parameterNames.index("r_leg_aky")] = centerOfMassFrame[0] * gain
			print(centerOfMassFrame[0] * gain)
		else:
			print("None")
		((_, centerOfMassFrame), _, _, _) = env.step(initAngles + delta)
		env.render("human")
		sleep(0.1)

if __name__ == "__main__":
	test_AtlasBulletEnv()
