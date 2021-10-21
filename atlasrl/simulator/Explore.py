from time import sleep

import numpy as np
import quaternion
from atlasrl.motions.MotionReader import MotionReader
from atlasrl.robots.AtlasBulletEnv import AtlasBulletEnv
import pybullet as p

env = AtlasBulletEnv(render=True)
env.reset()
time = 0
lastPosErr = np.zeros(3)
lastTorqueErr = np.zeros(3)

action_selector_ids = []
for i in range(env._p.getNumJoints(env.atlas)):
    action_selector_id = env._p.addUserDebugParameter(paramName=str(env._p.getJointInfo(env.atlas, i)[1]),
                                                 rangeMin=-1,
                                                 rangeMax=1,
                                                 startValue=0)
    action_selector_ids.append(action_selector_id)
    print(env._p.getDynamicsInfo(env.atlas, i))

while time < 100:
    action = np.zeros(30)
    for i in range(env._p.getNumJoints(env.atlas)):
        action[i] = env._p.readUserDebugParameter(action_selector_ids[i])

    obs, reward, done, info = env.step(action)

    # env._p.resetBasePositionAndOrientation(env.atlas, np.array([0, 0, 1]), np.array([0, 0, 0, 1]))
    # env._p.createConstraint(env.atlas, -1, -1, -1, env._p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], [0, 0, 1])

    env.render("human")
    time += env.timeDelta
    sleep(env.timeDelta)
