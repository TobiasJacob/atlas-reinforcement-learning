from time import sleep

import numpy as np
import quaternion
from atlasrl.motions.MotionReader import MotionReader
from atlasrl.robots.AtlasBulletEnv import AtlasBulletEnv
import pybullet as p

env = AtlasBulletEnv(render=True)
env.reset()
clip = MotionReader.readClip()
time = 0
lastPosErr = np.zeros(3)
lastTorqueErr = np.zeros(3)
while time < 100:
    motionState = clip.getState(time / 1.0)
    action = motionState.getAction()
    obs, reward, done, info = env.step(action)

    pos, orn = env._p.getBasePositionAndOrientation(env.atlas)
    targetPos, targetOrn = motionState.rootPosition, motionState.rootRotation
    targetOrnAsArray = quaternion.as_float_array(targetOrn)
    env._p.resetBasePositionAndOrientation(env.atlas, targetPos + np.array([0, 0, 1]), [*targetOrnAsArray[1:4], targetOrnAsArray[0]])

    env.render("human")
    time += env.timeDelta
    sleep(env.timeDelta)
