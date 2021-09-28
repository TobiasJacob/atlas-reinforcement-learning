from time import sleep

import numpy as np
import quaternion
from atlasrl.motions.MotionReader import MotionReader
from atlasrl.robots.AtlasBulletEnv import AtlasBulletEnv
import pybullet as p

env = AtlasBulletEnv(render=True)
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

    # force = np.array([time, 0, 1]) - pos
    # force += (force - lastPosErr) * 2
    # force *= 100
    # # p.applyExternalForce(objectUniqueId=env.atlas, linkIndex=-1, forceObj=force, posObj=targetPos, flags=p.WORLD_FRAME)
    # lastPosErr = (targetPos + np.array([0, 0, 1]) - pos)
    # torque = quaternion.as_rotation_vector(targetOrn.inverse()) - quaternion.as_rotation_vector(quaternion.from_float_array(orn[3:4] + orn[:3]))
    # torque += (torque - lastTorqueErr) * 1.0
    # torque *= 10000
    # p.applyExternalTorque(objectUniqueId=env.atlas, linkIndex=-1, torqueObj=torque, flags=p.WORLD_FRAME)
    # lastTorqueErr = quaternion.as_rotation_vector(targetOrn) - quaternion.as_rotation_vector(quaternion.from_float_array(orn[3:4] + orn[:3]))

    env.render("human")
    time += env.timeDelta
    sleep(env.timeDelta)
