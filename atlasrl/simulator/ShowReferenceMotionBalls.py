"""This is a script to show the reference motion markers."""
from typing import List
import numpy as np
import pybullet as p
import pybullet_data
import time,math

import quaternion
from atlasrl.motions.MotionReader import MotionReader
from itertools import cycle

clip = MotionReader.readClip()
path = "data/atlas/atlas_v4_with_multisense.urdf"

fps = 60
dt = 1. / fps
p.connect(p.GUI, options="--width=1280 --height=720 --mp4=\"test.mp4\" --mp4fps=60")
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF("plane.urdf",[0,0,0], useFixedBase=True)
atlas = p.loadURDF(path, [0, 0, 0.95])

markers = []
for _ in range(30):
    virtualShapeId = p.createVisualShape(shapeType=p.GEOM_BOX,
                                            halfExtents=[0.025, 0.025, 0.025],
                                            rgbaColor=[1, 0, 0, 1])
    bodyId = p.createMultiBody(baseMass=0,
                            baseCollisionShapeIndex=-1,
                            baseVisualShapeIndex=virtualShapeId,
                            basePosition=[0,0,0],
                            useMaximalCoordinates=True)                            
    markers.append(bodyId)

action = np.zeros(3)
action_selector_ids = []
for i in range(len(action)):
    action_selector_id = p.addUserDebugParameter(paramName=str(i), rangeMin=-1, rangeMax=1, startValue=0)
    action_selector_ids.append(action_selector_id)

def r2p(q: quaternion.quaternion):
    q = quaternion.as_float_array(q).tolist()
    return q[1:4] + q[0:1]

def r2ppos(pos: np.array):
    return [pos[0], -pos[2], pos[1]]

t = 0
while True:
    for j in range(len(action)):
        action[j] = p.readUserDebugParameter(action_selector_ids[j])

    frame = clip.getState(t)

    cycleI = i // len(clip.frames)
    basePos, baseOrn = [0, 0, 0], r2p(quaternion.from_rotation_vector([np.pi / 2, 0, 0]))
    rootPosOffset = np.array(clip.frames[-1].rootPosition) * cycleI
    cycleRootPos = [frame.rootPosition[0] + rootPosOffset[0], frame.rootPosition[1], frame.rootPosition[2] + rootPosOffset[2]]
    rootPos, rootOrn = p.multiplyTransforms(basePos, baseOrn, cycleRootPos, r2p(frame.rootRotation))
    chestPos, chestOrn = p.multiplyTransforms(rootPos, rootOrn, [0.0, 0.2, 0], r2p(frame.chestRotation))
    neckPos, neckOrn = p.multiplyTransforms(chestPos, chestOrn, [0, 0.2, 0], r2p(frame.neckRotation))

    rightHipPos, rightHipOrn = p.multiplyTransforms(rootPos, rootOrn, [0, -.1, .1], r2p(frame.rightHipRotation))
    rightKneePos, rightKneeOrn = p.multiplyTransforms(rightHipPos, rightHipOrn, [0, -.4, 0], r2p(quaternion.from_rotation_vector([0, 0, frame.rightKneeRotation])))
    rightAnklePos, rightAnkleOrn = p.multiplyTransforms(rightKneePos, rightKneeOrn, [0, -.4, 0], r2p(frame.rightAnkleRotation))

    leftHipPos, leftHipOrn = p.multiplyTransforms(rootPos, rootOrn, [0, -.1, -.1], r2p(frame.leftHipRotation))
    leftKneePos, leftKneeOrn = p.multiplyTransforms(leftHipPos, leftHipOrn, [0, -.4, 0], r2p(quaternion.from_rotation_vector([0, 0, frame.leftKneeRotation])))
    leftAnklePos, leftAnkleOrn = p.multiplyTransforms(leftKneePos, leftKneeOrn, [0, -.4, 0], r2p(frame.leftAnkleRotation))

    rightShoulderPos, rightShoulderOrn = p.multiplyTransforms(neckPos, neckOrn, [0, 0, 0.2], r2p(frame.rightShoulderRotation))
    rightElbowPos, rightElbowOrn = p.multiplyTransforms(rightShoulderPos, rightShoulderOrn, [0, -0.3, 0], r2p(quaternion.from_rotation_vector([0, 0, frame.rightElbowRotation])))
    rightHandPos, rightHandOrn = p.multiplyTransforms(rightElbowPos, rightElbowOrn, [0, -0.3, 0], [1, 0, 0, 0])

    leftShoulderPos, leftShoulderOrn = p.multiplyTransforms(neckPos, neckOrn, [0, 0, -0.2], r2p(frame.leftShoulderRotation))
    leftElbowPos, leftElbowOrn = p.multiplyTransforms(leftShoulderPos, leftShoulderOrn, [0, -0.3, 0], r2p(quaternion.from_rotation_vector([0, 0, frame.leftElbowRotation])))
    leftHandPos, leftHandOrn = p.multiplyTransforms(leftElbowPos, leftElbowOrn, [0, -0.3, 0], [1, 0, 0, 0])

    p.resetBasePositionAndOrientation(markers[0], rootPos, rootOrn)
    p.resetBasePositionAndOrientation(markers[1], chestPos, chestOrn)
    p.resetBasePositionAndOrientation(markers[2], neckPos, neckOrn)
    p.resetBasePositionAndOrientation(markers[3], rightHipPos, rightHipOrn)
    p.resetBasePositionAndOrientation(markers[4], rightKneePos, rightKneeOrn)
    p.resetBasePositionAndOrientation(markers[5], rightAnklePos, rightAnkleOrn)
    p.resetBasePositionAndOrientation(markers[6], leftHipPos, leftHipOrn)
    p.resetBasePositionAndOrientation(markers[7], leftKneePos, leftKneeOrn)
    p.resetBasePositionAndOrientation(markers[8], leftAnklePos, leftAnkleOrn)
    p.resetBasePositionAndOrientation(markers[9], rightShoulderPos, rightShoulderOrn)
    p.resetBasePositionAndOrientation(markers[10], rightElbowPos, rightElbowOrn)
    p.resetBasePositionAndOrientation(markers[11], rightHandPos, rightHandOrn)
    p.resetBasePositionAndOrientation(markers[12], leftShoulderPos, leftShoulderOrn)
    p.resetBasePositionAndOrientation(markers[13], leftElbowPos, leftElbowOrn)
    p.resetBasePositionAndOrientation(markers[14], leftHandPos, leftHandOrn)
    p.resetDebugVisualizerCamera(cameraDistance=2, cameraYaw=60, cameraPitch=-30, cameraTargetPosition=rootPos)

    angles = frame.getAngles()
    p.resetBasePositionAndOrientation(atlas, r2ppos(frame.rootPosition) + np.array([0, 1, 0.1]), r2p(frame.rootRotation))
    for i in range(30):
        p.resetJointState(atlas, i, angles[i])
    time.sleep(dt)
    t += dt
