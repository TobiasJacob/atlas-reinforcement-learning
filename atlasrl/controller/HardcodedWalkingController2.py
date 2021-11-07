"""This is a standalone script to explort how the Atlas model looks like."""
import numpy as np
import pybullet as p
import pybullet_data
import time,math
from atlasrl.robots.Constants import parameterNames

path = "data/atlas/atlas_v4_with_multisense.urdf"

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
atlas = p.loadURDF(path, [0, 0, 0.95])

p.setJointMotorControlArray(atlas, np.arange(30), p.VELOCITY_CONTROL, forces=np.zeros(30,))
dt = 0.001
p.setTimeStep(dt)
p.loadURDF("plane.urdf",[0,0,0], useFixedBase=True)

p.resetDebugVisualizerCamera(cameraDistance=3, cameraYaw=70, cameraPitch=-20, cameraTargetPosition=[0, 0, 0.5])
p.setGravity(0,0,-9.81)

np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=np.inf)
np.set_printoptions(precision=4)
t=0

def skew(x):
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])

while (1):
    # p.resetJointState(atlas, 2, 2, 0)
    (basePose, baseOrn) = p.getBasePositionAndOrientation(atlas)
    (baseLinearSpeed, baseOrnSpeed) = p.getBaseVelocity(atlas)
    jointStates = p.getJointStates(atlas, np.arange(30))
    jointPositions = [j[0] for j in jointStates]
    jointSpeeds = [j[1] for j in jointStates]
    jointTorques = [j[3] for j in jointStates]

    q = np.array(list(baseOrn) + list(basePose) + list(jointPositions))
    
    qDot = np.array(list(baseOrnSpeed) + list(baseLinearSpeed) + list(jointSpeeds))
    qJoints = np.array(jointPositions)
    qDotJoints = np.array(jointSpeeds)
    zeroVec30 = [0.] * 30
    zeroVec36 = [0.] * 36

    posDelta = np.array([-0.05, 0, 0.6]) - np.array(basePose)
    qDotDot = np.concatenate([(0, 0, 0), posDelta * 10 - 5 * np.array(baseLinearSpeed), -qJoints * 2 - qDotJoints])
    qDotDot = qDotDot.clip(-1, 1)
    qDotDot = np.zeros_like(qDotDot)
    # qDotDot[6+parameterNames.index("l_arm_shx")] += 4

    # Calculate jacobian
    jacobian = np.zeros((30, 6, 36))
    # axis = np.zeros((30, 3))
    for i in range(30):
        (_, _, _, _, _, _, _, _, _, _, _, _, _, jointAxis, _, _, _) = p.getJointInfo(atlas, i)
        (linkWorldPos, linkWorldOrn, linkInertiaPos, linkInertiaOrn, _, _,) = p.getLinkState(atlas, i)
        # axis[i] = p.rotateVector(linkWorldOrn, np.array(jointAxis))
        jac_t, jac_r = p.calculateJacobian(atlas, i, linkInertiaPos, qJoints.tolist(), zeroVec30, zeroVec30)
        jacobian[i, :3] = jac_r
        jacobian[i, 3:] = jac_t
    # print("jacobian")
    # print(jacobian[0])

    # Calculate jacobian derivative
    v = jacobian @ qDot
    vAngular = v[:, :3] # (30, 3)
    # vAngular2 = axis * qDotJoints[:, None]
    jacobianDot = np.zeros((30, 6, 36))
    for i in range(36):
        jacobianDot[:, 3:, i] = np.cross(vAngular, jacobian[:, 3:, i])
    # print("jacobianDot")
    # print(jacobianDot[0])

    # Calculating inertia and mass matrix and gravity matrix
    I = np.zeros((30, 6, 6))
    FGrav = np.zeros((30, 6))
    for i in range(30):
        (_, linkWorldOrn, linkInertiaPos, linkInertiaOrn, _, _,) = p.getLinkState(atlas, i)
        (mass, _, local_inertia_diagonal, local_inertia_pos, local_inertia_orn, _, _, _, _, _, _, _) = p.getDynamicsInfo(atlas, i)
        R = np.array(p.getMatrixFromQuaternion(linkWorldOrn)).reshape(3, 3) @ np.array(p.getMatrixFromQuaternion(local_inertia_orn)).reshape(3, 3)
        # local_inertia_pos = p.rotateVector(linkWorldOrn, np.array(local_inertia_pos))
        I[i, 0:3, 0:3] = R @ np.diag(local_inertia_diagonal) @ np.linalg.inv(R)
        # I[i, 0:3, 3:6] = skew(local_inertia_pos) * mass
        # I[i, 3:6, 0:3] = skew(local_inertia_pos) * mass
        I[i, 3:6, 3:6] = np.eye(3) * mass
        FGrav[i, 5] = -9.81 * mass

    # Calculating forces
    # (30, 6) = (30, 6, 6) @ (30, 6, 36) @ (36) + (30, 6, 6) @ (30, 6, 36) @ (36) + (30, 6) * (30, 6, 36) @ (36)
    vDot = jacobian @ qDotDot + jacobianDot @ qDot # (30, 6)
    inertiaForces = -(I @ vDot[:, :, None])[:, :, 0] # (30, 6)
    centroidalMomentum = inertiaForces.sum(0) # (6,)
    rootJacobian = jacobian[:, :, 0:6] # (30, 6, 6)
    gravityMoment = (FGrav[:, None, :] @ rootJacobian).sum(0)[0] # (6,)

    # Calculating Wrists
    iLeft = parameterNames.index("r_leg_akx")
    iRight = parameterNames.index("l_leg_akx")
    (leftWorldPos, leftWorldOrn, leftPos, _, _, _,) = p.getLinkState(atlas, iLeft)
    (rightWorldPos, rightWorldOrn, rightPos, _, _, _,) = p.getLinkState(atlas, iRight)
    leftPos = np.array(p.rotateVector(leftWorldOrn, leftPos)) + leftWorldPos
    rightPos = np.array(p.rotateVector(rightWorldOrn, rightPos)) + rightWorldPos
    MomentDirection = (leftPos - rightPos) / np.linalg.norm(leftPos - rightPos)
    leftJacobian = jacobian[iLeft, :, :6] # (6, 6)
    rightJacobian = jacobian[iRight, :, :6] # (6, 6)
    # WA = np.concatenate((leftJacobian, rightJacobian), axis=1)
    
    WA = np.zeros((6, 7))
    # 1 Condition: F1 + F2 = Fges
    WA[3:6, 0:3] = np.eye(3)
    WA[3:6, 3:6] = np.eye(3)
    # 2 Condition: 2 * Alpha * (R1 - R2) + R1 x F1 + R2 x F2 = Mges
    WA[0:3, 0:3] = skew(leftPos - basePose)
    WA[0:3, 3:6] = skew(rightPos - basePose)
    WA[0:3, 6] = 2 * MomentDirection
    Wb = -gravityMoment - centroidalMomentum
    wrists = np.linalg.lstsq(WA, Wb, rcond=1)[0]
    wristLeft = np.concatenate((wrists[6] * MomentDirection, wrists[0:3]))
    wristRight = np.concatenate((wrists[6] * MomentDirection, wrists[3:6]))
    Fwrists = np.zeros((30, 6))
    Fwrists[iLeft] = wristLeft
    Fwrists[iRight] = wristRight

    # Calculating joint torques
    totalForce = inertiaForces + FGrav + Fwrists # (30, 6)
    jointTorques = np.zeros(30)
    for i in range(-1, 30):
        if i == -1:
            jointJacobian = jacobian[:, :, 0:6] # (30, 6, 6)
            print(-(totalForce[:, None, :] @ jointJacobian))
        else:
            jointJacobian = jacobian[:, :, 6+i:7+i]  # (30, 6, 1)
            # (30, 1, 6) * (30, 6, 1)
            jointTorques[i] = -(totalForce[:, None, :] @ jointJacobian).sum()
    while True:
        p.setJointMotorControlArray(atlas, np.arange(30), p.TORQUE_CONTROL, forces=jointTorques)
        p.stepSimulation()
        time.sleep(dt * 10)
        t += dt
        break


