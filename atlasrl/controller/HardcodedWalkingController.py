"""This is a standalone script to explort how the Atlas model looks like."""
import numpy as np
import pybullet as p
import pybullet_data
import time,math
from atlasrl.robots.Constants import parameterNames

path = "data/atlas/atlas_v4_with_multisense.urdf"

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0)
atlas = p.loadURDF(path, [0, 0, 0.95])

def skew(x):
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])

p.setJointMotorControlArray(atlas, np.arange(30), p.VELOCITY_CONTROL, forces=np.zeros(30,))#, forces=[10000] * 30) #, positionGain=0, velocityGain=0)
dt = 0.001
p.setTimeStep(dt)

for i in range (p.getNumJoints(atlas)):
    info = p.getJointInfo(atlas,i)
    print("Joint info", i, info)
for i in range (p.getNumJoints(atlas)):
    info = p.getLinkState(atlas,i)
    print("Link info", i, info)

p.loadURDF("plane.urdf",[0,0,0], useFixedBase=True)

p.resetDebugVisualizerCamera(cameraDistance=3, cameraYaw=70, cameraPitch=-20, cameraTargetPosition=[0, 0, 0.5])
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1)
p.setGravity(0,0,-9.81)

np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=np.inf)
np.set_printoptions(precision=4)
t=0
(startWorldPos, _) = p.getBasePositionAndOrientation(atlas)
# p.setRealTimeSimulation(1)
while (1):
    time.sleep(0.01)
    t+=dt
    p.stepSimulation()

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

    k_p = 50.
    k_d = k_p * 0.1

    zero_vec = [0.0] * len(qJoints)
    zero_vec_36 = [0.0] * 36
    qDesired = np.zeros(30)
    desiredQDotDot = - k_p * (qJoints - qDesired) - k_d * qDotJoints
    desiredQDotDot = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 9.81] + desiredQDotDot.tolist())

    i_leg_akx = 22
    foot_loc = p.getLinkState(atlas, i_leg_akx)[0]
    jac_t_1, jac_r_1 = p.calculateJacobian(atlas, i_leg_akx, foot_loc, qJoints.tolist(), (qDotJoints * 0.0).tolist(), zero_vec)
    # torques += np.array([0, 0, -200]) @ jac_t
    # torques += linearForce / 2 @ jac_t #+ momentum / 2 @ jac_r

    i_leg_akx = 28
    foot_loc = p.getLinkState(atlas, i_leg_akx)[0]
    jac_t_2, jac_r_2 = p.calculateJacobian(atlas, i_leg_akx, foot_loc, qJoints.tolist(), zero_vec, zero_vec)
    # torques = np.array([0, -1000, 0]) @ jac_t
    A = np.concatenate((np.array(jac_t_1), np.array(jac_t_2)), axis=0)
    # desiredQDotDot[6:] = np.linalg.lstsq(A[:, 6:], -A[:, :6] @ np.array([0, 0, 0, 0, 0, 9.81]))[0]
    # print(np.linalg.lstsq(np.concatenate((np.array(jac_t_1)[:, 6:], np.array(jac_t_2)[:, 6:]), axis=0), np.array([0, 0, 9.81, 0, 0, 9.81])))
    # desiredQDotDot += np.linalg.lstsq(np.concatenate((np.array(jac_t_1), np.array(jac_t_2)), axis=0), np.array([0, 0, -9.81, 0, 0, -9.81]))[0]
    # desiredQDotDot[:6] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    # torques += linearForce / 2 @ jac_t #+ momentum / 2 @ jac_r

    # torques = np.array(p.calculateInverseDynamics(atlas, q.tolist(), qDot.tolist(), desiredQDotDot.tolist()))

    # Solve inverse dynamics problem
    # Step 0: Get center of mass
    totalMass = 0
    centerOfMass = np.zeros(3)
    totalInertia = np.zeros((3, 3))
    for i in range(30):
        (mass, _, local_inertia_diagonal, local_inertia_pos, local_inertia_orn, _, _, _, _, _, _, _) = p.getDynamicsInfo(atlas, i)
        (linkWorldPos, linkWorldOrn, linkInertiaPos, linkInertiaOrn, _, _, _, linkAngularVelocity) = p.getLinkState(atlas, i, computeLinkVelocity=True)
        linkInertiaPos = np.array(p.rotateVector(linkWorldOrn, linkInertiaPos))
        centerOfMass += mass * np.array(linkInertiaPos + linkWorldPos)
        totalMass += mass
    centerOfMass /= totalMass
    for i in range(30):
        (mass, _, local_inertia_diagonal, local_inertia_pos, local_inertia_orn, _, _, _, _, _, _, _) = p.getDynamicsInfo(atlas, i)
        (linkWorldPos, linkWorldOrn, linkInertiaPos, linkInertiaOrn, _, _, _, linkAngularVelocity) = p.getLinkState(atlas, i, computeLinkVelocity=True)
        linkInertiaPos = np.array(p.rotateVector(linkWorldOrn, linkInertiaPos))
        totalInertia += skew(linkInertiaPos - centerOfMass) * mass
        totalInertia += np.diag(p.rotateVector(linkWorldOrn, p.rotateVector(linkInertiaOrn, local_inertia_diagonal)))
    # Calculate Wrists
    (atlasWorldPos, atlasWorldOrn) = p.getBasePositionAndOrientation(atlas)
    (atlasWorldSpeed, atlasAngularSpeed) = p.getBaseVelocity(atlas)
    positionError = np.array([0.00, 0, 1.3]) - centerOfMass
    positionError -= np.array(atlasWorldSpeed) * 0.1
    wristsPower = 1.0 + positionError[2]
    print(wristsPower)
    wristsMoments = -np.array((-positionError[1], positionError[0], 0)) * 300
    wristRight = np.array([0, 0, 9.81 * totalMass / 2 * wristsPower * (1 - positionError[1] * 0.2)])
    wristLeft = np.array([0, 0, 9.81 * totalMass / 2 * wristsPower * (1 + positionError[1] * 0.2)])

    # 1. Step: Calculate accelerations of all inertias
    linearAcceleration = np.zeros((31, 3)) # TODO: calculate the actual linear acceleration based on wrist power
    angularAcceleration = np.zeros((31, 3)) # TODO: calculate the actual linear acceleration based on wrist power
    wristLeftLoc = p.getLinkState(atlas, parameterNames.index("l_leg_akx"))[0]
    wristRightLoc = p.getLinkState(atlas, parameterNames.index("r_leg_akx"))[0]
    linearAcceleration[0] = (np.array([0, 0, 9.81 * totalMass]) - wristRight - wristLeft) / totalMass
    angularAcceleration[0] = np.linalg.inv(totalInertia) @ (np.cross(wristLeftLoc - centerOfMass, wristLeft) + np.cross(wristRightLoc - centerOfMass, wristRight) + 2 * wristsMoments)
    print(linearAcceleration[0], angularAcceleration[0])
    desiredJointAcceleration = np.zeros(30)
    k_p = 1.
    k_d = k_p * 0.0
    desiredJointAcceleration = - k_p * (qJoints - qDesired) - k_d * qDotJoints
    for i in range(30):
        (_, _, _, _, _, flags, damping, friction, _, _, _, _, linkName, jointAxis, parentFramePos, parentFrameOrn, parentIndex) = p.getJointInfo(atlas, i)
        if parentIndex != -1:
            (_, _, _, _, _, _, _, _, _, _, _, _, _, parentJointAxis, _, _, _) = p.getJointInfo(atlas, parentIndex)
            (parentWorldPos, parentWorldOrn, _, _, _, _, _, parentAngularVel) = p.getLinkState(atlas, parentIndex, computeLinkVelocity=True)
        else:
            parentJointAxis = np.zeros(3)
            (parentWorldPos, parentWorldOrn) = p.getBasePositionAndOrientation(atlas)
            (_, parentAngularVel) = p.getBaseVelocity(atlas)
        (mass, _, local_inertia_diagonal, local_inertia_pos, local_inertia_orn, _, _, _, _, _, _, _) = p.getDynamicsInfo(atlas, i)
        (linkWorldPos, linkWorldOrn, linkInertiaPos, linkInertiaOrn, _, _, _, linkAngularVelocity) = p.getLinkState(atlas, i, computeLinkVelocity=True)
        globalJointAxis = np.array(p.rotateVector(linkWorldOrn, np.array(jointAxis) * desiredJointAcceleration[i]))
        parentJointAxis = np.array(p.rotateVector(parentWorldOrn, parentJointAxis))
        parentFramePos = np.array(p.rotateVector(parentWorldOrn, parentFramePos))
        linkInertiaPos = np.array(p.rotateVector(linkWorldOrn, linkInertiaPos))
        # consisting of acceleration force and centrifugal force
        linearAcceleration[1+i] = linearAcceleration[1+parentIndex] + (desiredJointAcceleration[parentIndex] if parentIndex != -1 else 0) * np.cross(parentJointAxis, parentFramePos) + np.cross(np.cross(parentFramePos, parentAngularVel), parentAngularVel)
        angularAcceleration[1+i] = angularAcceleration[1+parentIndex] + globalJointAxis

    # Choose wrists and root acc
    forces = np.zeros((31, 3))
    moments = np.zeros((31, 3))
    torques = np.zeros(30)
    for i in reversed(range(30)):
        (_, _, _, _, _, flags, damping, friction, _, _, _, _, linkName, jointAxis, parentFramePos, parentFrameOrn, parentIndex) = p.getJointInfo(atlas, i)
        if parentIndex != -1:
            (_, _, _, _, _, _, _, _, _, _, _, _, _, parentJointAxis, _, _, _) = p.getJointInfo(atlas, parentIndex)
            (parentWorldPos, parentWorldOrn, _, _, _, _, _, parentAngularVel) = p.getLinkState(atlas, parentIndex, computeLinkVelocity=True)
        else:
            parentJointAxis = np.zeros(3)
            (parentWorldPos, parentWorldOrn) = p.getBasePositionAndOrientation(atlas)
            (_, parentAngularVel) = p.getBaseVelocity(atlas)
        (mass, _, local_inertia_diagonal, local_inertia_pos, local_inertia_orn, _, _, _, _, _, _, _) = p.getDynamicsInfo(atlas, i)
        (linkWorldPos, linkWorldOrn, linkInertiaPos, linkInertiaOrn, _, _, _, linkAngularVelocity) = p.getLinkState(atlas, i, computeLinkVelocity=True)
        globalJointAxis = np.array(p.rotateVector(linkWorldOrn, jointAxis))
        parentFramePos = np.array(p.rotateVector(parentWorldOrn, parentFramePos))
        local_inertia_pos = np.array(p.rotateVector(linkWorldOrn, local_inertia_pos))
        forces[1+i] += mass * (linearAcceleration[i] + np.array([0, 0, 9.81])) + mass * np.cross(np.cross(local_inertia_pos, linkAngularVelocity), linkAngularVelocity) + mass * np.cross(local_inertia_pos, angularAcceleration[i])
        moments[1+i] += np.cross(local_inertia_pos, forces[1+i])
        if i == parameterNames.index("r_leg_akx"):
            forces[1+i] -= wristRight
            moments[1+i] += wristsMoments
        if i == parameterNames.index("l_leg_akx"):
            forces[1+i] -= wristLeft
            moments[1+i] += wristsMoments
            # TODO: Add moment here depending on center of pressure
        forces[1+parentIndex] += forces[1+i]
        moments[1+i] += np.diag(p.rotateVector(linkWorldOrn, p.rotateVector(linkInertiaOrn, local_inertia_diagonal))) @ angularAcceleration[i]
        moments[1+parentIndex] += moments[1+i] + np.cross(parentFramePos, forces[1+i])
        torques[i] = np.dot(moments[1+i], globalJointAxis) # np.dot(np.cross(forces[1+i], globalJointAxis), globalJointAxis)
    # print("linearAcceleration")
    # print(linearAcceleration)
    # print("angularAcceleration")
    # print(angularAcceleration)
    # print("desiredJointAcceleration")
    # print(desiredJointAcceleration)
    # print("forces")
    # print(forces)
    # print("moments")
    # print(moments)
    # print("torques")
    # print(torques)
    print("Leftover force and moment:", forces[0], moments[0], "total:", totalMass)
    # print(torques)
    # break
    # 2. Step: Calculate all torques according to the pricinple of virtual work
    # torques = np.array(p.calculateInverseDynamics(atlas, q.tolist(), qDot.tolist(), desiredQDotDot.tolist()))
    torques = torques.clip(-1000, 1000)
    # p.removeAllUserDebugItems()
    # for i in range(30):
    #     link_loc = p.getLinkState(atlas, i)[0]
    #     link_orn = p.getLinkState(atlas, i)[1]
    #     delta = p.rotateVector(link_orn, [torques[i+6] / 100, 0, 0])
    #     p.addUserDebugLine(link_loc, (np.array(link_loc) + np.array(delta)).tolist(), lineWidth=10)
    if torques.max() > 5000:
        # time.sleep(2.0)
        print("q")
        print(q)
        print("qDot")
        print(qDot)
        print("desiredQDotDot")
        print(desiredQDotDot)
    # linearForce = torques[0:3]
    # momentum = torques[3:6]
    # jointForces = torques[6:]

    # print(t, len(torques), torques.max())
    p.setJointMotorControlArray(atlas, np.arange(30), p.TORQUE_CONTROL, forces=torques)#, forces=[10000] * 30) #, positionGain=0, velocityGain=0)
    # for _ in range(5):
    #     p.setJointMotorControlArray(atlas, np.arange(30), p.TORQUE_CONTROL, forces=torques[6:] * 1.0)#, forces=[10000] * 30) #, positionGain=0, velocityGain=0)
    #     p.stepSimulation()
    #     time.sleep(0.1)
    #     time.sleep(dt)
    # while True:
    #     p.setJointMotorControlArray(atlas, np.arange(30), p.TORQUE_CONTROL, forces=torques[6:] * 1.0)#, forces=[10000] * 30) #, positionGain=0, velocityGain=0)
    #     time.sleep(0.1)
    #     p.stepSimulation()
    # p.setJointMotorControlArray(atlas, np.arange(30), p.POSITION_CONTROL, np.zeros(30))#, forces=[10000] * 30) #, positionGain=0, velocityGain=0)
    # for i in range(p.getNumJoints(atlas)):
    #     p.setJointMotorControl2(atlas, i, p.TORQUE_CONTROL, force=torques[i + 6]) # , positionGain=5)

