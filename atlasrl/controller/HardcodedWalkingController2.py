"""This is a standalone script to show how to implement a hard-coded walking controller."""
import numpy as np
import pybullet as p
import pybullet_data
import time,math
from atlasrl.robots.Constants import parameterNames
from scipy.optimize import linprog
from cvxopt import matrix, solvers
solvers.options['show_progress'] = False # Prevent printing

path = "data/atlas/atlas_v4_with_multisense.urdf"

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
atlas = p.loadURDF(path, [0, 0, 0.95])

p.setJointMotorControlArray(atlas, np.arange(30), p.POSITION_CONTROL, forces=np.zeros(30,))
dt = 0.01
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

def moveTowards(currentPoint: np.ndarray, targetPoint: np.ndarray, speed=1) -> np.ndarray:
    delta = targetPoint - currentPoint
    dist = np.linalg.norm(delta)
    stepSize = min(dist, speed * dt)
    return currentPoint + delta / dist * stepSize

for j in range(-1, 30):
    p.changeDynamics(atlas, j, linearDamping=0, angularDamping=0, jointDamping=0, restitution=1)

p.resetJointState(atlas, 1, 0.0, 0)
# p.resetJointState(atlas, 26, -1.0, 0) # Right foot in the air
# p.resetJointState(atlas, 27, 2.0, 0)
state = "init"
while (1):
    # Sensor readings
    (basePose, baseOrn) = p.getBasePositionAndOrientation(atlas)
    (baseLinearSpeed, baseOrnSpeed) = p.getBaseVelocity(atlas)
    jointStates = p.getJointStates(atlas, np.arange(30))
    jointPositions = [j[0] for j in jointStates]
    jointSpeeds = [j[1] for j in jointStates]
    jointTorques = [j[3] for j in jointStates]
    iLeft = parameterNames.index("l_leg_akx")
    iRight = parameterNames.index("r_leg_akx")

    q = np.array(list(baseOrn) + list(basePose) + list(jointPositions))
    qDot = np.array(list(baseOrnSpeed) + list(baseLinearSpeed) + list(jointSpeeds))
    qJoints = np.array(jointPositions)
    qDotJoints = np.array(jointSpeeds)
    zeroVec30 = [0.] * 30
    zeroVec36 = [0.] * 36

    # Calculate jacobian
    jacobian = np.zeros((31, 6, 36))
    for i in range(-1, 30):
        (mass, _, local_inertia_diagonal, local_inertia_pos, local_inertia_orn, _, _, _, _, _, _, _) = p.getDynamicsInfo(atlas, i)
        jac_t, jac_r = p.calculateJacobian(atlas, i, local_inertia_pos, qJoints.tolist(), zeroVec30, zeroVec30)
        jac_t = np.array(jac_t)
        jac_r = np.array(jac_r)
        for j in range(jac_t.shape[1]):
            jac_t[:, j] = p.rotateVector(baseOrn, jac_t[:, j].tolist())
            jac_r[:, j] = p.rotateVector(baseOrn, jac_r[:, j].tolist())
        jacobian[1+i, :3] = jac_r
        jacobian[1+i, 3:] = jac_t

    # Calculating inertia and mass matrix and gravity matrix
    I = np.zeros((31, 6, 6))
    FGrav = np.zeros((31, 6))
    for i in range(-1, 30):
        if i == -1:
            linkWorldOrn = baseOrn
        else:
            (_, linkWorldOrn, _, _, _, _,) = p.getLinkState(atlas, i)
        (mass, _, local_inertia_diagonal, local_inertia_pos, local_inertia_orn, _, _, _, _, _, _, _) = p.getDynamicsInfo(atlas, i)
        R = np.array(p.getMatrixFromQuaternion(linkWorldOrn)).reshape(3, 3) @ np.array(p.getMatrixFromQuaternion(local_inertia_orn)).reshape(3, 3)
        I[1+i, 0:3, 0:3] = R @ np.diag(local_inertia_diagonal) @ np.linalg.inv(R)
        I[1+i, 3:6, 3:6] = np.eye(3) * mass
        FGrav[1+i, 5] = -9.81 * mass

    # Center of mass
    rootJacobian = jacobian[:, :, 0:6] # (30, 6, 6)
    centroidalMass = (I @ rootJacobian).sum(0)
    centroidalMassInertia = centroidalMass[3:6, 0:3]
    centroidalMassMass = centroidalMass[3:6, 3:6]
    centerOfMassSkewToRoot = centroidalMassInertia @ np.linalg.inv(centroidalMassMass)
    centerOfMass = np.array(basePose) + (centerOfMassSkewToRoot[1, 2], centerOfMassSkewToRoot[0, 2], centerOfMassSkewToRoot[0, 1])

    # Foot state machine (l, b, or r)
    (leftFootWorldPos, leftFootWorldOrn, _, _, _, _, leftFootLinearVelocity, leftFootAngularVelocity) = p.getLinkState(atlas, iLeft, computeLinkVelocity=True)
    (rightFootWorldPos, rightFootWorldOrn, _, _, _, _, rightFootLinearVelocity, rightFootAngularVelocity) = p.getLinkState(atlas, iRight, computeLinkVelocity=True)
    comHeight = 1.1
    if state == "init":
        targetCOM = (np.array(rightFootWorldPos) + leftFootWorldPos) / 2 + np.array([0, 0, comHeight])
        currentFoot = "b"
        if t > 5:
            state = "b->l"
            print(state)
    elif state == "b->r":
        targetCOM = moveTowards(targetCOM, rightFootWorldPos + np.array([0, 0, comHeight]), speed=0.1)
        currentFoot = "b"
        if np.linalg.norm((rightFootWorldPos + np.array([0, 0, comHeight]) - centerOfMass)[0:2]) < 0.03:
            targetFootPos = np.array(leftFootWorldPos)
            state = "r"
            targetTargetFootPos = np.array([0.0, 0.0, 0.2]) + leftFootWorldPos
            print(state)
    elif state == "r":
        targetFootPos = moveTowards(targetFootPos, targetTargetFootPos, speed=0.3)
        currentFoot = "r"
        if np.linalg.norm(leftFootWorldPos[2]) > 0.07:
            state = "rForward"
            print(state)
    elif state == "rForward":
        targetFootPos = moveTowards(targetFootPos, np.array([0.2, 0.2, 0.2]) + rightFootWorldPos, speed=0.3)
        currentFoot = "r"
        if np.linalg.norm((np.array([0.2, 0.2, 0.2]) + rightFootWorldPos - leftFootWorldPos)) < 0.03:
            state = "rDown"
            print(state)
    elif state == "rDown":
        targetFootPos = moveTowards(targetFootPos, np.array([0.2, 0.2, 0.0]) + rightFootWorldPos, speed=0.3)
        targetCOM = moveTowards(targetCOM, rightFootWorldPos + np.array([0.1, 0.1, comHeight]), speed=0.1)
        currentFoot = "r"
        if leftFootWorldPos[2] < 0.01 + rightFootWorldPos[2]:
            state = "b->l"
            print(state)
    elif state == "b->l":
        targetCOM = moveTowards(targetCOM, leftFootWorldPos + np.array([0.0, 0.0, comHeight]), speed=0.1)
        currentFoot = "b"
        if np.linalg.norm((leftFootWorldPos + np.array([0, 0, comHeight]) - centerOfMass)[0:2]) < 0.03:
            targetFootPos = np.array(rightFootWorldPos)
            targetTargetFootPos = np.array([0.0, 0.0, 0.2]) + rightFootWorldPos
            state = "l"
            print(state)
    elif state == "l":
        targetFootPos = moveTowards(targetFootPos, targetTargetFootPos, speed=0.3)
        currentFoot = "l"
        if np.linalg.norm(rightFootWorldPos[2]) > 0.07:
            state = "lForward"
            print(state)
    elif state == "lForward":
        targetFootPos = moveTowards(targetFootPos, np.array([0.2, -0.2, 0.2]) + leftFootWorldPos, speed=0.3)
        currentFoot = "l"
        if np.linalg.norm((np.array([0.2, -0.2, 0.2]) + leftFootWorldPos - rightFootWorldPos)) < 0.03:
            state = "lDown"
            print(state)
    elif state == "lDown":
        targetFootPos = moveTowards(targetFootPos, np.array([0.2, -0.2, 0.0]) + leftFootWorldPos, speed=0.3)
        targetCOM = moveTowards(targetCOM, leftFootWorldPos + np.array([0.1, -0.1, comHeight]), speed=0.1)
        currentFoot = "l"
        if rightFootWorldPos[2] < 0.01 + leftFootWorldPos[2]:
            state = "b->r"
            print(state)
    else:
        print(state)
        raise NotImplementedError()

    # Kinematic manager
    qDotDotLeftFoot = np.zeros(6)
    qDotDotRightFoot = np.zeros(6)
    kp = 10
    kd = 2
    if currentFoot == "r":
        qDotDotLeftFoot[0:3] = -kp * np.array(p.getEulerFromQuaternion(leftFootWorldOrn)) - kd * np.array(leftFootAngularVelocity)
        qDotDotLeftFoot[3:6] = kp * (targetFootPos - np.array(leftFootWorldPos)) - kd * np.array(leftFootLinearVelocity)
    elif currentFoot == "l":
        qDotDotRightFoot[0:3] = -kp * np.array(p.getEulerFromQuaternion(rightFootWorldOrn)) - kd * np.array(rightFootAngularVelocity)
        qDotDotRightFoot[3:6] = kp * (targetFootPos - np.array(rightFootWorldPos)) - kd * np.array(rightFootLinearVelocity)
    qDotDotRootJoint = np.concatenate([-kp * np.array(p.getEulerFromQuaternion(baseOrn)) - kd * np.array(baseOrnSpeed), kp * (targetCOM - centerOfMass) - kd * np.array(baseLinearSpeed)])

    # Solve kinematic constraints
    qDotDotUpperBody = (-qJoints * kp - kd * qDotJoints)[0:18]
    leftJacobian = jacobian[1+iLeft] # (6, 36)
    rightJacobian = jacobian[1+iRight] # (6, 36)
    KA = np.concatenate((leftJacobian, rightJacobian), axis=0) # (12,36)
    Kb = np.concatenate((qDotDotLeftFoot, qDotDotRightFoot), axis=0) # (12,)
    KbRed = np.copy(Kb)
    KbRed -= KA[:, :6] @ qDotDotRootJoint
    KbRed -= KA[:, 6:24] @ qDotDotUpperBody
    KARed = KA[:, 24:]
    qDotDotLegs = np.linalg.solve(KARed, KbRed)
    qDotDot = np.concatenate((qDotDotRootJoint, qDotDotUpperBody, qDotDotLegs))
    # qDotDot = qDotDot.clip(-1, 1)

    # Calculate jacobian derivative
    v = jacobian @ qDot
    vAngular = v[:, :3] # (30, 3)
    jacobianDot = np.zeros((31, 6, 36))
    for i in range(36):
        jacobianDot[:, 3:, i] = np.cross(vAngular, jacobian[:, 3:, i])
        jacobianDot[:, :3, i] = np.cross(vAngular, jacobian[:, :3, i]) # TODO: Verify if this is necessary

    # Calculating forces
    # (30, 6) = (30, 6, 6) @ (30, 6, 36) @ (36) + (30, 6, 6) @ (30, 6, 36) @ (36) + (30, 6) * (30, 6, 36) @ (36)
    vDot = jacobian @ qDotDot + jacobianDot @ qDot # (30, 6)
    inertiaForces = -(I @ vDot[:, :, None])[:, :, 0] # (30, 6)
    centroidalMomentum = (inertiaForces[:, None, :] @ rootJacobian).sum(0)[0] # (6,)
    gravityMoment = (FGrav[:, None, :] @ rootJacobian).sum(0)[0] # (6,)

    # Equality constraints for Wrists
    leftJacobian = rootJacobian[1+iLeft] # (6, 6)
    rightJacobian = rootJacobian[1+iRight] # (6, 6)
    WA = np.concatenate((leftJacobian, rightJacobian), axis=0).transpose() # (6, 12)
    Wb = -gravityMoment - centroidalMomentum # (6,)

    # Inequality constraints for Wrists
    Fwrists = np.zeros((31, 6))
    if currentFoot == "b":
        WA = np.concatenate((WA, np.array([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), np.array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]])), axis=0)
        Wb = np.concatenate((Wb, [0, 0]))
        mu = 0.3
        lBack = 0.1
        l = 0.1
        BA = np.array(([
            [0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0], # -F_z < -100
            [0, 0, 0, -1, 0, -mu, 0, 0, 0, 0, 0, 0], # -mu F_z < F_x
            [0, 0, 0, 0, -1, -mu, 0, 0, 0, 0, 0, 0], # -mu F_z < F_y
            [0, 0, 0, 1, 0, -mu, 0, 0, 0, 0, 0, 0], # F_x < mu F_z
            [0, 0, 0, 0, 1, -mu, 0, 0, 0, 0, 0, 0], # F_y < mu F_z
            [-1, 0, 0, 0, 0, -l, 0, 0, 0, 0, 0, 0], # -F_z * l < M_x
            [0, -1, 0, 0, 0, -lBack, 0, 0, 0, 0, 0, 0], # -F_z * l < M_y
            [1, 0, 0, 0, 0, -l, 0, 0, 0, 0, 0, 0], # M_x < F_z * l
            [0, 1, 0, 0, 0, -lBack, 0, 0, 0, 0, 0, 0], # M_y < F_z * l
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1], # -F_z < -100
            [0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, -mu], # -mu F_z < F_x
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -mu], # -mu F_z < F_y
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, -mu], # F_x < mu F_z
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -mu], # F_y < mu F_z
            [0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, -l], # -F_z * l < M_x
            [0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, -lBack], # -F_z * l < M_y
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, -l], # M_x < F_z * l
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, -lBack], # M_y < F_z * l
        ]))
        weightMoment = 1
        weightForce = 0.001
        Bb = np.zeros(BA.shape[0])
        Bb[0] = -100
        Bb[9] = -100
        cost = np.array([weightMoment, weightMoment, weightMoment, weightForce, weightForce, weightForce, weightMoment, weightMoment, weightMoment, weightForce, weightForce, weightForce]) # Minimize the moments
        res = solvers.qp(matrix(np.diag(cost)), matrix(np.zeros_like(cost)), matrix(BA), matrix(Bb), matrix(WA), matrix(Wb))
        if res["status"] == "optimal":
            wrists = np.array(res["x"])[:, 0]
            print(wrists)
        else:
            print("Infeasible")
            pass
        Fwrists[1+iLeft] = wrists[:6]
        Fwrists[1+iRight] = wrists[6:]
    elif currentFoot == "l":
        wrists = np.linalg.solve(WA[:, :6], Wb)
        Fwrists[1+iLeft] = wrists
    elif currentFoot == "r":
        wrists = np.linalg.solve(WA[:, 6:], Wb)
        Fwrists[1+iRight] = wrists
    else:
        raise NotImplementedError()
    # wrists @ WA.transpose() - Wb = 0
    # =>
    # FWrists @ rootJacobian + FGrav @ rootJacobian + inertiaForces @ rootJacobian = 0
    # Calculating joint torques
    totalForce = inertiaForces + FGrav + Fwrists # (30, 6)
    jointTorques = np.zeros(30)
    for i in range(-1, 30):
        if i == -1:
            jointJacobian = jacobian[:, :, 0:6] # (30, 6, 6)
            # print(-(totalForce[:, None, :] @ jointJacobian).sum(0)) # This one should be 0 if everything went right
        else:
            jointJacobian = jacobian[:, :, 6+i:7+i]  # (30, 6, 1)
            # (30, 1, 6) * (30, 6, 1)
            jointTorques[i] = -(totalForce[:, None, :] @ jointJacobian).sum()
    # jointTorques[-6:] *= 0
    while True:
        p.setJointMotorControlArray(atlas, np.arange(30), p.TORQUE_CONTROL, forces=jointTorques)
        p.stepSimulation()
        time.sleep(dt * 1)
        t += dt
        break


