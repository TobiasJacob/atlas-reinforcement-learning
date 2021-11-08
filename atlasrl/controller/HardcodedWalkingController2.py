"""This is a standalone script to explort how the Atlas model looks like."""
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

p.resetJointState(atlas, 1, 0.3, 0)
while (1):
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

    posDelta = np.array([-0.00, 0, 0.90]) - np.array(basePose)
    posDelta *= np.array([1, 1, 0])
    # print(posDelta)
    eulerAngles = np.array(p.getEulerFromQuaternion(baseOrn))
    qDotDot = np.concatenate([-eulerAngles - 0.1 * np.array(baseOrnSpeed), posDelta * 1 - 0.1 * np.array(baseLinearSpeed), -qJoints * 2 - 0.1 * qDotJoints])
    # qDotDot = np.concatenate([(0, 0, 0), posDelta * 1, -qJoints * 0])
    # print("q", q)
    # print("qDot", qDot)
    # print(qDotDot)
    qDotDot = qDotDot.clip(-1, 1)
    # print(qDotDot)
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
        jac_t = np.array(jac_t)
        jac_r = np.array(jac_r)
        for j in range(jac_t.shape[1]):
            jac_t[:, j] = p.rotateVector(baseOrn, jac_t[:, j].tolist())
            jac_r[:, j] = p.rotateVector(baseOrn, jac_r[:, j].tolist())
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
        jacobianDot[:, :3, i] = np.cross(vAngular, jacobian[:, :3, i]) # TODO: Verify if this is necessary
    # print("jacobianDot")
    # print(jacobianDot)

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
    # print("qDot")
    # print(qDot)
    # print("vDot")
    # print(vDot)
    rootJacobian = jacobian[:, :, 0:6] # (30, 6, 6)
    inertiaForces = -(I @ vDot[:, :, None])[:, :, 0] # (30, 6)
    # inertiaForces *= 0 # TODO: Add centroidal Momentum
    centroidalMomentum = (inertiaForces[:, None, :] @ rootJacobian).sum(0)[0] # (6,)
    gravityMoment = (FGrav[:, None, :] @ rootJacobian).sum(0)[0] # (6,)

    # Calculating Wrists
    iLeft = parameterNames.index("r_leg_akx")
    iRight = parameterNames.index("l_leg_akx")
    # (leftWorldPos, leftWorldOrn, leftPos, _, _, _,) = p.getLinkState(atlas, iLeft)
    # (rightWorldPos, rightWorldOrn, rightPos, _, _, _,) = p.getLinkState(atlas, iRight)
    # leftPos = np.array(p.rotateVector(leftWorldOrn, leftPos)) + leftWorldPos
    # rightPos = np.array(p.rotateVector(rightWorldOrn, rightPos)) + rightWorldPos
    # MomentDirection = (leftPos - rightPos) / np.linalg.norm(leftPos - rightPos)
    leftJacobian = rootJacobian[iLeft] # (6, 6)
    rightJacobian = rootJacobian[iRight] # (6, 6)
    WA = np.concatenate((leftJacobian, rightJacobian), axis=0).transpose() # (6, 12)
    Wb = -gravityMoment - centroidalMomentum # (6,)
    print("centroidalMomentum", centroidalMomentum)
    # print(gravityMoment)
    # print(centroidalMomentum)
    # Constrain M_z = 0
    print("gravityMoment", gravityMoment)
    WA = np.concatenate((WA, np.array([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), np.array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]])), axis=0)
    Wb = np.concatenate((Wb, [0, 0]))
    # Inequality constraints
    mu = 0.3
    lBack = 0.2
    l = 0.2
    BA = np.array(([
        [0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0], # -F_z < 0
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1], # -F_z < 0
        [0, 0, 0, -1, 0, -mu, 0, 0, 0, 0, 0, 0], # -mu F_z < F_x
        [0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, -mu], # -mu F_z < F_x
        [0, 0, 0, 0, -1, -mu, 0, 0, 0, 0, 0, 0], # -mu F_z < F_y
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -mu], # -mu F_z < F_y
        [0, 0, 0, 1, 0, -mu, 0, 0, 0, 0, 0, 0], # F_x < mu F_z
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, -mu], # F_x < mu F_z
        [0, 0, 0, 0, 1, -mu, 0, 0, 0, 0, 0, 0], # F_y < mu F_z
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -mu], # F_y < mu F_z
        [-1, 0, 0, 0, 0, -l, 0, 0, 0, 0, 0, 0], # -F_z * l < M_x
        [0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, -l], # -F_z * l < M_x
        [0, -1, 0, 0, 0, -lBack, 0, 0, 0, 0, 0, 0], # -F_z * l < M_y
        [0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, -lBack], # -F_z * l < M_y
        [1, 0, 0, 0, 0, -l, 0, 0, 0, 0, 0, 0], # M_x < F_z * l
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, -l], # M_x < F_z * l
        [0, 1, 0, 0, 0, -lBack, 0, 0, 0, 0, 0, 0], # M_y < F_z * l
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, -lBack], # M_y < F_z * l
    ]))
    Bb = np.zeros(BA.shape[0])
    # wrists = np.linalg.lstsq(WA, Wb, rcond=1)[0]
    weightMoment = 1
    weightForce = 0.001
    cost = np.array([weightMoment, weightMoment, weightMoment, weightForce, weightForce, weightForce, weightMoment, weightMoment, weightMoment, weightForce, weightForce, weightForce]) # Minimize the moments
    # res = linprog(cost, A_eq=WA, b_eq=Wb, A_ub=BA, b_ub=Bb, bounds=(None, None), method="highs-ipm")
    res = solvers.qp(matrix(np.diag(cost)), matrix(np.zeros_like(cost)), matrix(BA), matrix(Bb), matrix(WA), matrix(Wb))
    if res["status"] == "optimal":
        wrists = np.array(res["x"])[:, 0]
    else:
        # print(res)
        print("Infeasible")
        pass
    # wrists = np.array(res["x"])[:, 0]
    # wrists[0:5] *= 1.0
    # wrists[6:11] *= 1.0
    # break
    Fwrists = np.zeros((30, 6))
    Fwrists[iLeft] = wrists[:6]
    Fwrists[iRight] = wrists[6:]
    # print(wrists @ WA.transpose() - Wb)
    # wrists @ WA.transpose() - Wb = 0
    # FWrists @ rootJacobian + gravityMoment + centroidalMomentum = 0
    # FWrists @ rootJacobian + FGrav @ rootJacobian + inertiaForces @ rootJacobian = 0
    # print("vdot", np.linalg.norm(vDot, axis=1))
    # print(centroidalMomentum)
    # print(gravityMoment)
    # print(wrists)
    # break

    # Calculating joint torques
    totalForce = inertiaForces + FGrav + Fwrists # (30, 6)
    # print(totalForce)
    jointTorques = np.zeros(30)
    for i in range(-1, 30):
        if i == -1:
            jointJacobian = jacobian[:, :, 0:6] # (30, 6, 6)
            print(-(totalForce[:, None, :] @ jointJacobian).sum(0))
        else:
            jointJacobian = jacobian[:, :, 6+i:7+i]  # (30, 6, 1)
            # (30, 1, 6) * (30, 6, 1)
            jointTorques[i] = -(totalForce[:, None, :] @ jointJacobian).sum()
    # jointTorques[parameterNames.index("l_leg_aky")] = 0
    # jointTorques[parameterNames.index("l_leg_akx")] = 0
    # jointTorques[parameterNames.index("r_leg_aky")] = 0
    # jointTorques[parameterNames.index("r_leg_akx")] = 0
    print(np.round(jointTorques))
    # if jointTorques[-2] < 0:
    #     # print(jacobianDot[0])
    #     # print(qDot)
    #     break
    while True:
        p.setJointMotorControlArray(atlas, np.arange(30), p.TORQUE_CONTROL, forces=jointTorques)
        p.stepSimulation()
        time.sleep(dt * 100)
        t += dt
        break


