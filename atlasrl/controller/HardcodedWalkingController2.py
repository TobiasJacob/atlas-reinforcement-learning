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

for j in range(-1, 30):
    p.changeDynamics(atlas, j, linearDamping=0, angularDamping=0, jointDamping=0, restitution=1)

p.resetJointState(atlas, 1, 0.0, 0)
while (1):
    print(t)
    (basePose, baseOrn) = p.getBasePositionAndOrientation(atlas)
    (baseLinearSpeed, baseOrnSpeed) = p.getBaseVelocity(atlas)
    jointStates = p.getJointStates(atlas, np.arange(30))
    jointPositions = [j[0] for j in jointStates]
    jointSpeeds = [j[1] for j in jointStates]
    jointTorques = [j[3] for j in jointStates]
    iLeft = parameterNames.index("r_leg_akx")
    iRight = parameterNames.index("l_leg_akx")

    q = np.array(list(baseOrn) + list(basePose) + list(jointPositions))
    
    qDot = np.array(list(baseOrnSpeed) + list(baseLinearSpeed) + list(jointSpeeds))
    qJoints = np.array(jointPositions)
    qDotJoints = np.array(jointSpeeds)
    zeroVec30 = [0.] * 30
    zeroVec36 = [0.] * 36

    # Calculate jacobian
    jacobian = np.zeros((30, 6, 36))
    for i in range(30):
        (_, _, _, _, _, _, _, _, _, _, _, _, _, jointAxis, _, _, _) = p.getJointInfo(atlas, i)
        (linkWorldPos, linkWorldOrn, linkInertiaPos, linkInertiaOrn, _, _,) = p.getLinkState(atlas, i)
        jac_t, jac_r = p.calculateJacobian(atlas, i, linkInertiaPos, qJoints.tolist(), zeroVec30, zeroVec30)
        jac_t = np.array(jac_t)
        jac_r = np.array(jac_r)
        for j in range(jac_t.shape[1]):
            jac_t[:, j] = p.rotateVector(baseOrn, jac_t[:, j].tolist())
            jac_r[:, j] = p.rotateVector(baseOrn, jac_r[:, j].tolist())
        jacobian[i, :3] = jac_r
        jacobian[i, 3:] = jac_t

    # Solve kinematic constraints
    posDelta = np.array([-0.00, 0, 0.90]) - np.array(basePose)
    posDelta *= np.array([10, 10, 5])
    eulerAngles = np.array(p.getEulerFromQuaternion(baseOrn))
    qDotDotRootJoint = np.concatenate([-10 * eulerAngles - 2 * np.array(baseOrnSpeed), posDelta - 2 * np.array(baseLinearSpeed)])
    qDotDotUpperBody = (-qJoints * 10 - 1.0 * qDotJoints)[0:18]
    qDotDotLeftFoot = np.zeros(6)
    qDotDotRightFoot = np.zeros(6)
    leftJacobian = jacobian[iLeft] # (6, 36)
    rightJacobian = jacobian[iRight] # (6, 36)
    KA = np.concatenate((leftJacobian, rightJacobian), axis=0) # (12,36)
    Kb = np.concatenate((qDotDotLeftFoot, qDotDotRightFoot), axis=0) # (12,)
    KbRed = np.copy(Kb)
    KbRed -= KA[:, :6] @ qDotDotRootJoint
    KbRed -= KA[:, 6:24] @ qDotDotUpperBody
    KARed = KA[:, 24:]
    qDotDotLegs = np.linalg.solve(KARed, KbRed)
    qDotDot = np.concatenate((qDotDotRootJoint, qDotDotUpperBody, qDotDotLegs))
    qDotDot = qDotDot.clip(-10, 10)

    # Calculate jacobian derivative
    v = jacobian @ qDot
    vAngular = v[:, :3] # (30, 3)
    jacobianDot = np.zeros((30, 6, 36))
    for i in range(36):
        jacobianDot[:, 3:, i] = np.cross(vAngular, jacobian[:, 3:, i])
        jacobianDot[:, :3, i] = np.cross(vAngular, jacobian[:, :3, i]) # TODO: Verify if this is necessary

    # Calculating inertia and mass matrix and gravity matrix
    I = np.zeros((30, 6, 6))
    FGrav = np.zeros((30, 6))
    for i in range(30):
        (_, linkWorldOrn, linkInertiaPos, linkInertiaOrn, _, _,) = p.getLinkState(atlas, i)
        (mass, _, local_inertia_diagonal, local_inertia_pos, local_inertia_orn, _, _, _, _, _, _, _) = p.getDynamicsInfo(atlas, i)
        R = np.array(p.getMatrixFromQuaternion(linkWorldOrn)).reshape(3, 3) @ np.array(p.getMatrixFromQuaternion(local_inertia_orn)).reshape(3, 3)
        I[i, 0:3, 0:3] = R @ np.diag(local_inertia_diagonal) @ np.linalg.inv(R)
        I[i, 3:6, 3:6] = np.eye(3) * mass
        FGrav[i, 5] = -9.81 * mass

    # Calculating forces
    # (30, 6) = (30, 6, 6) @ (30, 6, 36) @ (36) + (30, 6, 6) @ (30, 6, 36) @ (36) + (30, 6) * (30, 6, 36) @ (36)
    vDot = jacobian @ qDotDot + jacobianDot @ qDot # (30, 6)
    rootJacobian = jacobian[:, :, 0:6] # (30, 6, 6)
    inertiaForces = -(I @ vDot[:, :, None])[:, :, 0] # (30, 6)
    centroidalMomentum = (inertiaForces[:, None, :] @ rootJacobian).sum(0)[0] # (6,)
    gravityMoment = (FGrav[:, None, :] @ rootJacobian).sum(0)[0] # (6,)

    # Equality constraints for Wrists
    leftJacobian = rootJacobian[iLeft] # (6, 6)
    rightJacobian = rootJacobian[iRight] # (6, 6)
    WA = np.concatenate((leftJacobian, rightJacobian), axis=0).transpose() # (6, 12)
    Wb = -gravityMoment - centroidalMomentum # (6,)
    WA = np.concatenate((WA, np.array([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), np.array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]])), axis=0)
    Wb = np.concatenate((Wb, [0, 0]))

    # Inequality constraints for Wrists
    mu = 0.4
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
    weightMoment = 1
    weightForce = 0.001
    cost = np.array([weightMoment, weightMoment, weightMoment, weightForce, weightForce, weightForce, weightMoment, weightMoment, weightMoment, weightForce, weightForce, weightForce]) # Minimize the moments
    res = solvers.qp(matrix(np.diag(cost)), matrix(np.zeros_like(cost)), matrix(BA), matrix(Bb), matrix(WA), matrix(Wb))
    if res["status"] == "optimal":
        wrists = np.array(res["x"])[:, 0]
    else:
        print("Infeasible")
        pass
    Fwrists = np.zeros((30, 6))
    Fwrists[iLeft] = wrists[:6]
    Fwrists[iRight] = wrists[6:]
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

    while True:
        p.setJointMotorControlArray(atlas, np.arange(30), p.TORQUE_CONTROL, forces=jointTorques)
        p.stepSimulation()
        time.sleep(dt * 0)
        t += dt
        break


