"""This is a standalone script to explort how the Atlas model looks like."""
import numpy as np
import pybullet as p
import pybullet_data
import time,math

path = "data/atlas/atlas_v4_with_multisense.urdf"


np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=np.inf)
np.set_printoptions(precision=4)
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0)
atlas = p.loadURDF(path, [0, 0, 0.95])
# atlas = p.loadSDF(path)[0]
for i in range (p.getNumJoints(atlas)):
    p.setJointMotorControl2(atlas,i,p.POSITION_CONTROL,0)
    print(i, " Joint info", p.getJointInfo(atlas,i))
    print(i, " Dynamics info", p.getDynamicsInfo(atlas,i))
    print(i, " Link state", p.getLinkState(atlas, i))
    print(i, " Joint state", p.getJointState(atlas, i))

p.loadURDF("plane.urdf",[0,0,0], useFixedBase=True)

p.resetDebugVisualizerCamera(cameraDistance=3, cameraYaw=70, cameraPitch=-20, cameraTargetPosition=[0, 0, 0.5])

p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1)

p.getCameraImage(320,200)#, renderer=p.ER_BULLET_HARDWARE_OPENGL )

print("Num joints", p.getNumJoints(atlas))
action_selector_ids = []
for i in range(p.getNumJoints(atlas)):
    info = p.getJointInfo(atlas,i)
    print(info[8:10])
    action_selector_id = p.addUserDebugParameter(paramName=str(p.getJointInfo(atlas, i)[1]),
                                                 rangeMin=-1,
                                                 rangeMax=1,
                                                 startValue=0)
    action_selector_ids.append(action_selector_id)

t=0
p.setRealTimeSimulation(1)
while (1):
    p.setGravity(0,0,-10)
    time.sleep(0.01)
    t+=0.01
    keys = p.getKeyboardEvents()
    for k in keys:
        if (keys[k]&p.KEY_WAS_TRIGGERED):
            if (k == ord('i')):
                jointStates = p.getJointStates(atlas, np.arange(30))
                jointPositions = [j[0] for j in jointStates]
                qJoints = np.array(jointPositions)
                zero_vec = [0.0] * len(qJoints)
                i_leg_akx = 28
                foot_loc = p.getLinkState(atlas, i_leg_akx)[0]
                jac_t_2, jac_r_2 = p.calculateJacobian(atlas, i_leg_akx, foot_loc, qJoints.tolist(), zero_vec, zero_vec)
                print(np.array(jac_t_2)[:, :6])
                print(np.array(jac_t_2)[:, 6:])

    for i in range(p.getNumJoints(atlas)):
        val = p.readUserDebugParameter(action_selector_ids[i])
        p.setJointMotorControl2(atlas,i,p.POSITION_CONTROL,val) # , positionGain=5)
        p.resetBasePositionAndOrientation(atlas, np.array([0, 0, 1]), [0, 0, 0, 1])
