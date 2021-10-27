"""This is a standalone script to explort how the Atlas model looks like."""
import numpy as np
import pybullet as p
import pybullet_data
import time,math

path = "data/atlas/atlas_v4_with_multisense.urdf"

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0)
atlas = p.loadURDF(path, [-2,3,2.5])
# atlas = p.loadSDF(path)[0]
for i in range (p.getNumJoints(atlas)):
    p.setJointMotorControl2(atlas,i,p.POSITION_CONTROL,0)
    info = p.getJointInfo(atlas,i)
    print("Joint info", i, info)

p.loadURDF("plane.urdf",[0,0,0], useFixedBase=True)

p.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=148, cameraPitch=-9, cameraTargetPosition=[0.36,5.3,-0.62])

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
                x = 10.*math.sin(t)
                y = 10.*math.cos(t)
                p.getCameraImage(320,200,lightDirection=[x,y,10],shadow=1)#, renderer=p.ER_BULLET_HARDWARE_OPENGL )

    for i in range(p.getNumJoints(atlas)):
        val = p.readUserDebugParameter(action_selector_ids[i])
        p.setJointMotorControl2(atlas,i,p.POSITION_CONTROL,val) # , positionGain=5)
        p.resetBasePositionAndOrientation(atlas, np.array([0, 0, 1]), [0, 0, 0, 1])

