
import numpy as np

parameterNames = [
    'back_bkz',
    'back_bky',
    'back_bkx',
    'l_arm_shz',
    'l_arm_shx',
    'l_arm_ely',
    'l_arm_elx',
    'l_arm_wry',
    'l_arm_wrx',
    'l_arm_wry2',
    'neck_ry',
    'r_arm_shz',
    'r_arm_shx',
    'r_arm_ely',
    'r_arm_elx',
    'r_arm_wry',
    'r_arm_wrx',
    'r_arm_wry2',
    'l_leg_hpz',
    'l_leg_hpx',
    'l_leg_hpy',
    'l_leg_kny',
    'l_leg_aky',
    'l_leg_akx',
    'r_leg_hpz',
    'r_leg_hpx',
    'r_leg_hpy',
    'r_leg_kny',
    'r_leg_aky',
    'r_leg_akx'
]

controllerGains = {
    "l_leg_aky": 500.0,
    "r_leg_aky": 500.0,
    "l_leg_akx": 100.0,
    "r_leg_akx": 100.0,
    "l_leg_kny": 500.0,
    "r_leg_kny": 500.0,
    "l_leg_hpz": 100.0,
    "r_leg_hpz": 100.0,
    "l_leg_hpx": 300.0,
    "r_leg_hpx": 300.0,
    "l_leg_hpy": 300.0,
    "r_leg_hpy": 300.0,
    "back_bkz": 500.0,
    "back_bky": 500.0,
    "back_bkx": 500.0,
    "r_arm_shz": 100.0,
    "l_arm_shx": 100.0,
    "r_arm_shx": 100.0,
    "l_arm_ely": 100.0,
    "r_arm_ely": 100.0,
    "l_arm_elx": 100.0,
    "r_arm_elx": 100.0,
    "l_arm_wry": 10.0,
    "r_arm_wry": 10.0,
    "l_arm_wrx": 10.0,
    "r_arm_wrx": 10.0,
    "l_arm_wry2": 10.0,
    "r_arm_wry2": 10.0,
    "l_arm_shz": 10.0,
    "neck_ry": 10.0,
}


gainArray = []
for k in parameterNames:
    gainArray.append(controllerGains[k])

JOINT_LIMITS_PYBULLET = {
    "back_bkz": (-0.663225, 0.663225),
    "back_bky": (-0.219388, 0.538783),
    "back_bkx": (-0.523599, 0.523599),
    "l_arm_shz": (-1.5708, 0.785398),
    "l_arm_shx": (-1.5708, 1.5708),
    "l_arm_ely": (0.0, 3.14159),
    "l_arm_elx": (0.0, 2.35619),
    "l_arm_wry": (0.0, 3.14159),
    "l_arm_wrx": (-1.1781, 1.1781),
    "l_arm_wry2": (-0.001, 0.001),
    "neck_ry": (-0.602139, 1.14319),
    "r_arm_shz": (-0.785398, 1.5708),
    "r_arm_shx": (-1.5708, 1.5708),
    "r_arm_ely": (0.0, 3.14159),
    "r_arm_elx": (-2.35619, 0.0),
    "r_arm_wry": (0.0, 3.14159),
    "r_arm_wrx": (-1.1781, 1.1781),
    "r_arm_wry2": (-0.001, 0.001),
    "l_leg_hpz": (-0.174358, 0.786794),
    "l_leg_hpx": (-0.523599, 0.523599),
    "l_leg_hpy": (-1.61234, 0.65764),
    "l_leg_kny": (0.0, 2.35637),
    "l_leg_aky": (-1.0, 0.7),
    "l_leg_akx": (-0.8, 0.8),
    "r_leg_hpz": (-0.786794, 0.174358),
    "r_leg_hpx": (-0.523599, 0.523599),
    "r_leg_hpy": (-1.61234, 0.65764),
    "r_leg_kny": (0.0, 2.35637),
    "r_leg_aky": (-1.0, 0.7),
    "r_leg_akx": (-0.8, 0.8)
}

def convertAngleToActionSpace(jointName: str, angle: float):
    limits = JOINT_LIMITS_PYBULLET[jointName]
    angle = (angle + np.pi) % (2 * np.pi) - np.pi

    if angle >= 0:
        action = angle / (limits[1] + 1e-7)
    else:
        action = -angle / (limits[0] + 1e-7)

    return np.clip(action, -1, 1)

def convertActionSpaceToAngle(jointName: str, action: float):
    limits = JOINT_LIMITS_PYBULLET[jointName]
    if action >= 0:
        angle = action * limits[1]
    else:
        angle = -action * limits[0]
    return np.clip(angle, limits[0], limits[1])

def convertActionsToAngle(actions: np.ndarray):
    angles = np.zeros(30)
    for i, k in enumerate(parameterNames):
        angles[i] = convertActionSpaceToAngle(k, actions[i])

    return angles
