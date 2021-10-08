from atlasrl.motions.QuaternionToAtlasEuler import convertQuaternionToAtlasEuler, convertQuaternionToAtlasEulerBack
from atlasrl.robots.Constants import convertActionSpaceToAngle, convertAngleToActionSpace
from dataclasses import dataclass
import dataclasses

import numpy as np
import quaternion
from quaternion.numba_wrapper import njit
from numba.experimental import jitclass

from atlasrl.robots.AtlasRemoteEnv import parameterNames

parameterIndex = {n: i for i, n in enumerate(parameterNames)}

# duration of frame in seconds (1D),
# root position (3D),
# root rotation (4D),
# chest rotation (4D),
# neck rotation (4D),
# right hip rotation (4D),
# right knee rotation (1D),
# right ankle rotation (4D),
# right shoulder rotation (4D),
# right elbow rotation (1D),
# left hip rotation (4D),
# left knee rotation (1D),
# left ankle rotation (4D),
# left shoulder rotation (4D),
# left elbow rotation (1D)

@dataclass
class MotionState:
    # Parsed
    deltaTime: float
    rootPosition:  np.ndarray
    rootRotation: quaternion.quaternion
    chestRotation: quaternion.quaternion
    neckRotation: quaternion.quaternion
    rightHipRotation: quaternion.quaternion
    rightKneeRotation: float
    rightAnkleRotation: quaternion.quaternion
    rightShoulderRotation: quaternion.quaternion
    rightElbowRotation: float
    leftHipRotation: quaternion.quaternion
    leftKneeRotation: float
    leftAnkleRotation: quaternion.quaternion
    leftShoulderRotation: quaternion.quaternion
    leftElbowRotation: float
    # Not parsed
    absoluteTime: float

    @classmethod
    def fromRow(cls, absoluteTime: float, row: np.ndarray) -> "MotionState":
        data = []
        parsedObjects = [f.type for f in dataclasses.fields(cls)[:-1]]
        i = 0
        for parsedObject in parsedObjects:
            if parsedObject is float:
                data.append(np.array(row[i]))
                i += 1
            elif parsedObject is np.ndarray:
                data.append(np.array(row[i:i+3]))
                i += 3
            elif parsedObject is quaternion.quaternion:
                data.append(quaternion.from_float_array(row[i:i+4]))
                i += 4
            else:
                raise NotImplementedError(f"Invalid dtype in MotionState: {parsedObject}")
            if i > len(row):
                raise Exception(f"Row does not have enough values, index {i}/{len(row)}: {row}")
        return cls(*data, absoluteTime)

    @classmethod
    def fromInterpolation(cls, state1: "MotionState", state2: "MotionState", alpha: float):
        return cls(
            (1 - alpha) * state1.deltaTime + alpha * state2.deltaTime,
            (1 - alpha) * state1.rootPosition + alpha * state2.rootPosition,
            quaternion.slerp_evaluate(state1.rootRotation, state2.rootRotation, alpha), 
            quaternion.slerp_evaluate(state1.chestRotation, state2.chestRotation, alpha), 
            quaternion.slerp_evaluate(state1.neckRotation, state2.neckRotation, alpha), 
            quaternion.slerp_evaluate(state1.rightHipRotation, state2.rightHipRotation, alpha), 
            (1 - alpha) * state1.rightKneeRotation + alpha * state2.rightKneeRotation,
            quaternion.slerp_evaluate(state1.rightAnkleRotation, state2.rightAnkleRotation, alpha), 
            quaternion.slerp_evaluate(state1.rightShoulderRotation, state2.rightShoulderRotation, alpha), 
            (1 - alpha) * state1.rightElbowRotation + alpha * state2.rightElbowRotation,
            quaternion.slerp_evaluate(state1.leftHipRotation, state2.leftHipRotation, alpha), 
            (1 - alpha) * state1.leftKneeRotation + alpha * state2.leftKneeRotation,
            quaternion.slerp_evaluate(state1.leftAnkleRotation, state2.leftAnkleRotation, alpha), 
            quaternion.slerp_evaluate(state1.leftShoulderRotation, state2.leftShoulderRotation, alpha), 
            (1 - alpha) * state1.leftElbowRotation + alpha * state2.leftElbowRotation,
            (1 - alpha) * state1.absoluteTime + alpha * state2.absoluteTime,
        )
    
    def getAngles(self) -> np.ndarray:
        angles = np.zeros((30,))

        leftShoulder = convertQuaternionToAtlasEulerBack(self.leftShoulderRotation)
        rightShoulder = convertQuaternionToAtlasEulerBack(self.rightShoulderRotation)
        angles[parameterIndex["l_arm_shx"]] = convertActionSpaceToAngle("l_arm_shx", -1)
        angles[parameterIndex["l_arm_shz"]] = leftShoulder[2]
        angles[parameterIndex["l_arm_ely"]] = convertActionSpaceToAngle("l_arm_shx", 1)
        angles[parameterIndex["l_arm_elx"]] = self.leftElbowRotation
        angles[parameterIndex["r_arm_shx"]] = convertActionSpaceToAngle("l_arm_shx", 1)
        angles[parameterIndex["r_arm_shz"]] = rightShoulder[2]
        angles[parameterIndex["r_arm_ely"]] = convertActionSpaceToAngle("l_arm_shx", -1)
        angles[parameterIndex["r_arm_elx"]] = self.leftElbowRotation

        backEuler = convertQuaternionToAtlasEulerBack(self.chestRotation)
        angles[parameterIndex["back_bkx"]] = backEuler[0]
        angles[parameterIndex["back_bky"]] = backEuler[1]
        angles[parameterIndex["back_bkz"]] = backEuler[2]

        leftHipEuler = convertQuaternionToAtlasEuler(self.leftHipRotation)
        rightHipEuler = convertQuaternionToAtlasEuler(self.rightHipRotation)
        angles[parameterIndex["l_leg_hpx"]] = leftHipEuler[0]
        angles[parameterIndex["l_leg_hpy"]] = leftHipEuler[1]
        angles[parameterIndex["l_leg_hpz"]] = leftHipEuler[2]
        angles[parameterIndex["r_leg_hpx"]] = rightHipEuler[0]
        angles[parameterIndex["r_leg_hpy"]] = rightHipEuler[1]
        angles[parameterIndex["r_leg_hpz"]] = rightHipEuler[2]

        angles[parameterIndex["l_leg_kny"]] = -self.leftKneeRotation
        angles[parameterIndex["r_leg_kny"]] = -self.rightKneeRotation

        leftAnkleEuler = convertQuaternionToAtlasEuler(self.leftAnkleRotation)
        rightAnkleEuler = convertQuaternionToAtlasEuler(self.rightAnkleRotation)
        angles[parameterIndex["l_leg_akx"]] = leftAnkleEuler[0]
        angles[parameterIndex["l_leg_aky"]] = leftAnkleEuler[1]
        angles[parameterIndex["r_leg_akx"]] = rightAnkleEuler[0]
        angles[parameterIndex["r_leg_aky"]] = rightAnkleEuler[1]
        return angles


    def getAction(self) -> np.ndarray:
        action = np.zeros((30,))

        leftShoulder = convertQuaternionToAtlasEulerBack(self.leftShoulderRotation)
        rightShoulder = convertQuaternionToAtlasEulerBack(self.rightShoulderRotation)
        action[parameterIndex["l_arm_shx"]] = -1
        action[parameterIndex["l_arm_shz"]] = convertAngleToActionSpace("l_arm_shz", leftShoulder[2])
        action[parameterIndex["l_arm_ely"]] = 1
        action[parameterIndex["l_arm_elx"]] = convertAngleToActionSpace("l_arm_elx", self.leftElbowRotation)
        action[parameterIndex["r_arm_shx"]] = 1
        action[parameterIndex["r_arm_shz"]] = convertAngleToActionSpace("r_arm_shz", rightShoulder[2])
        action[parameterIndex["r_arm_ely"]] = -1
        action[parameterIndex["r_arm_elx"]] = convertAngleToActionSpace("r_arm_elx", self.leftElbowRotation)

        backEuler = convertQuaternionToAtlasEulerBack(self.chestRotation)
        action[parameterIndex["back_bkx"]] = convertAngleToActionSpace("back_bkx", backEuler[0])
        action[parameterIndex["back_bky"]] = convertAngleToActionSpace("back_bky", backEuler[1])
        action[parameterIndex["back_bkz"]] = convertAngleToActionSpace("back_bkz", backEuler[2])

        leftHipEuler = convertQuaternionToAtlasEuler(self.leftHipRotation)
        rightHipEuler = convertQuaternionToAtlasEuler(self.rightHipRotation)
        action[parameterIndex["l_leg_hpx"]] = convertAngleToActionSpace("l_leg_hpx", leftHipEuler[0])
        action[parameterIndex["l_leg_hpy"]] = convertAngleToActionSpace("l_leg_hpy", leftHipEuler[1])
        action[parameterIndex["l_leg_hpz"]] = convertAngleToActionSpace("l_leg_hpz", leftHipEuler[2])
        action[parameterIndex["r_leg_hpx"]] = convertAngleToActionSpace("r_leg_hpx", rightHipEuler[0])
        action[parameterIndex["r_leg_hpy"]] = convertAngleToActionSpace("r_leg_hpy", rightHipEuler[1])
        action[parameterIndex["r_leg_hpz"]] = convertAngleToActionSpace("r_leg_hpz", rightHipEuler[2])

        action[parameterIndex["l_leg_kny"]] = convertAngleToActionSpace("l_leg_kny", -self.leftKneeRotation)
        action[parameterIndex["r_leg_kny"]] = convertAngleToActionSpace("r_leg_kny", -self.rightKneeRotation)

        leftAnkleEuler = convertQuaternionToAtlasEuler(self.leftAnkleRotation)
        rightAnkleEuler = convertQuaternionToAtlasEuler(self.rightAnkleRotation)
        action[parameterIndex["l_leg_akx"]] = convertAngleToActionSpace("l_leg_akx", leftAnkleEuler[0])
        action[parameterIndex["l_leg_aky"]] = convertAngleToActionSpace("l_leg_aky", leftAnkleEuler[1])
        action[parameterIndex["r_leg_akx"]] = convertAngleToActionSpace("r_leg_akx", rightAnkleEuler[0])
        action[parameterIndex["r_leg_aky"]] = convertAngleToActionSpace("r_leg_aky", rightAnkleEuler[1])
        return action
