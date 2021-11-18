import quaternion
import numpy as np

def convertQuaternionToAtlasEuler(quat: quaternion.quaternion):
    # This one is R = R_y * R_x * R_z in pybullet frame, which is R = R_-z * R_+x * R_+y
    R = quaternion.as_rotation_matrix(quat)
    alpha_at_x = np.arctan2(R[2, 1], np.sqrt(R[0, 1] * R[0, 1] + R[1, 1] * R[1, 1]))
    alpha_at_y = np.arctan2(-R[2, 0], R[2, 2])
    alpha_at_min_z = np.arctan2(R[0, 1], R[1, 1])
    return np.array((alpha_at_x, alpha_at_min_z, alpha_at_y))

def convertQuaternionToAtlasEulerBack(quat: quaternion.quaternion):
    # This one is R = R_x * R_z * R_y in pybullet frame, which is R = R_+x * R_+y * R_-z
    R = quaternion.as_rotation_matrix(quat)
    alpha_at_x = np.arctan2(-R[2, 1], R[2, 2])
    alpha_at_y = -np.arctan2(R[0, 2], np.sqrt(R[0, 0] * R[0, 0] + R[0, 1] * R[0, 1]))
    alpha_at_min_z = np.arctan2(R[0, 1], R[0, 0])
    return np.array((alpha_at_x, alpha_at_min_z, alpha_at_y))

# Enter the following in WolframAlpha
# RotationMatrix[-\[Theta]_z, {0, 0, 1}].RotationMatrix[\[Theta]_x, {1, 0, 0}].RotationMatrix[\[Theta]_y, {0, 1, 0}]
# results in 
# {
#     {Cos[θ_y] Cos[θ_z] + Sin[θ_x] Sin[θ_y] Sin[θ_z], Cos[θ_x] Sin[θ_z], Cos[θ_z] Sin[θ_y] - Cos[θ_y] Sin[θ_x] Sin[θ_z]},
#     {Cos[θ_z] Sin[θ_x] Sin[θ_y] - Cos[θ_y] Sin[θ_z], Cos[θ_x] Cos[θ_z], -Cos[θ_y] Cos[θ_z] Sin[θ_x] - Sin[θ_y] Sin[θ_z]},
#     {-Cos[θ_x] Sin[θ_y], Sin[θ_x], Cos[θ_x] Cos[θ_y]}
# }

# And for the back:
# RotationMatrix[\[Theta]_x, {1, 0, 0}].RotationMatrix[\[Theta]_y, {0, 1, 0}].RotationMatrix[-\[Theta]_z, {0, 0, 1}]
# {
#     {Cos[θ_y] Cos[θ_z], Cos[θ_y] Sin[θ_z], Sin[θ_y]},
#     {Cos[θ_z] Sin[θ_x] Sin[θ_y] - Cos[θ_x] Sin[θ_z], Cos[θ_x] Cos[θ_z] + Sin[θ_x] Sin[θ_y] Sin[θ_z], -Cos[θ_y] Sin[θ_x]},
#     {-Cos[θ_x] Cos[θ_z] Sin[θ_y] - Sin[θ_x] Sin[θ_z], Cos[θ_z] Sin[θ_x] - Cos[θ_x] Sin[θ_y] Sin[θ_z], Cos[θ_x] Cos[θ_y]}
# }
