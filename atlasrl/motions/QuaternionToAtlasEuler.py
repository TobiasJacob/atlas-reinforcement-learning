import quaternion
import numpy as np

def convertQuaternionToAtlasEuler(quat: quaternion.quaternion):
    # This one is R = R_y * R_x * R_z in mocap frame
    R = quaternion.as_rotation_matrix(quat)
    alpha_at_x = np.arctan2(-R[1, 2], np.sqrt(R[1, 0] * R[1, 0] + R[1, 1] * R[1, 1]))
    alpha_at_y = -np.arctan2(R[1, 0], R[1, 1])
    alpha_at_z = np.arctan2(R[0, 2], R[2, 2])
    return np.array((alpha_at_x, alpha_at_y, alpha_at_z))

def convertQuaternionToAtlasEulerBack(quat: quaternion.quaternion):
    # This one is R = R_x * R_z * R_y in mocap frame
    R = quaternion.as_rotation_matrix(quat)
    alpha_at_x = np.arctan2(R[2, 1], R[1, 1])
    alpha_at_y = -np.arctan2(-R[0, 1], np.sqrt(R[0, 0] * R[0, 0] + R[0, 2] * R[0, 2]))
    alpha_at_z = np.arctan2(R[0, 2], R[0, 0])
    return np.array((alpha_at_x, alpha_at_y, alpha_at_z))

# Enter the following in WolframAlpha
# RotationMatrix[\[Theta]_y, {0, 1, 0}].RotationMatrix[\[Theta]_x, {1, 0, 0}].RotationMatrix[\[Theta]_z, {0, 0, 1}]
# results in 
# {
#     {Cos[θ_y] Cos[θ_z] + Sin[θ_x] Sin[θ_y] Sin[θ_z], Cos[θ_z] Sin[θ_x] Sin[θ_y] - Cos[θ_y] Sin[θ_z], Cos[θ_x] Sin[θ_y]},
#     {Cos[θ_x] Sin[θ_z], Cos[θ_x] Cos[θ_z], -Sin[θ_x]},
#     {-Cos[θ_z] Sin[θ_y] + Cos[θ_y] Sin[θ_x] Sin[θ_z], Cos[θ_y] Cos[θ_z] Sin[θ_x] + Sin[θ_y] Sin[θ_z], Cos[θ_x] Cos[θ_y]}
# }

# And for the back:
# RotationMatrix[\[Theta]_x, {1, 0, 0}].RotationMatrix[\[Theta]_z, {0, 0, 1}].RotationMatrix[\[Theta]_y, {0, 1, 0}]
# {
#     {Cos[θ_y] Cos[θ_z], -Sin[θ_z], Cos[θ_z] Sin[θ_y]},
#     {Sin[θ_x] Sin[θ_y] + Cos[θ_x] Cos[θ_y] Sin[θ_z], Cos[θ_x] Cos[θ_z], -Cos[θ_y] Sin[θ_x] + Cos[θ_x] Sin[θ_y] Sin[θ_z]},
#     {-Cos[θ_x] Sin[θ_y] + Cos[θ_y] Sin[θ_x] Sin[θ_z], Cos[θ_z] Sin[θ_x], Cos[θ_x] Cos[θ_y] + Sin[θ_x] Sin[θ_y] Sin[θ_z]}
# }