import gym, gym.spaces, gym.utils, gym.utils.seeding
import numpy as np
import socket
from time import sleep

import quaternion

from atlasrl.motions.MotionReader import MotionReader
from atlasrl.robots.Constants import parameterNames

class AtlasRemoteEnv(gym.Env):
	metadata = {
		'render.modes': ['human', 'rgb_array'],
		'video.frames_per_second': 60
	}

	def __init__(self, render=False, address=("0.0.0.0", 15923), controlFreq=30.):
		self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		self.motionReader = MotionReader.readClip()
		while True:
			try:
				self.s.connect(address)
				break
			except ConnectionRefusedError:
				print("ConnectionRefusedError, retry")
				sleep(0.5)
		print("Connected")
		self.action_space = gym.spaces.Box(low=-1, high=1, shape=(30,))
		self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(136,))
		self.time = 0
		self.timeDelta = 1. / controlFreq

	def seed(self, seed=None):
		self.np_random, seed = gym.utils.seeding.np_random(seed)
		return [seed]

	def reset(self):
		return self.getObservation()[0]

	def render(self, mode = "human", close=False):
		if mode == "human":
			return
		if mode != "rgb_array":
			return np.array([])

		(_, _, px, _, _) = self._p.getCameraImage(width=480, height=480)
		rgb_array = np.array(px)
		return rgb_array

	def close(self):
		pass

	def step(self, action):
		msg = ""
		for (targetAngle, name) in zip(action, parameterNames):
			msg += name + "=" + str(targetAngle) + ","
		msg = msg[:-1] + "\n"
		print(msg)
		self.s.send(bytes(msg, "ascii"))
		obs = self.getObservation()[0]
		self.time += self.timeDelta
		return obs, None, False, {}

	def getObservation(self):
		# Wait for obs
		resp = str(self.s.recv(4096), "ascii")
		print(resp)

		# Message parsing
		sensors = resp.split("/")
		angles = sensors[0]
		if len(angles) > 0 and angles != "null":
			angles = angles.split(",")
			angles = [a.split("=") for a in angles]
			angles = {k: float(v) for (k, v) in angles}
			jointAngles = np.array([angles[n] for n in parameterNames])
		else:
			jointAngles = np.zeros(30)
		speeds = sensors[1]
		if len(speeds) > 0 and speeds != "null":
			speeds = speeds.split(",")
			speeds = [a.split("=") for a in speeds]
			speeds = {k: float(v) for (k, v) in speeds}
			jointSpeeds = np.array([speeds[n] for n in parameterNames])
		else:
			jointSpeeds = np.zeros(30)
		centerOfMassFrame = sensors[2]
		if len(centerOfMassFrame) > 0 and centerOfMassFrame != "null":
			centerOfMassFrame = centerOfMassFrame.split(",")
			pos = np.array([float(c) for c in centerOfMassFrame])
			orn = np.array([0, 0, 0, 1])
		else:
			(pos, orn) = (np.zeros(3), np.array([0, 0, 0, 1]))

		# Convert values
		posSpeed, ornSpeed = (np.zeros(3), np.zeros(3))
		orn = quaternion.from_float_array((orn[3], *orn[:3]))
		vecX = quaternion.rotate_vectors(orn, np.array([1, 0, 0]))
		vecY = quaternion.rotate_vectors(orn, np.array([0, 1, 0]))
		desiredState = self.motionReader.getState(self.time)
		desiredAngles = desiredState.getAngles()
		dT = 0.01
		nextDesiredState = self.motionReader.getState(self.time + dT)
		desiredJointSpeeds = (nextDesiredState.getAngles() - desiredAngles) / dT
		desiredBaseSpeed = (nextDesiredState.rootPosition - desiredState.rootPosition) / dT

		# Concat
		obs = np.concatenate((pos[2:3], vecX, vecY, posSpeed, desiredBaseSpeed, ornSpeed, jointAngles, jointSpeeds, desiredAngles, desiredJointSpeeds))
		return obs, desiredAngles, jointAngles, jointSpeeds, desiredJointSpeeds, posSpeed, desiredBaseSpeed, pos, orn, desiredState

	def __parseSensors(self):
		resp = str(self.s.recv(4096), "ascii")
		print(resp)
		cop = sensors[2]
		wristLeft = sensors[3]
		wristRight = sensors[4]
		print(angles, centerOfMassFrame)
		return (angles, centerOfMassFrame)
