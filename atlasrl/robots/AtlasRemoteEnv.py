import os
import gym, gym.spaces, gym.utils, gym.utils.seeding
import numpy as np
import pybullet as p
from pybullet_utils.bullet_client import BulletClient
import pybullet_data
import socket
from atlasrl.robots.Constants import parameterNames

from pkg_resources import parse_version

class AtlasRemoteEnv(gym.Env):
	metadata = {
		'render.modes': ['human', 'rgb_array'],
		'video.frames_per_second': 60
	}

	def __init__(self, render=False, address=("localhost", 15923)):
		self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		self.s.connect(address)
		self.action_space = gym.spaces.Box(low=-1, high=1, shape=(30,))
		self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(7,))

	def seed(self, seed=None):
		self.np_random, seed = gym.utils.seeding.np_random(seed)
		return [seed]

	def reset(self):
		return self.__parseSensors()

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
		obs = self.__parseSensors()
		return obs, None, False, {}

	def __parseSensors(self):
		resp = str(self.s.recv(4096), "ascii")
		# print(resp)
		sensors = resp.split("/")
		angles = sensors[0]
		angles = angles.split(",")
		angles = [a.split("=") for a in angles]
		angles = {k: float(v) for (k, v) in angles}
		angles = [angles[n] for n in parameterNames]
		centerOfMassFrame = sensors[1]
		centerOfMassFrame = centerOfMassFrame.split(",")
		centerOfMassFrame = [float(c) for c in centerOfMassFrame]
		cop = sensors[2]
		wristLeft = sensors[3]
		wristRight = sensors[4]
		print(angles, centerOfMassFrame)
		return (angles, centerOfMassFrame)
