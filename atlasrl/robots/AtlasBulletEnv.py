import os
import gym, gym.spaces, gym.utils, gym.utils.seeding
import numpy as np
import pybullet as p
from pybullet_utils.bullet_client import BulletClient
import pybullet_data

from pkg_resources import parse_version


class AtlasBulletEnv(gym.Env):
	metadata = {
		'render.modes': ['human', 'rgb_array'],
		'video.frames_per_second': 60
	}

	def __init__(self, render=False):
		self.isRender = render
		if self.isRender:
			self._p = BulletClient(connection_mode=p.GUI)
		else:
			self._p = BulletClient()
		self._p.setAdditionalSearchPath(pybullet_data.getDataPath())
		self._p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0)
		self.atlas = self._p.loadURDF("data/atlas/atlas_v4_with_multisense.urdf", [-2, 3, 2.5])
		for i in range (self._p.getNumJoints(self.atlas)):
			self._p.setJointMotorControl2(self.atlas, i, p.POSITION_CONTROL, 0)
		self._p.loadURDF("plane.urdf", [0, 0, 0], useFixedBase=True)
		# self._p.setGravity(0,0,-9.81)
		self.action_space = gym.spaces.Box(low=-1, high=1, shape=(30,))
		self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(7,))
		self._p.resetDebugVisualizerCamera(cameraDistance=3, cameraYaw=45, cameraPitch=-30, cameraTargetPosition=self._p.getBasePositionAndOrientation(self.atlas)[0])
		self._p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
		self._p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 1)
		if os.path.exists("data/initialState.bullet"):
			self.initialState = self._p.restoreState(fileName="data/initialState.bullet")
		else:
			for _ in range(300):
				self._p.stepSimulation()
			self._p.saveBullet("data/initialState.bullet")
		self.initialState = self._p.saveState()
		self.timeDelta = self._p.getPhysicsEngineParameters()["fixedTimeStep"]
		self.alpha = 0.05
		self.cameraStates = [[45, -30], [0, -15], [90, -15]]
		self.activeI = 0

	def seed(self, seed=None):
		self.np_random, seed = gym.utils.seeding.np_random(seed)
		return [seed]

	def reset(self):
		self._p.restoreState(self.initialState)
		(pos, orn) = self._p.getBasePositionAndOrientation(self.atlas)
		obs = np.concatenate((pos, orn))
		return obs

	def render(self, mode = "human", close=False):
		self._p.resetDebugVisualizerCamera(cameraDistance=3, cameraYaw=self.cameraStates[self.activeI][0], cameraPitch=self.cameraStates[self.activeI][1], cameraTargetPosition=self._p.getBasePositionAndOrientation(self.atlas)[0])
		
		keys = p.getKeyboardEvents()
		if keys.get(100) == 3:  #D
			self.activeI = (self.activeI + 1) % len(self.cameraStates)
		
		if mode == "human":
			return
		if mode != "rgb_array":
			return np.array([])

		(_, _, px, _, _) = self._p.getCameraImage(width=480, height=480)
		rgb_array = np.array(px)
		return rgb_array

	def close(self):
		self._p.disconnect()

	def step(self, action):
		self._p.stepSimulation()
		for (i, targetAngle) in enumerate(action):
			(currentAngle, currentVel, _, _) = self._p.getJointState(self.atlas, i)
			# newVel = currentVel + self.timeDelta * (targetAngle - currentAngle) * 0.5
			# newAngle = currentAngle + self.timeDelta * newVel
			# targetAngle /= (self.alpha) # Pink noise 1/f gain compensation
			filteredAngle = (self.alpha * targetAngle + (1 - self.alpha) * currentAngle)
			# newPos = self.timeDelta * np.clip(filteredAngle - currentAngle, -0.7, 0.7) + currentAngle # limit speed
			self._p.setJointMotorControl2(self.atlas, i, p.POSITION_CONTROL, targetAngle) #, positionGain=0, velocityGain=0)
		(pos, orn) = self._p.getBasePositionAndOrientation(self.atlas)
		obs = np.concatenate((pos, orn))
		reward = pos[1]
		return obs, reward, False, {}
