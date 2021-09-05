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
			self._p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
		else:
			self._p = BulletClient()
		self._p.setAdditionalSearchPath(pybullet_data.getDataPath())
		self.atlas = self._p.loadURDF("data/atlas/atlas_v4_with_multisense.urdf", [-2, 3, -0.5])
		for i in range (self._p.getNumJoints(self.atlas)):
			self._p.setJointMotorControl2(self.atlas, i, p.POSITION_CONTROL, 0)
		self._p.loadURDF("plane.urdf", [0, 0, -3])
		self._p.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=148, cameraPitch=-9, cameraTargetPosition=[0.36, 5.3, -0.62])
		self._p.setGravity(0,0,-10)
		self.initialState = self._p.saveState()
		self.action_space = gym.spaces.Box(low=0, high=1, shape=(30,))
		self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(6,))

	def seed(self, seed=None):
		self.np_random, seed = gym.utils.seeding.np_random(seed)
		return [seed]

	def reset(self):
		self._p.restoreState(self.initialState)
		return None

	def render(self, mode, close=False):
		if mode == "human":
			return
		if mode != "rgb_array":
			return np.array([])

		(_, _, px, _, _) = self._p.getCameraImage(width=480, height=480)
		rgb_array = np.array(px)
		rgb_array = rgb_array[:, :, :3]
		return rgb_array

	def close(self):
		self._p.disconnect()
	
	def step(self, action):
		self._p.stepSimulation()
		return None, 1, False, None
