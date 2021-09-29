import os
import gym, gym.spaces, gym.utils, gym.utils.seeding
import numpy as np
import pybullet as p
from pybullet_utils.bullet_client import BulletClient
import pybullet_data

from pkg_resources import parse_version
from atlasrl.robots.Constants import gains, parameterNames


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
		self.atlas = self._p.loadURDF("data/atlas/atlas_v4_with_multisense.urdf", [-2, 3, 2.5])
		for i in range (self._p.getNumJoints(self.atlas)):
			self._p.setJointMotorControl2(self.atlas, i, p.POSITION_CONTROL, 0)
		self._p.loadURDF("plane.urdf", [0, 0, 0], useFixedBase=True)
		self._p.setGravity(0,0,-10)
		self.action_space = gym.spaces.Box(low=-1, high=1, shape=(30,))
		self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(7,))
		if os.path.exists("data/initialState.bullet"):
			self.initialState = self._p.restoreState(fileName="data/initialState.bullet")
		else:
			for _ in range(300):
				self._p.stepSimulation()
			self._p.saveBullet("data/initialState.bullet")
		self.initialState = self._p.saveState()
		self.stepsPerAction = 10
		self.timeDelta = self.stepsPerAction * self._p.getPhysicsEngineParameters()["fixedTimeStep"]
		self.alpha = 0.05
		self._p.setJointMotorControlArray(self.atlas, np.arange(30), p.VELOCITY_CONTROL, forces=np.zeros(30))
		self.gains = np.array([gains[str(p.getJointInfo(self.atlas, i)[1])[2:-1]] for i in range(30)])
		print(self._p.getPhysicsEngineParameters())
		self._p.setTimeStep(0.001)

	def seed(self, seed=None):
		self.np_random, seed = gym.utils.seeding.np_random(seed)
		return [seed]

	def reset(self):
		self._p.restoreState(self.initialState)
		(pos, orn) = self._p.getBasePositionAndOrientation(self.atlas)
		obs = np.concatenate((pos, orn))
		return obs

	def render(self, mode = "human", close=False):
		self._p.resetDebugVisualizerCamera(cameraDistance=3, cameraYaw=45, cameraPitch=-30, cameraTargetPosition=self._p.getBasePositionAndOrientation(self.atlas)[0])
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
		for _ in range(self.stepsPerAction):
			readings = self._p.getJointStates(self.atlas, np.arange(30))
			currentAngle = np.array([r[0] for r in readings])
			currentVel = np.array([r[1] for r in readings])
			forces = self.gains * (action - currentAngle) - 0.3 * currentVel
			forces *= 2
			print(forces)
			self._p.setJointMotorControlArray(self.atlas, np.arange(30), p.TORQUE_CONTROL, forces=forces) #, positionGain=0, velocityGain=0)
			self._p.stepSimulation()
		(pos, orn) = self._p.getBasePositionAndOrientation(self.atlas)
		obs = np.concatenate((pos, orn))
		reward = pos[1]
		self._p.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=148, cameraPitch=-9, cameraTargetPosition=np.array([0.36, 5.3, -0.62]) + pos)
		return obs, reward, False, {}
