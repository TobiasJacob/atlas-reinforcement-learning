from typing import Any, List, Type
import quaternion
from stable_baselines3.common.vec_env.base_vec_env import VecEnvIndices
from atlasrl.motions.MotionReader import MotionReader
import os
import gym, gym.spaces, gym.utils, gym.utils.seeding
import numpy as np
import pybullet as p
from pybullet_utils.bullet_client import BulletClient
import pybullet_data
from stable_baselines3.common.vec_env import VecEnv

from pkg_resources import parse_version


class AtlasBulletVecEnv(VecEnv):
	metadata = {
		'render.modes': ['human', 'rgb_array'],
		'video.frames_per_second': 60
	}

	def __init__(self, numRobots: int, render: bool=False):
		super(AtlasBulletVecEnv, self).__init__(numRobots, gym.spaces.Box(low=-np.inf, high=np.inf, shape=(8 + 2 * 30,)), gym.spaces.Box(low=-1, high=1, shape=(30,)))
		self.numRobots = numRobots
		self.motionReader = MotionReader.readClip()
		self.isRender = render
		if self.isRender:
			self._p = BulletClient(connection_mode=p.GUI)
		else:
			self._p = BulletClient()
		self._p.setAdditionalSearchPath(pybullet_data.getDataPath())
		self._p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0)
		self.atlas = []
		for a in range(numRobots):
			self.atlas.append(self._p.loadURDF("data/atlas/atlas_v4_with_multisense.urdf", [-2 + 10 * a, 3, 2.5]))
			for i in range (self._p.getNumJoints(self.atlas[a])):
				self._p.setJointMotorControl2(self.atlas[a], i, p.POSITION_CONTROL, 0)
		self._p.loadURDF("plane.urdf", [0, 0, 0], useFixedBase=True)
		self._p.setTimeStep(1/30)
		self._p.setGravity(0,0,-9.81)
		self._p.resetDebugVisualizerCamera(cameraDistance=3, cameraYaw=45, cameraPitch=-30, cameraTargetPosition=self._p.getBasePositionAndOrientation(self.atlas[0])[0])
		self._p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
		self._p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 1)
		if os.path.exists("data/initialStateVecEnv.bullet"):
			self.initialState = self._p.restoreState(fileName="data/initialStateVecEnv.bullet")
		else:
			for _ in range(10):
				self._p.stepSimulation()
			self._p.saveBullet("data/initialStateVecEnv.bullet")
		self.initialState = self._p.saveState()
		self.timeDelta = self._p.getPhysicsEngineParameters()["fixedTimeStep"]
		self.cameraStates = [[45, -30], [0, -15], [90, -15]]
		self.activeI = 0
		self.time = 0
		self.lastDesiredAction = np.zeros((self.numRobots, 30))
		self.lastChosenAction = np.zeros((self.numRobots, 30))
		self.initialObs = self.getObs(0)

	def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
		"""Return attribute from vectorized environment (see base class)."""
		pass


	def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
		"""Set attribute inside vectorized environments (see base class)."""
		pass


	def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> List[Any]:
		"""Call instance methods of vectorized environments."""
		pass


	def env_is_wrapped(self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None) -> List[bool]:
		"""Check if worker environments are wrapped with a given wrapper"""
		return False

	def seed(self, seed=None):
		self.np_random, seed = gym.utils.seeding.np_random(seed)
		return [seed]

	def getObs(self, a: int):
		(pos, orn) = self._p.getBasePositionAndOrientation(self.atlas[a])
		self.time = 0
		return np.concatenate((pos - np.array([- 10 * a, 0, 0]), orn, [self.time], self.lastDesiredAction[a], self.lastChosenAction[a]))

	def reset(self):
		self._p.restoreState(self.initialState)
		obs = []
		self.time = 0
		for a in range(self.numRobots):
			obs.append(self.getObs(a))
		self.lastDones = [False] * self.numRobots
		return np.array(obs)

	def render(self, mode = "human", close=False):
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

	def step_async(self, actions):
		self.actions = actions

	def step_wait(self):
		actions = self.actions
		desiredState = self.motionReader.getState(self.time)
		desiredAction = desiredState.getAction()
		# action = desiredAction + action / 10.

		self._p.stepSimulation()
		obs = []
		rewards = []
		dones = []
		for a in range(self.numRobots):
			self._p.setJointMotorControlArray(self.atlas[a], np.arange(30), p.POSITION_CONTROL, actions[a], forces=[10000] * 30) #, positionGain=0, velocityGain=0)
			(pos, orn) = self._p.getBasePositionAndOrientation(self.atlas[a])

			# Action and action speed difference
			desiredDifference = desiredAction - self.lastDesiredAction[a]
			chosenDifference = actions[a] - self.lastChosenAction[a]
			reward = np.exp(-2 * np.square(desiredAction - actions[a]).mean())
			reward += np.exp(-0.1 * np.square(desiredDifference - chosenDifference).mean())
			self.lastDesiredAction[a] = desiredAction
			self.lastChosenAction[a] = actions[a]

			eulerDif = 2 * np.arccos(quaternion.as_float_array(desiredState.rootRotation.conjugate() * quaternion.from_float_array((orn[3], *orn[:3])))[0])
			reward += np.exp(-40 * eulerDif)
			reward += np.exp(-40 * np.square(desiredState.rootPosition - pos).mean())
			self.time += self.timeDelta
			done = self.time > 10
			if eulerDif > 60 / 180 * np.pi:
				done = True
				reward -= 1
			rewards.append(reward)
			dones.append(done or self.lastDones[a])
			if dones[a]:
				obs.append(self.initialObs)
			else:
				obs.append(self.getObs(a))
				
			self._p.resetDebugVisualizerCamera(cameraDistance=3, cameraYaw=self.cameraStates[self.activeI][0], cameraPitch=self.cameraStates[self.activeI][1], cameraTargetPosition=self._p.getBasePositionAndOrientation(self.atlas[a])[0])
		self.lastDones = dones
		return np.array(obs), np.array(rewards), np.array(dones), {}