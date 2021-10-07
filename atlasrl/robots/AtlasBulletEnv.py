import quaternion
from atlasrl.motions.MotionReader import MotionReader
import os
import gym, gym.spaces, gym.utils, gym.utils.seeding
import numpy as np
import pybullet as p
from pybullet_utils.bullet_client import BulletClient
import pybullet_data

from pkg_resources import parse_version
from time import sleep
import datetime

from torch.utils.tensorboard import SummaryWriter
from .Constants import parameterNames

class AtlasBulletEnv(gym.Env):
	metadata = {
		'render.modes': ['human', 'rgb_array'],
		'video.frames_per_second': 60
	}

	def __init__(self, render=False, controlFreq=30., simStepsPerControlStep=1):
		super().__init__()
		if render:
			self.logger = SummaryWriter(f"runs/{datetime.datetime.now()}")
			self.globalStep = 0
		self.motionReader = MotionReader.readClip()
		self.isRender = render
		self.simStepsPerControlStep = simStepsPerControlStep
		if self.isRender:
			self._p = BulletClient(connection_mode=p.GUI)
		else:
			self._p = BulletClient()
		self._p.setAdditionalSearchPath(pybullet_data.getDataPath())
		self._p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0)
		self.atlas = self._p.loadURDF("data/atlas/atlas_v4_with_multisense.urdf", [0, 0, 1.0])
		for i in range (self._p.getNumJoints(self.atlas)):
			self._p.setJointMotorControl2(self.atlas, i, p.POSITION_CONTROL, 0)
		self.plane = self._p.loadURDF("plane.urdf", [0, 0, 0], useFixedBase=True)
		self._p.setTimeStep(1/(controlFreq * simStepsPerControlStep))
		self._p.setGravity(0,0,-9.81)
		self.action_space = gym.spaces.Box(low=-1, high=1, shape=(30,))
		self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(13 + 2 * 30,))
		self._p.resetDebugVisualizerCamera(cameraDistance=3, cameraYaw=45, cameraPitch=-30, cameraTargetPosition=self._p.getBasePositionAndOrientation(self.atlas)[0])
		self._p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
		self._p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 1)
		self.initialState = self._p.saveState()
		self.timeDelta = self._p.getPhysicsEngineParameters()["fixedTimeStep"] * self.simStepsPerControlStep
		self.cameraStates = [[45, -30], [0, -15], [90, -15]]
		self.activeI = 0
		self.time = 0
		footJoints = ["l_leg_akx", "l_leg_aky", "l_leg_kny", "r_leg_akx", "r_leg_aky", "r_leg_kny"]
		self.footLinks = [parameterNames.index(j) for j in footJoints]

	def getObservation(self):
		(pos, orn) = self._p.getBasePositionAndOrientation(self.atlas)
		posSpeed, ornSpeed = self._p.getBaseVelocity(self.atlas)
		obs = np.concatenate((pos, p.getEulerFromQuaternion(orn), posSpeed, ornSpeed, [self.time], self.lastDesiredAction, self.lastChosenAction))
		return obs

	def seed(self, seed=None):
		self.np_random, seed = gym.utils.seeding.np_random(seed)
		return [seed]

	def reset(self):
		self._p.restoreState(self.initialState)
		self.time = 0
		self.lastDesiredAction = np.zeros(30)
		self.lastChosenAction = np.zeros(30)
		return self.getObservation()

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

	def step(self, action):
		desiredState = self.motionReader.getState(self.time)
		desiredAction = desiredState.getAction()
		action = desiredAction + action / 5.
		# action = action / 5.

		self._p.setJointMotorControlArray(self.atlas, np.arange(30), p.POSITION_CONTROL, action)#, forces=[10000] * 30) #, positionGain=0, velocityGain=0)
		(pos, orn) = self._p.getBasePositionAndOrientation(self.atlas)
		for _ in range(self.simStepsPerControlStep):
			self._p.stepSimulation()
			# sleep(self._p.getPhysicsEngineParameters()["fixedTimeStep"])
		# Action and action speed difference
		desiredDifference = desiredAction - self.lastDesiredAction
		chosenDifference = action - self.lastChosenAction
		actionDiff = np.square(desiredAction - action).mean()
		rewardAction = np.exp(-5 * actionDiff) # Too loose
		actionSpeedDiff = np.square(desiredDifference - chosenDifference).mean()
		rewardActionSpeed = np.exp(-5 * actionSpeedDiff)
		self.lastDesiredAction = desiredAction
		self.lastChosenAction = action

		eulerDif = 2 * np.arccos(quaternion.as_float_array(desiredState.rootRotation.conjugate() * quaternion.from_float_array((orn[3], *orn[:3])))[0])
		rewardGlobalRotDiff = np.exp(-10 * eulerDif)
		posDif = np.square(desiredState.rootPosition - pos).mean()
		rewardRootPosDiff = np.exp(-10 * posDif)  # To strict
		self.time += self.timeDelta
		done = self.time > 10

		robot_ground_contacts = self._p.getContactPoints(bodyA=self.atlas, bodyB=self.plane)

		for contact in robot_ground_contacts:
			if contact[3] not in self.footLinks:
				done = True
				break

			# rewardDead -= 1
		reward = 0.3 * rewardAction + 0.1 * rewardActionSpeed + 0.65 * rewardGlobalRotDiff + 0.1 * rewardRootPosDiff
		if np.isnan(reward):
			reward = 0
		reward = np.clip(reward, 0, 2) # Clip reward for unexpected nans
		if self.isRender:
			self.logger.add_scalar("rollout/rewardAction", rewardAction, self.globalStep)
			self.logger.add_scalar("rollout/rewardActionSpeed", rewardActionSpeed, self.globalStep)
			self.logger.add_scalar("rollout/rewardGlobalRotDiff", rewardGlobalRotDiff, self.globalStep)
			self.logger.add_scalar("rollout/rewardRootPosDiff", rewardRootPosDiff, self.globalStep)
			self.logger.add_scalar("rollout/reward", reward, self.globalStep)
			self.logger.add_scalar("rollout/eulerDif", eulerDif, self.globalStep)
			self.logger.add_scalar("rollout/posDif", posDif, self.globalStep)
			self.logger.add_scalar("rollout/actionDiff", actionDiff, self.globalStep)
			self.logger.add_scalar("rollout/actionSpeedDiff", actionSpeedDiff, self.globalStep)
			if done:
				self.logger.add_scalar("rollout/episodeLen", self.time, self.globalStep)
			self.globalStep += 1
		self._p.resetDebugVisualizerCamera(cameraDistance=3, cameraYaw=self.cameraStates[self.activeI][0], cameraPitch=self.cameraStates[self.activeI][1], cameraTargetPosition=self._p.getBasePositionAndOrientation(self.atlas)[0])
		return self.getObservation(), reward, done, {}
