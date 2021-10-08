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
		self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(133,))
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
		orn = quaternion.from_float_array((orn[3], *orn[:3]))
		vecX = quaternion.rotate_vectors(orn, np.array([1, 0, 0]))
		vecY = quaternion.rotate_vectors(orn, np.array([0, 1, 0]))
		jointAngles = []
		jointSpeeds = []
		for i in range(30):
			(currentAngle, currentVel, _, _) = self._p.getJointState(self.atlas, i)
			jointAngles.append(currentAngle)
			jointSpeeds.append(currentVel)
		desiredState = self.motionReader.getState(self.time)
		desiredAngles = desiredState.getAngles()
		dT = 0.01
		nextDesiredState = self.motionReader.getState(self.time + dT)
		nextDesiredAngles = nextDesiredState.getAngles()
		desiredJointSpeeds = (nextDesiredAngles - desiredAngles) / dT
		obs = np.concatenate((pos[2:3], vecX, vecY, posSpeed, ornSpeed, jointAngles, jointSpeeds, desiredAngles, desiredJointSpeeds))
		return obs, desiredAngles, jointAngles, jointSpeeds, desiredJointSpeeds

	def seed(self, seed=None):
		self.np_random, seed = gym.utils.seeding.np_random(seed)
		return [seed]

	def reset(self):
		self._p.restoreState(self.initialState)
		# Setting atlas to random initial position with noise
		self.time = np.random.rand() * self.motionReader.frames[-1].absoluteTime
		motionState = self.motionReader.getState(self.time)
		action = motionState.getAction() + np.random.normal(30) * 0.1
		targetPos, targetOrn = motionState.rootPosition, motionState.rootRotation
		targetOrnAsArray = quaternion.as_float_array(targetOrn)
		self._p.resetBasePositionAndOrientation(self.atlas, targetPos + np.array([0, 0, 1]), [*targetOrnAsArray[1:4], targetOrnAsArray[0]])
		self._p.setJointMotorControlArray(self.atlas, np.arange(30), p.POSITION_CONTROL, action)
		return self.getObservation()[0]

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
		# Get action in residual style
		desiredState = self.motionReader.getState(self.time)
		desiredAction = desiredState.getAction()
		action = desiredAction + action / 5.
		# action = action / 5.

		# Step simulation
		self._p.setJointMotorControlArray(self.atlas, np.arange(30), p.POSITION_CONTROL, action)#, forces=[10000] * 30) #, positionGain=0, velocityGain=0)
		(pos, orn) = self._p.getBasePositionAndOrientation(self.atlas)
		for _ in range(self.simStepsPerControlStep):
			self._p.stepSimulation()
			# sleep(self._p.getPhysicsEngineParameters()["fixedTimeStep"])
		self._p.resetDebugVisualizerCamera(cameraDistance=3, cameraYaw=self.cameraStates[self.activeI][0], cameraPitch=self.cameraStates[self.activeI][1], cameraTargetPosition=self._p.getBasePositionAndOrientation(self.atlas)[0])
		self.time += self.timeDelta

		# Observe
		obs, desiredAngles, jointAngles, jointSpeeds, desiredJointSpeeds = self.getObservation()
	
		# Calculate Reward
		jointDiff = np.square(desiredAngles - jointAngles).mean()
		rewardJoint = np.exp(-10 * jointDiff)
		jointSpeedDiff = np.square(jointSpeeds - desiredJointSpeeds).mean()
		rewardJointSpeed = np.exp(-5 * jointSpeedDiff)
		rotationDif = 2 * np.arccos(quaternion.as_float_array(desiredState.rootRotation.inverse() * quaternion.from_float_array((orn[3], *orn[:3])))[0] - 1e-5)
		rewardGlobalRotDiff = np.exp(-10 * rotationDif)
		posDif = np.square(desiredState.rootPosition - pos).mean()
		rewardRootPosDiff = np.exp(-2 * posDif)
		done = self.time > 10

		reward = 0.3 * rewardJoint + 0.1 * rewardJointSpeed + 0.65 * rewardGlobalRotDiff + 0.1 * rewardRootPosDiff
		if np.isnan(reward):
			reward = 0
		
		# Check for termination because of ground contacts
		robot_ground_contacts = self._p.getContactPoints(bodyA=self.atlas, bodyB=self.plane)
		for contact in robot_ground_contacts:
			if contact[3] not in self.footLinks:
				done = True
				break

		# Logging
		if self.isRender:
			self.logger.add_scalar("rollout/rewardJoint", rewardJoint, self.globalStep)
			self.logger.add_scalar("rollout/rewardJointSpeed", rewardJointSpeed, self.globalStep)
			self.logger.add_scalar("rollout/rewardGlobalRotDiff", rewardGlobalRotDiff, self.globalStep)
			self.logger.add_scalar("rollout/rewardRootPosDiff", rewardRootPosDiff, self.globalStep)
			self.logger.add_scalar("rollout/reward", reward, self.globalStep)
			self.logger.add_scalar("rollout/eulerDif", rotationDif, self.globalStep)
			self.logger.add_scalar("rollout/posDif", posDif, self.globalStep)
			self.logger.add_scalar("rollout/jointDiff", jointDiff, self.globalStep)
			self.logger.add_scalar("rollout/jointSpeedDiff", jointSpeedDiff, self.globalStep)
			if done:
				self.logger.add_scalar("rollout/episodeLen", self.time, self.globalStep)
			self.globalStep += 1

		return obs, reward, done, {}
