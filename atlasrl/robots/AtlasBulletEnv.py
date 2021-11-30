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
from queue import Queue

from torch.utils.tensorboard import SummaryWriter
from .Constants import convertActionSpaceToAngle, convertActionsToAngle, parameterNames, gainArray, dampingArray

class AtlasBulletEnv(gym.Env):
	metadata = {
		'render.modes': ['human', 'rgb_array'],
		'video.frames_per_second': 60
	}

	def __init__(self, render=False, controlFreq=30., simStepsPerControlStep=30):
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
		self.atlas = self._p.loadURDF("data/atlas/atlas_v4_with_multisense.urdf", [0, 0, 0.92])
		self.plane = self._p.loadURDF("plane.urdf", [0, 0, 0], useFixedBase=True)
		self._p.setTimeStep(1/(controlFreq * simStepsPerControlStep))
		self._p.setGravity(0,0,-9.81)
		self.action_space = gym.spaces.Box(low=-1, high=1, shape=(30,))
		self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(136,))
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
		for j in range(-1, 30):
			self._p.changeDynamics(self.atlas, j, linearDamping=0, angularDamping=0, jointDamping=0, restitution=1)
		self._p.setJointMotorControlArray(self.atlas, np.arange(30), p.POSITION_CONTROL, forces=np.zeros(30,))#, forces=[10000] * 30) #, positionGain=0, velocityGain=0)

	def getObservation(self):
		(pos, orn) = self._p.getBasePositionAndOrientation(self.atlas)
		posSpeed, ornSpeed = self._p.getBaseVelocity(self.atlas)
		orn = quaternion.from_float_array((orn[3], *orn[:3]))
		vecX = quaternion.rotate_vectors(orn, np.array([1, 0, 0]))
		vecY = quaternion.rotate_vectors(orn, np.array([0, 1, 0]))
		jointAngles, jointSpeeds = self.getJointAnglesAndSpeeds()
		desiredState = self.motionReader.getState(self.time)
		desiredAngles = desiredState.getAngles()
		dT = 0.01
		nextDesiredState = self.motionReader.getState(self.time + dT)
		desiredJointSpeeds = (nextDesiredState.getAngles() - desiredAngles) / dT
		desiredBaseSpeed = (nextDesiredState.rootPosition - desiredState.rootPosition) / dT
		obs = np.concatenate((pos[2:3], vecX, vecY, posSpeed, desiredBaseSpeed, ornSpeed, jointAngles, jointSpeeds, desiredAngles, desiredJointSpeeds))
		return obs, desiredAngles, jointAngles, jointSpeeds, desiredJointSpeeds, posSpeed, desiredBaseSpeed, pos, orn, desiredState

	def getJointAnglesAndSpeeds(self):
		jointAngles = np.zeros((30,))
		jointSpeeds = np.zeros((30,))
		for i in range(30):
			(currentAngle, currentVel, _, _) = self._p.getJointState(self.atlas, i)
			jointAngles[i] = currentAngle
			jointSpeeds[i] = currentVel
		return jointAngles,jointSpeeds

	def seed(self, seed=None):
		self.np_random, seed = gym.utils.seeding.np_random(seed)
		return [seed]

	def reset(self, randomStartPosition=False):
		self._p.restoreState(self.initialState)
		if randomStartPosition:
			# Setting atlas to random initial position with noise
			self.time = np.random.rand() * self.motionReader.frames[-1].absoluteTime
			# motionState = self.motionReader.getState(self.time)
			# angles = motionState.getAngles() + np.random.normal(size=30) * 0.3
			# targetPos, targetOrn = motionState.rootPosition, motionState.rootRotation
			# targetOrnAsArray = quaternion.as_float_array(targetOrn)
			# for i in range(30):
			# 	self._p.resetJointState(self.atlas, i, angles[i])
			self.gainArray = gainArray * (1 + np.random.normal(size=30) * 0.1)
			self.dampingArray = dampingArray * (1 + np.random.normal(size=30) * 0.1)
			# self.latencyQueue = Queue(np.random.randint(1, 5))
			self.latencyQueue = Queue(1)
			for i in range(-1, 30):
				info = self._p.getDynamicsInfo(self.atlas, i)
				mass = info[0] * (1 + np.random.normal() * 0.1)
				lateralFriction = info[1] * (1 + np.random.normal() * 0.1)
				spinningFriction = info[7] * (1 + np.random.normal() * 0.1)
				self._p.changeDynamics(self.atlas, -1, mass=mass, lateralFriction=lateralFriction, spinningFriction=spinningFriction)
			# self._p.resetBasePositionAndOrientation(self.atlas, np.array([targetPos[0], targetPos[1], 1]), [*targetOrnAsArray[1:4], targetOrnAsArray[0]])
			(aabbR, _) = self._p.getAABB(self.atlas, parameterNames.index("r_leg_aky"))
			(aabbL, _) = self._p.getAABB(self.atlas, parameterNames.index("l_leg_aky"))
			self._p.resetBasePositionAndOrientation(self.atlas, np.array([0, 0, 1 - min(aabbL[2], aabbR[2]) + np.random.exponential(0.03) + 0.1]), [0, 0, 0, 1])
			# self._p.resetBasePositionAndOrientation(self.atlas, np.array([targetPos[0], targetPos[1], 1 - min(aabbL[2], aabbR[2]) + np.random.exponential(0.03) + 0.01]), [*targetOrnAsArray[1:4], targetOrnAsArray[0]])
		else:
			self.time = 0
			self._p.resetBasePositionAndOrientation(self.atlas, np.array([0, 0, 0.95]), [0, 0, 0, 1])

			for i in range(30):
				self._p.resetJointState(self.atlas, i, 0)
			self.gainArray = gainArray * (1 + np.random.normal(size=30) * 0.1)
			self.dampingArray = dampingArray * (1 + np.random.normal(size=30) * 0.1)
			self.latencyQueue = Queue(1)
			
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
		self.latencyQueue.put(action)
		if self.latencyQueue.qsize() == self.latencyQueue.maxsize:
			action = self.latencyQueue.get()
			# Execute action
			desiredAngles = convertActionsToAngle(action)

			# Step simulation
			randomForce = np.random.normal(size=3) * 10
			for _ in range(self.simStepsPerControlStep):
				jointAngles, jointSpeeds = self.getJointAnglesAndSpeeds()
				torques = self.gainArray * (desiredAngles - jointAngles) - self.dampingArray * jointSpeeds
				self._p.setJointMotorControlArray(self.atlas, np.arange(30), p.TORQUE_CONTROL, forces=torques)#, forces=[10000] * 30) #, positionGain=0, velocityGain=0)
				self._p.applyExternalForce(self.atlas, -1, randomForce, np.zeros(3), p.WORLD_FRAME)
				self._p.stepSimulation()
				# sleep(self._p.getPhysicsEngineParameters()["fixedTimeStep"])
		self._p.resetDebugVisualizerCamera(cameraDistance=3, cameraYaw=self.cameraStates[self.activeI][0], cameraPitch=self.cameraStates[self.activeI][1], cameraTargetPosition=self._p.getBasePositionAndOrientation(self.atlas)[0])
		self.time += self.timeDelta

		# Observe
		obs, desiredAngles, jointAngles, jointSpeeds, desiredJointSpeeds, posSpeed, desiredBaseSpeed, pos, orn, desiredState = self.getObservation()

		# Calculate Reward
		jointDiff = np.square(desiredAngles - jointAngles).mean()
		rewardJoint = np.exp(-10 * jointDiff)
		jointSpeedDiff = np.square(jointSpeeds - desiredJointSpeeds).mean()
		rewardJointSpeed = np.exp(-1.5 * jointSpeedDiff)
		rotationDif = 2 * np.arccos(quaternion.as_float_array(desiredState.rootRotation.inverse() * orn)[0] - 1e-5)
		rewardGlobalRotDiff = np.exp(-10 * rotationDif)
		posDif = np.square(desiredState.rootPosition[2] - pos[2]).mean()
		rewardRootPosDiff = np.exp(-2 * posDif)
		rootSpeedDif = np.square(posSpeed - desiredBaseSpeed).mean()
		rewardRootSpeedDif = np.exp(-2 * rootSpeedDif)

		reward = 0.4 * rewardJoint + 0.1 * rewardJointSpeed + 0.3 * rewardGlobalRotDiff + 0.1 * rewardRootPosDiff + 0.1 * rewardRootSpeedDif
		if np.isnan(reward):
			reward = 0

		# Check for termination because of ground contacts
		done = False
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
			self.logger.add_scalar("rollout/rewardRootSpeedDif", rewardRootSpeedDif, self.globalStep)
			self.logger.add_scalar("rollout/reward", reward, self.globalStep)
			self.logger.add_scalar("rollout/eulerDif", rotationDif, self.globalStep)
			self.logger.add_scalar("rollout/posDif", posDif, self.globalStep)
			self.logger.add_scalar("rollout/jointDiff", jointDiff, self.globalStep)
			self.logger.add_scalar("rollout/jointSpeedDiff", jointSpeedDiff, self.globalStep)
			if done:
				self.logger.add_scalar("rollout/episodeLen", self.time, self.globalStep)
			self.globalStep += 1

		return obs, reward, done, {"orn": orn}
