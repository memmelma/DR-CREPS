
import numpy as np
import time

import pybullet as p
import pybullet_data

import gym
from gym import spaces
from gym.utils import seeding
gym.logger.set_level(40)

from custom_env import darias

largeValObservation = 100

from mushroom_rl.environments import Environment, MDPInfo
from mushroom_rl.utils.spaces import *


class BallRollingGym(Environment):
    """
    Interface for Ball Rolling Gym environment.

    """
    def __init__(self, horizon=1000, gamma=0.99, observation_ids=[0,1,2,3], render=False,
                 **env_args):
        """
        Constructor.

        Args:
             horizon (int): the horizon;
             gamma (float): the discount factor;
             observation_ids (list(int)): oberservations to be returned,
                                0: (jointsPos, jointsVel), 1: (pBall, oBall), 
                                2: (lVelBall, aVelBall), 3: (handState)
        """
        self._first = True

        # MDP creation
        self.env = BallRollingEnv(
                     observation_ids=observation_ids,
                     renders=render,
                     seed=None,
                     isDiscrete=False,
                     actionRepeat=1,
                     maxSteps=horizon,
                     mode=darias.R_ARM,
                     is_deterministic_env = True)
        
        # required for self.env.observation_space to be set
        self.env.reset()

        self.env._max_episode_steps = np.inf  # Hack to ignore gym time limit.

        # MDP properties
        assert not isinstance(self.env.observation_space,
                              spaces.MultiDiscrete)
        assert not isinstance(self.env.action_space, spaces.MultiDiscrete)

        action_space = self._convert_gym_space(self.env.action_space)
        observation_space = self._convert_gym_space(self.env.observation_space)
        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon)

        if isinstance(action_space, Discrete):
            self._convert_action = lambda a: a[0]
        else:
            self._convert_action = lambda a: a

        super().__init__(mdp_info)

    def reset(self, state=None):
        if state is None:
            return np.atleast_1d(self.env.reset())
        else:
            self.env.reset()
            self.env.state = state

            return np.atleast_1d(state)

    def step(self, action):
        action = self._convert_action(action)
        obs, reward, absorbing, info = self.env.step(action)

        return np.atleast_1d(obs), reward, absorbing, info

    def render(self, mode='human'):
        if self._first:
            self.env.render(mode=mode)
            self._first = False

    def stop(self):
        try:
            self.env.close()
        except:
            pass

    @staticmethod
    def _convert_gym_space(space):
        if isinstance(space, spaces.Discrete):
            return Discrete(space.n)
        elif isinstance(space, spaces.Box):
            return Box(low=space.low, high=space.high, shape=space.shape)
        else:
            raise ValueError

class BallRollingEnv(gym.Env):
    def __init__(self,
                 observation_ids=[0,1,2,3],
                 renders=False,
                 seed=None,
                 isDiscrete=False,
                 actionRepeat=1,
                 maxSteps=1000,
                 mode=darias.R_ARM,
                 is_deterministic_env = False):
        self.observation_ids = observation_ids
        self._renders = renders
        self._seed = seed
        self._p = p
        self._isDiscrete = isDiscrete
        self._timeStep = 1. / 240.
        self._actionRepeat = actionRepeat
        self._maxSteps = maxSteps
        self.terminated = 0
        self._envStepCounter = 0
        self._mode = mode
        self._isDeterministic = is_deterministic_env

        if self._renders:
            cid = p.connect(p.SHARED_MEMORY)
            if cid < 0:
                cid = p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally

        self.seed(self._seed)

        if self._mode == darias.R_ARM:
            self._action_dim = 7
        elif self._mode == darias.R_ARM_HAND:
            self._action_dim = 22
        elif self._mode == darias.L_ARM:
            self._action_dim = 7
        elif self._mode == darias.L_ARM_HAND:
            self._action_dim = 22
        elif self._mode == darias.TWO_ARM:
            self._action_dim = 14
        elif self._mode == darias.TWO_ARM_HAND:
            self._action_dim = 44
        else:
            self._action_dim = 7

        if self._isDiscrete:
            self.action_space = spaces.MultiDiscrete(
                [(0, 2)] * self._action_dim)
        else:
            self.action_bound = 100
            action_high = np.array([self.action_bound] * self._action_dim)
            self.action_space = spaces.Box(-action_high, action_high)

        p.setPhysicsEngineParameter(numSolverIterations=150)
        p.setTimeStep(self._timeStep)
        p.setGravity(0, 0, -9.81)

        self.planeId = p.loadURDF("plane.urdf")
        self.darias = darias.Darias(timeStep=self._timeStep, mode=self._mode)

        _pos = [0.4, 0, 1]
        _orient = p.getQuaternionFromEuler([(100 / 180) * np.pi, (15 / 180) * np.pi, (90 / 180) * np.pi])
        self.initial_joints = p.calculateInverseKinematics(
            self.darias.id, self.darias.R_palm, _pos, _orient, maxNumIterations=1000)
        self.darias.resetJoints(self.initial_joints)

        table_collision = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[0.5, 1.5, 0.3])
        table_visual = p.createVisualShape(
            p.GEOM_BOX, halfExtents=[
                0.5, 1.5, 0.3], rgbaColor=[
                0.2, 1, 0.5, 1])
        self.table = p.createMultiBody(
            baseMass=10.0,
            baseCollisionShapeIndex=table_collision,
            baseVisualShapeIndex=table_visual,
            basePosition=[0.5, 0, 0.3])
        ball_collision = p.createCollisionShape(p.GEOM_SPHERE, radius=0.2)
        ball_visual = p.createVisualShape(
            p.GEOM_SPHERE, radius=0.2, rgbaColor=[
                0.2, 0.5, 1, 1])
        self.ball = p.createMultiBody(
            baseMass=0.2,
            baseCollisionShapeIndex=ball_collision,
            baseVisualShapeIndex=ball_visual,
            basePosition=[0.5, 1, 0.8])

        self._envStepCounter = 0

        p.changeDynamics(self.ball, -1, restitution=0.8, rollingFriction=0.01)
        p.changeDynamics(self.planeId, -1, restitution=1)
        p.changeDynamics(self.darias.id, -1, restitution=1)
        p.changeDynamics(self.table, -1, restitution=1)
        for i in range(p.getNumJoints(self.darias.id)):
            p.changeDynamics(self.darias.id, i, restitution=1)

        self.context_low = np.array([0.4, 2, -10/180*np.pi])
        self.context_high = np.array([0.6, 4, 10/180*np.pi])

    def reset(self):
        self.terminated = 0
        self._envStepCounter = 0
        self.touched = False

        if self._isDeterministic:
            x_start = 0.5
            speed = 3
            theta = 0
        else:
            x_start = np.random.rand() * 0.2 + 0.4
            speed = np.random.rand() * 2 + 2
            theta = (np.random.rand() * 20 - 10)/180*np.pi
        self.darias.resetJoints(self.initial_joints)
        p.resetBasePositionAndOrientation(self.table, [0.5, 0, 0.3], [0, 0, 0, 1])
        p.resetBasePositionAndOrientation(self.ball, [x_start, 1, 0.8], [0, 0, 0, 1])
        # p.resetBaseVelocity(self.ball, [0, 0, 0], [speed * np.cos(theta), -speed * np.sin(theta), 0])
        p.resetBaseVelocity(self.ball, [speed * np.sin(theta), -speed * np.cos(theta), 0], [0, 0, 0])

        self.context = np.array([x_start, speed, theta])

        # observation_dim = len(self.getExtendedObservation())
        self.getExtendedObservation()
        observation_dim = len(self._observation_select)
        observation_high = np.array([largeValObservation] * observation_dim)
        self.observation_space = spaces.Box(-observation_high,
                                            observation_high)

        self._observation = []
        self.ball_vel_nrom = np.array([speed * np.sin(theta), -speed * np.cos(theta), 0])/speed
        p.stepSimulation()
        self._observation = self.getExtendedObservation()

        return self._observation_select

    def step(self, action):
        self.pre_ball_pos = np.array(self._observation[2])

        for i in range(self._actionRepeat):
            self.darias.applyAction(action)
            p.stepSimulation()
            if self._termination():
                break
            self._envStepCounter += 1
        if self._renders:
            time.sleep(self._timeStep)
        self._observation = self.getExtendedObservation()

        ball_cur_vel_norm = np.array(self._observation[4])/ self.getMagnitude(np.array(self._observation[4]))
        if self.getMagnitude(self.ball_vel_nrom - ball_cur_vel_norm )>1e-3 and ball_cur_vel_norm[2]> -1e-6:
            self.touched = True

        done = self._termination()
        reward = self._reward()

        return self._observation_select, reward, done, {}

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def get_context(self):
        return self.context

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def __del__(self):
        p.disconnect()

    def getExtendedObservation(self):
        observation = []
        jointsPos, jointsVel = self.darias.getJointsObservation()
        observation.extend([jointsPos, jointsVel])

        pBall, oBall = self._p.getBasePositionAndOrientation(self.ball)
        observation.extend([pBall, oBall])
        lVelBall, aVelBall = self._p.getBaseVelocity(self.ball)
        observation.extend([lVelBall, aVelBall])

        handState = self._p.getLinkState(self.darias.id, self.darias.R_palm)
        observation.extend([handState[0]])

        self._observation_select = [item for sublist in [observation[i] for i in self.observation_ids] for item in sublist]
    
        return observation

    def _termination(self):
        if (self.terminated or self._envStepCounter >= self._maxSteps):
            return True
        if (self.getMagnitude(self._observation[4])< 0.0001):
            return True
        if self._observation[2][2] < 0.7:
            return True
        return False

    def _reward(self):
        # reward = -3
        # if self._observation[4][1]>0.0001:
        #     reward = reward + 1
        # elif abs(self._observation[4][0])> 0.0001 or abs(self._observation[4][2]) > 0.0001:
        #     reward = reward + 1
        # if self._observation[2][2] < 0.7:
        #     reward = -1000

        R = [-1, -1, -1]
        reward_ball = np.dot((np.array(self._observation[4])/ self.context[1]) **2 , R)
        Q = [-2, -1.8, -1.5, -1.3, -1, -1, -1]
        reward_darias = np.dot(np.array(self._observation[1]) ** 2, Q)

        hand_ball_distance = self.getMagnitude(np.array(self._observation[2]) - np.array(self._observation[6]))
        reward_position = -1 * (hand_ball_distance)**2

        # print(reward_ball, reward_darias, reward_position)

        # reward = 10*reward_ball+reward_darias
        # reward = 10 * reward_ball+ 50*reward_position
        if self.touched:
            reward = 5 * reward_ball
        else:
            reward = 100 * reward_position

        if self.getMagnitude(self._observation[4])< 0.0001:
            reward = 20000
        return reward/self._maxSteps

    def getMagnitude(self, vec):
        return np.sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2)

if __name__ == "__main__":
    ballRollingEnv = BallRollingEnv(renders=True, mode=darias.R_ARM, maxSteps=1000)

    y = 0
    # frame_id_palm = None
    # frame_id = None
    demos = []
    for samples in range(200):
        total_reward = 0
        ballRollingEnv.reset()
        traj =[]
        traj.append(ballRollingEnv.context)
        for t in range(ballRollingEnv._maxSteps):
            ball_vel = ballRollingEnv._observation[4][1]
            ball_pos = ballRollingEnv._observation[2]
            _pos = [ball_pos[0], y, 1.0]
            y += 0.45 * min(ball_vel, 0) * ballRollingEnv._timeStep
            _orient = p.getQuaternionFromEuler([(100/180)*np.pi, (15/180)*np.pi, (90/180)*np.pi])
            joints = p.calculateInverseKinematics(ballRollingEnv.darias.id, ballRollingEnv.darias.R_palm, _pos, _orient)
            observation, reward, done, _ = ballRollingEnv.step(joints[0:7])
            total_reward +=reward

            traj.append(np.array(observation[0]))
            # frame_id_palm = draw_frame.drawFrame(ballRollingEnv.darias.id, ballRollingEnv.darias.R_palm, frame_id_palm)
            if done:
                print("episode {} finish, total time: {}, total reward: {}".format(samples, t, total_reward))
                # frame_id_palm = None
                # frame_id = None
                y = 0
                total_reward=0
                break
            # p.addUserDebugParameter("gains", 0,10000,1000)
        save_list2csv('../data/ball_rolling_demo_context_'+str(samples)+'.csv', traj)

