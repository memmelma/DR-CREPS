import pybullet as p
import numpy as np

R_ARM = 1
R_ARM_HAND = 2
L_ARM = 3
L_ARM_HAND = 4
TWO_ARM = 5
TWO_ARM_HAND = 6

class Darias:
    def __init__(self, timeStep, mode=R_ARM):
        self._mode = mode
        self._timeStep = timeStep
        self.reset()
        self.R_EE_link = 8
        self.R_palm = 9
        self.L_EE_link = 32
        self.L_palm = 33
        self.jointsPosLowerLimits = [-2.967059, -2.094395, -2.967059, -2.094395, -2.967059, -2.094395, -2.967059,
                                     -0.261, 0.087, 0.087, -0.261, 0.087, 0.087, -0.261, 0.087, 0.087,
                                     -0.261, 0.087, 0.087,-0.261, 0.087, 0.087,
                                     -2.967059, -2.094395, -2.967059, -2.094395, -2.967059, -2.094395, -2.094395,
                                     -0.261, 0.087, 0.087, -0.261, 0.087, 0.087, -0.261, 0.087, 0.087,
                                     -0.261, 0.087, 0.087, -0.261, 0.087, 0.087]

        self.jointsPosUpperLimits = [2.967059, 2.094395, 2.967059, 2.094395, 2.967059, 2.094395, 2.967059,
                                     0.2618, 1.4835, 1.1345 ,0.2618, 1.4835, 1.1345, 0.2618, 1.4835, 1.1345,
                                     0.2618, 1.4835, 1.1345, 0.2618, 1.4835, 1.1345,
                                     2.967059, 2.094395, 2.967059, 2.094395, 2.967059, 2.094395, 2.094395,
                                     0.2618, 1.4835, 1.1345, 0.2618, 1.4835, 1.1345, 0.2618, 1.4835, 1.1345,
                                     0.2618, 1.4835, 1.1345, 0.2618, 1.4835, 1.1345]

        self.jointsVelLimits = [1.91986217719, 1.91986217719, 2.26892802759, 2.26892802759, 2.26892802759,
                                3.14159265359, 3.14159265359,
                                100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0,
                                100.0, 100.0, 100.0, 100.0, 100.0, 100.0,
                                1.91986217719, 1.91986217719, 2.26892802759, 2.26892802759, 2.26892802759,
                                3.14159265359, 3.14159265359,
                                100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0,
                                100.0, 100.0, 100.0, 100.0, 100.0, 100.0]

        self.jointsForceLimits = [10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0,
                                  100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0,
                                  100.0, 100.0, 100.0, 100.0, 100.0, 100.0,
                                  10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0,
                                  100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0,
                                  100.0, 100.0, 100.0, 100.0, 100.0, 100.0]

    def reset(self):
        import os
        print(os.getcwd())
        self.id = p.loadURDF("/custom_env/urdf/darias.urdf")

        if self._mode == R_ARM:
            self.jointIndexes = np.arange(2,9)
        elif self._mode == R_ARM_HAND:
            _joints = np.arange(2, 25)
            self.jointIndexes = np.delete(_joints, np.where(_joints==9))
        elif self._mode == L_ARM:
            self.jointIndexes = np.arange(26,33)
        elif self._mode == L_ARM_HAND:
            _joints = np.arange(26,49)
            self.jointIndexes = np.delete(_joints, np.where(_joints == 33))
        elif self._mode == TWO_ARM:
            self.jointIndexes = np.concatenate((np.arange(2,9),np.arange(26,33)))
        elif self._mode == TWO_ARM_HAND:
            self.jointIndexes = np.concatenate((np.arange(2,9),np.arange(10,25), np.arange(26,33), np.arange(34,49)))

        self.numOfJoints = len(self.jointIndexes)

        self.jointsPos = []
        self.jointsVel = []
        for i in range(self.numOfJoints):
            self.jointsVel.extend([0])
            self.jointsPos.extend([0])

    def resetJoints(self, q):
        nCommands = len(q)
        _indexes = np.concatenate((np.arange(2, 9), np.arange(10, 25), np.arange(26, 33), np.arange(34, 49)))

        assert len(_indexes) == nCommands, "Number of Joints must consistent with number of commands"
        for i in range(len(_indexes)):
            p.resetJointState(self.id, _indexes[i], q[i], targetVelocity=0.0)

        for i in range(self.numOfJoints):
            p.setJointMotorControl2(self.id, jointIndex=self.jointIndexes[i], controlMode=p.POSITION_CONTROL,
                                    targetPosition=q[i])

        self.getJointsObservation()

    def applyAction(self, motorCommands):
        nCommands = len(motorCommands)
        assert self.numOfJoints == nCommands, "Number of Joints must consistent with number of commands"
        # TODO Apply different type of control
        if len(motorCommands) == 2:
            q = list(motorCommands[0])
            dq = list(motorCommands[1])
            q = np.clip(q, self.jointsPosLowerLimits, self.jointsPosUpperLimits)
            for i in range(self.numOfJoints):
                q[i] = np.clip(q[i], self.jointsPosLowerLimits[i], self.jointsPosUpperLimits[i])
                dq[i] = np.clip(q[i], -self.jointsVelLimits[i], self.jointsVelLimits[i])
                p.setJointMotorControl2(self.id, jointIndex=self.jointIndexes[i], controlMode=p.POSITION_CONTROL,
                                        targetPosition=q[i], targetVelocity=dq[i])
        else:
            q = list(motorCommands)
            for i in range(self.numOfJoints):
                q[i] = np.clip(q[i], self.jointsPosLowerLimits[i], self.jointsPosUpperLimits[i])
                p.setJointMotorControl2(self.id, jointIndex=self.jointIndexes[i], controlMode=p.POSITION_CONTROL, targetPosition = q[i])

    def getJointsObservation(self):
        for i in range(self.numOfJoints):
            self.jointsPos[i], self.jointsVel[i], _, _ = p.getJointState(self.id, self.jointIndexes[i])
        return tuple(self.jointsPos), tuple(self.jointsVel)



def test():
    cid = p.connect(p.SHARED_MEMORY)
    if cid < 0:
        cid = p.connect(p.GUI)
    darias = Darias(mode=L_ARM_HAND, timeStep=1./240.)

    num_joints = len(darias.jointIndexes)

    import time
    for i in range(10000):
        p.stepSimulation()
        darias.applyAction(2*np.ones(num_joints)*np.sin(i/240./np.pi/2*10))
        time.sleep(darias._timeStep)


if __name__ == "__main__":
    test()