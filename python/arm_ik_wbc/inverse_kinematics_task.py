""" References:
[1] Carpentier, J., Valenza, F., and Mansard, N. Pinocchio (2.6.1). [Software]. 2021. [Accessed 23 June 2021].
[2] Virtanen, P., Gommer, R., Oliphant, T. E. et al. SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python. Nature Methods. 2020, 17, pp.261-272
[3] Harris, C. R., Millman, K. J., van der Walt, S. J. et al. Array Programming with NumPy. Nature. 2020, 585, pp.357-362
"""

import pinocchio as pin #[1]
from scipy.spatial.transform import Rotation as R #[2]
import numpy as np #[3]
import math
""" Class Input:
    frame_index (unsigned int): The index of the frame in the robot model that this Cartesian task is applied to.
    task_weight (float): The task weight with a magnitude based on the priority of this task being met.
    A_weight (DoFxDoF float matrix): The matrix that defines how strictly the task is tracked
    b_gain (float): The gain applied to the difference in target and current CoM position.
    frame (pinocchio definition): The is is the pinocchio reference frame and is one of the following: pin.WORLD, pin.LOCAL, pin.LOCAL_WORLD_ALIGNED.
"""
class InverseKinematicsTask:
    def __init__(self, frame_index, task_weight, A_weight, b_gain, frame):
        self.frame_index = frame_index
        self.task_weight = task_weight
        self.A_weight = A_weight
        self.b_gain = b_gain
        # Separate the position and orientation gains
        self.b_pos_gain = self.b_gain[0:3, 0:3]
        self.b_ori_gain = self.b_gain[3:, 3:]
        self.robot_model = None
        self.robot_data = None
        self.n_of_targets = 6 # number of rows in the output matrices
        self.frame = frame

    def IKInitialiseTask(self, robot_model, robot_data):
        # complete task initialisation processes, fetching the position and orientation of the task frame
        self.robot_model = robot_model
        self.robot_data = robot_data
        self.prev_pos = np.copy(self.robot_data.oMf[self.frame_index].translation).reshape(3,1)
        prev_ori_mat = R.from_matrix(self.robot_data.oMf[self.frame_index].rotation)
        self.prev_ori = prev_ori_mat.as_euler('xyz')

    def IKA(self):
        # Find the frame jacobian and apply the task weight and jacobian weight
        A = pin.getFrameJacobian(self.robot_model, self.robot_data, self.frame_index, self.frame).T
        # Apply weights
        A = A * self.task_weight
        A = np.dot(self.A_weight, A.T)

        return A

    def IKb(self, target_pos, target_ori, dt):
        Nonetype = type(None)
        # Find the position and orientation of the frame through forward kinematics
        fk_pos = np.copy(self.robot_data.oMf[self.frame_index].translation).reshape(3,1)
        fk_ori = R.from_matrix(np.copy(self.robot_data.oMf[self.frame_index].rotation))
        # Find base desired velocity not accounting for residual
        ref_vel = (target_pos - self.prev_pos)/dt
        pos_vel = ref_vel + (np.dot(self.b_pos_gain, ((target_pos - fk_pos)/dt)))
        # Find the angular velocity
        fk_ori_euler = fk_ori.as_euler('xyz')
        ori_error = ((target_ori.reshape(3,) - fk_ori_euler)/dt).reshape(3,)
        w = np.dot(ori_error, self.b_ori_gain)
        w_vel = (target_ori.reshape(3,) - self.prev_ori.reshape(3,))/dt
        ori_vel = (w_vel + w).reshape(3,1)
        # Store previous values
        self.prev_pos = target_pos
        
        
        
        self.prev_ori = target_ori

        # Find overall target velocity
        target_vel = np.concatenate((pos_vel, ori_vel), axis=0)
        # Apply task weight
        target_vel = target_vel * self.task_weight
        
        #print(target_vel)

        return target_vel

    def IKFindAb(self, robot_model, robot_data, target_pos, target_ori, dt):
        # Update class variables
        self.robot_model = robot_model
        self.robot_data = robot_data

        # Find task A and b
        A = self.IKA()
        b = self.IKb(target_pos, target_ori, dt)

        return A, b

    def updateTaskParameters(self, frame_index=False, task_weight=False, A_weight=False, b_gain=False):
        # update class parameters to enable dynamics changes to the task to be made
        if type(frame_index) != type(False):
            self.frame_index = frame_index
        if type(task_weight) != type(False):
            self.task_weight = task_weight
        if type(A_weight) != type(False):
            self.A_weight = A_weight
        if type(b_gain) != type(False):
            self.b_gain = b_gain
            self.b_pos_gain = self.b_gain[0:3, 0:3]
            self.b_ori_gain = self.b_gain[3:, 3:]
        
