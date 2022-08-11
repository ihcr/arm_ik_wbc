""" References:
[1] Carpentier, J., Valenza, F., and Mansard, N. Pinocchio (2.6.1). [Software]. 2021. [Accessed 23 June 2021].
[2] Harris, C. R., Millman, K. J., van der Walt, S. J. et al. Array Programming with NumPy. Nature. 2020, 585, pp.357-362
"""

import pinocchio as pin #[1]
import numpy as np #[2]
import math

class CoMCartesianTask:
    """ Class Input:
        task_weight (float): The task weight with a magnitude based on the priority of this task being met.
        task_gain (float): The gain applied to the difference in target and current CoM position.
    """
    def __init__(self, task_weight, task_gain):
        self.task_weight = task_weight
        self.task_gain = task_gain
        self.robot_model = None
        self.robot_data = None
        self.joint_config = None
        self.prev_CoM_pos = None
        self.n_of_targets = 3 # number of rows in the output matrices

    def CoMCartesianInitialise(self, robot_model, robot_data, joint_config):
        # update the robot model, data, and joint configuration
        self.robot_model = robot_model
        self.robot_data = robot_data
        self.joint_config = joint_config

        # find the Jacobian and position of the CoM
        pin.jacobianCenterOfMass(self.robot_model, self.robot_data, self.joint_config)
        self.prev_CoM_pos = self.robot_data.com[0].reshape(3,1)

    def CoMCartesianA(self):
        # Find the CoM jacobian
        A = np.copy(pin.jacobianCenterOfMass(self.robot_model, self.robot_data, self.joint_config))
        # Apply task weight
        A = A * self.task_weight

        return A

    def CoMCartesianb(self, target_pos, dt):
        # Find the position of the CoM
        fk_CoM_pos = self.robot_data.com[0].reshape(3,1)

        # Find the desired velocity of the CoM
        ref_vel = (target_pos - self.prev_CoM_pos)/dt
        pos_vel = ref_vel + (np.dot(self.task_gain, ((target_pos - fk_CoM_pos)/dt)))

        # Store previous
        self.prev_CoM_pos = target_pos

        # Apply task weight
        pos_vel = pos_vel * self.task_weight

        return pos_vel

    def CoMCartesianFindAb(self, robot_model, robot_data, joint_config, target_pos, dt):
        # Update class variables
        self.robot_model = robot_model
        self.robot_data = robot_data
        self.joint_config = joint_config

        # Find task A and b
        A = self.CoMCartesianA()
        b = self.CoMCartesianb(target_pos, dt)

        return A, b

    def updateTaskParameters(self, task_weight=False, task_gain=False):
        # update class parameters to allow for dynamic task changes
        if type(task_weight) != type(False):
            self.task_weight = task_weight
        if type(task_gain) != type(False):
            self.task_gain = task_gain

