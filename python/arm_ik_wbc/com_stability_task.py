""" References:
[1] Carpentier, J., Valenza, F., and Mansard, N. Pinocchio (2.6.1). [Software]. 2021. [Accessed 23 June 2021].
[2] Harris, C. R., Millman, K. J., van der Walt, S. J. et al. Array Programming with NumPy. Nature. 2020, 585, pp.357-362
"""

import pinocchio as pin #[1]
import numpy as np #[2]
import math

class CoMStabilityTask:
    """ Class Input:
        task_weight (float): The task weight with a magnitude based on the priority of this task being met.
        task_gain (float): The gain applied to the difference in target and current CoM position.
    """
    def __init__(self, task_weight, task_gain):
        self.robot_model = None
        self.robot_data = None
        self.joint_config = None
        self.anchor_points_ABC = []
        self.anchor_points_pos = None
        self.n_of_targets = 2 # number of rows in the output matrices
        self.task_weight = task_weight
        self.task_gain = task_gain
        self.swing_feet = 0

    def CoMStabilityUpdate(self, anchor_points_pos):
        self.swing_feet = 0
        for i in range(len(anchor_points_pos)):
            if type(anchor_points_pos[i])==bool:
                self.swing_feet = self.swing_feet + 1
            else:
                pass
        if self.swing_feet <= 2:
            # based on the position of the anchor points find the position at the centre of the support polygon
            # this will serve as the tharget position of the CoM
            self.anchor_points_pos = []
            self.anchor_points_ABC = []
            for i in range(len(anchor_points_pos)):
                if type(anchor_points_pos[i]) != type(False):
                    self.anchor_points_pos.append(anchor_points_pos[i])
            x = 0
            y = 0
            for i in range(len(self.anchor_points_pos)):
                x = x + self.anchor_points_pos[i][0]
                y = y + self.anchor_points_pos[i][1]
            x_target = x/len(self.anchor_points_pos)
            y_target = y/len(self.anchor_points_pos)
            self.CoM_target = np.array([x_target, y_target]).reshape(2,1)
        else:
            pass

    def CoMStabilityInitialise(self, robot_model, robot_data, joint_config):
        # Complete the initialisation steps of this class, finding the Jacobian and position of the CoM
        self.robot_model = robot_model
        self.robot_data = robot_data
        self.joint_config = joint_config
        pin.jacobianCenterOfMass(self.robot_model, self.robot_data, self.joint_config)
        self.prev_CoM_pos = self.robot_data.com[0][:2].reshape(2,1)

    def CoMStabilityA(self):
        # find the Jacobian of the CoM
        A = np.copy(pin.jacobianCenterOfMass(self.robot_model, self.robot_data, self.joint_config))[:2]
        A = A * self.task_weight

        return A

    def CoMStabilityb(self, dt):
        # Calculate the the task's target velocity
        fk_CoM_pos = np.copy(self.robot_data.com[0][:2]).reshape(2,1)

        ref_vel = (self.CoM_target - self.prev_CoM_pos)/dt
        pos_vel = ref_vel + (np.dot(self.task_gain, ((self.CoM_target - fk_CoM_pos)/dt)))

        self.prev_CoM_pos = fk_CoM_pos

        pos_vel = pos_vel * self.task_weight

        return pos_vel

    def CoMStabilityFindAb(self, robot_model, robot_data, joint_config, dt):
        # update the robot model, data, and joint config
        self.robot_model = robot_model
        self.robot_data = robot_data
        self.joint_config = joint_config

        A = self.CoMStabilityA()
        b = self.CoMStabilityb(dt)

        return A, b

    def updateTaskParameters(self, task_weight=False, task_gain=False):
        # update the task parameters so that this task can be updated dynamically
        if type(task_weight) != type(False):
            self.task_weight = task_weight
        if type(task_gain) != type(False):
            self.task_gain = task_gain
        
        
