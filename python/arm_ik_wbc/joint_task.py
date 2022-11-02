""" References:
[1] Carpentier, J., Valenza, F., and Mansard, N. Pinocchio (2.6.1). [Software]. 2021. [Accessed 23 June 2021].
[2] Harris, C. R., Millman, K. J., van der Walt, S. J. et al. Array Programming with NumPy. Nature. 2020, 585, pp.357-362
"""

import pinocchio as pin #[1]
import numpy as np #[2]
import math

class JointTask:
    """ Class Input:
        joint_index (unsigned int): The index of the joint in the robot model that this joint task is applied to.
        system_vel_dimensions(unsigned int): The number of DoF of the robot.
        joint_task_weight (float): The task weight with a magnitude based on the priority of this task being met.
        task_type (string): The name of the type of joint task to be applied to this joint- "PREV"-removes high oscillations, True- good task conditioning, "MANI"- reduces chances of singularites occuring
        n_of_base_DoF (unsigned int): The number of the degrees of freedom of the trunk of the robot (typically this is 6)
    """
    def __init__(self, joint_index, system_vel_dimensions, joint_task_weight, task_type, n_of_base_DoF, init_config):
        self.joint_index = joint_index
        self.system_vel_dimensions = system_vel_dimensions
        self.joint_task_weight = joint_task_weight
        self.task_type = task_type
        self.joint_robot_model = None
        self.joint_robot_data = None
        self.n_of_targets = 1
        self.n_of_base_DoF = n_of_base_DoF
        self.init_config = init_config

    def quickUpdateState(self, joint_config):
        # update the robot model and data
        pin.forwardKinematics(self.robot_model, self.robot_data, joint_config)
        self.J = pin.computeJointJacobians(self.robot_model, self.robot_data, joint_config)
        pin.framesForwardKinematics(self.robot_model, self.robot_data, joint_config)
        pin.updateFramePlacements(self.robot_model, self.robot_data)

    def jointA(self):
        # Select joint
        A = np.zeros((self.system_vel_dimensions,))
        # Apply task weight
        A[self.joint_index] = 1/self.system_vel_dimensions
        A = A * self.joint_task_weight

        return A

    def jointb(self, joint_config):
        # Tikhonov Regularization
        if self.task_type == True:
            b = np.zeros((1,))

        # Elimenate high frequency oscillations
        if self.task_type == "PREV":
            #jc = np.delete(joint_config, 6)
            #b = np.array([jc[self.joint_index]])
            b = np.array([joint_config[self.joint_index]])
            
        if self.task_type == "POSE":
            b = np.array([self.init_config[self.joint_index]])

        # Manipulability gradient
        if self.task_type == "MANI":
            delta_q = 0.002
            joint_config[self.joint_index] = joint_config[self.joint_index] + delta_q
            self.quickUpdateState(joint_config)
            J = pin.getJointJacobian(self.robot_model, self.robot_data, (self.joint_index-self.n_of_base_DoF), pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            f1 = math.sqrt(np.linalg.det(np.dot(J, J.T)))
            joint_config[self.joint_index] = joint_config[self.joint_index] - (delta_q*2)
            self.quickUpdateState(joint_config)
            J = pin.getJointJacobian(self.robot_model, self.robot_data, (self.joint_index-self.n_of_base_DoF), pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            f2 = math.sqrt(np.linalg.det(np.dot(J, J.T)))
            b = np.array([(0.5*(f1-f2)/delta_q)])

        # Apply task weights
        b = b * (1/self.system_vel_dimensions)
        b = b * self.joint_task_weight

        return b

    def jointFindAb(self, robot_model, robot_data, joint_config):
        # Update class variables
        self.robot_model = robot_model
        self.robot_data = robot_data

        # Find task A and b
        A = self.jointA()
        b = self.jointb(joint_config)

        return A, b

    def updateTaskParameters(joint_index=False, system_vel_dimensions=False, joint_task_weight=False, task_type=False):
        # update the class parameters so that this task can be dynamically updated.
        if type(joint_index) != type(False):
            self.joint_index = joint_index
        if type(system_vel_dimensions) != type(False):
            self.system_vel_dimensions = system_vel_dimensions
        if type(joint_task_weight) != type(False):
            self.joint_task_weight = joint_task_weight
        if type(task_type) != type(False):
            self.task_type = task_type
