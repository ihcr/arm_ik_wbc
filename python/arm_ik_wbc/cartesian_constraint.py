""" References:
[1] Carpentier, J., Valenza, F., and Mansard, N. Pinocchio (2.6.1). [Software]. 2021. [Accessed 23 June 2021].
[2] Harris, C. R., Millman, K. J., van der Walt, S. J. et al. Array Programming with NumPy. Nature. 2020, 585, pp.357-362
"""

import pinocchio as pin #[1]
import numpy as np #[2]
import math

class CartesianConstraint:
    """ Class Input:
        frame_index (unsigned int): The index of the frame in the robot model that the constraint is applied to.
        reference_frame (pinocchio definition): The is is the pinocchio reference frame and is one of the following: pin.WORLD, pin.LOCAL, pin.LOCAL_WORLD_ALIGNED.
        x (bool): True if this plane is to be constrained to zero velocity, otherwise False.
        y (bool): True if this plane is to be constrained to zero velocity, otherwise False.
        z (bool): True if this plane is to be constrained to zero velocity, otherwise False.
        roll (bool): True if this plane is to be constrained to zero velocity, otherwise False.
        pitch (bool): True if this plane is to be constrained to zero velocity, otherwise False.
        yaw (bool): True if this plane is to be constrained to zero velocity, otherwise False.
        auto (bool): If True this constraint automatically constraints a plane of motion if two subsequent target velocities in a plane are equal.
    """
    def __init__(self, frame_index, reference_frame, x, y, z, roll, pitch, yaw, auto):
        self.frame_index = frame_index
        self.constraints = [x, y, z, roll, pitch, yaw]
        self.reference_frame = reference_frame
        self.robot_model = None
        self.robot_data = None
        self.n_of_constraints = 6 # number of rows in the output matrices
        self.auto = auto

    def CartesianC(self):
        # fetch the frame jacobian
        C = pin.getFrameJacobian(self.robot_model, self.robot_data, self.frame_index, self.reference_frame)
        
        # remove the constraint if that plane is not to be constrained as defined in self.constraints
        for i in range(len(self.constraints)):
            if self.constraints[i] == False:
                C[i] = 0
                
        # set the lower and upper bounds to zero to enforce a constraint of zero velocity
        Clb = np.zeros(C.shape[0]).reshape((C.shape[0],))
        Cub = np.zeros(C.shape[0]).reshape((C.shape[0],))
        return C, Clb, Cub

    def CartesianFindConstraint(self, robot_model, robot_data):
        # Update class parameters
        self.robot_model = robot_model
        self.robot_data = robot_data
        #print(self.constraints)
        # find C, Clb, and Cub
        C, Clb, Cub = self.CartesianC()
        return C, Clb, Cub

    def updateConstraintParameters(self, frame_index, reference_frame, constraints):
        # update the class variables, allows for dynamic constraint changes
        if type(frame_index) != type(False):
            self.frame_index = frame_index
        if type(reference_frame) != type(False):
            self.reference_frame = reference_frame
        self.constraints = constraints

