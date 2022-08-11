""" References:
[1] Carpentier, J., Valenza, F., and Mansard, N. Pinocchio (2.6.1). [Software]. 2021. [Accessed 23 June 2021].
[2] Harris, C. R., Millman, K. J., van der Walt, S. J. et al. Array Programming with NumPy. Nature. 2020, 585, pp.357-362
[3]    Topcoder. Geometry Concepts Part 2: Line Intersection and its Application. [Online]. 2018.
       [Accessed 26 August 2021]. Available from: https://www.topcoder.com/thrive/articles/Geometry%20Concepts%20part%202:%20%20Line%20Intersection%20and%20its%20Applications
       

"""


import pinocchio as pin #[1]
import numpy as np #[2]
import math

class CoMStabilityConstraint:
    def __init__(self):
        self.robot_model = None
        self.robot_data = None
        self.joint_config = None
        self.anchor_points_ABC = []
        self.anchor_points_pos = None
        self.n_of_constraints = 2 # number of rows in the output matrices
        self.swing_feet = 0

    def CoMStabilityUpdateABC(self, anchor_points_pos):
        self.swing_feet = 0
        for i in range(len(anchor_points_pos)):
            if type(anchor_points_pos[i])==bool:
                self.swing_feet = self.swing_feet + 1
            else:
                pass
        if self.swing_feet <= 2:
            # initialise parameters required to update the support polygon data
            self.anchor_points_pos = []
            self.anchor_points_ABC = []
            for i in range(len(anchor_points_pos)):
                if type(anchor_points_pos[i]) != type(False):
                    self.anchor_points_pos.append(anchor_points_pos[i])

            # For the anchor points find the equation of the virtual lines between each anchor point to for the support polygon
            # this is in the form of Ax + By = C and the method is taken from [3]
            #print("CALLLLLLLED",len(self.anchor_points_pos))
            for i in range(len(self.anchor_points_pos)):
                if i < (len(self.anchor_points_pos)-1):
                    x1 = self.anchor_points_pos[i][0]
                    y1 = self.anchor_points_pos[i][1]
                    x2 = self.anchor_points_pos[i+1][0]
                    y2 = self.anchor_points_pos[i+1][1]
                    A = y2-y1
                    B = x1-x2
                    C = A*x1 + B*y1
                    self.anchor_points_ABC.append([A,B,C])
                else:
                    x1 = self.anchor_points_pos[i][0]
                    y1 = self.anchor_points_pos[i][1]
                    x2 = self.anchor_points_pos[0][0]
                    y2 = self.anchor_points_pos[0][1]
                    A = y2-y1
                    B = x1-x2
                    C = A*x1 + B*y1
                    self.anchor_points_ABC.append([A,B,C])
        else:
            pass

    def CoMStabilityC(self, dt):
        # find the position and Jacobian of the centre of mass
        C= pin.jacobianCenterOfMass(self.robot_model, self.robot_data, self.joint_config)[:2]
        CoM_pos = self.robot_data.com[0][:2]
        if self.swing_feet <=2:
            # initialise parameters
            CoM_x_A = 1
            CoM_x_B = 0
            CoM_x_C = CoM_pos[0]
            CoM_y_A = 0
            CoM_y_B = 1
            CoM_y_C = CoM_pos[1]
            x_bounds = []
            y_bounds = []

            # Find where the CoM could intersect with the support polygon for x and y
            for i in range(len(self.anchor_points_ABC)):
                det_x = (CoM_x_A * self.anchor_points_ABC[i][1]) - (self.anchor_points_ABC[i][0] * CoM_x_B)
                det_y = (CoM_y_A * self.anchor_points_ABC[i][1]) - (self.anchor_points_ABC[i][0] * CoM_y_B)
                if det_x != 0:
                    x = ((self.anchor_points_ABC[i][1] * CoM_x_C) - (CoM_x_B * self.anchor_points_ABC[i][2]))/det_x
                    y = ((CoM_x_A * self.anchor_points_ABC[i][2]) - (self.anchor_points_ABC[i][0] * CoM_x_C))/det_x
                    x_bounds.append(x)
                    y_bounds.append(y)
                if det_y != 0:
                    x = ((self.anchor_points_ABC[i][1] * CoM_y_C) - (CoM_y_B * self.anchor_points_ABC[i][2]))/det_y
                    y = ((CoM_y_A * self.anchor_points_ABC[i][2]) - (self.anchor_points_ABC[i][0] * CoM_y_C))/det_y
                    x_bounds.append(x)
                    y_bounds.append(y)

            # Find the bounds for the CoM position to promote stability
            lb_pos = np.zeros((C.shape[0],))
            ub_pos = np.zeros((C.shape[0],))
            
            
            lb_pos[0] = min(x_bounds)
            lb_pos[1] = min(y_bounds)
            ub_pos[0] = max(x_bounds)
            ub_pos[1] = max(y_bounds)

            self.lb_lim_pos = lb_pos
            self.ub_lim_pos = ub_pos

            # Find the upper and lower bounds of the CoM stability to ensure the CoM stays within the support polygon bounds
            Clb = ((lb_pos - CoM_pos) / dt).reshape(C.shape[0])*1
            Cub = ((ub_pos - CoM_pos) / dt).reshape(C.shape[0])*1
            self.prev_Clb = Clb
            self.prev_Cub = Cub
           
        else:
            Clb = self.prev_Clb
            Cub = self.prev_Cub

        return C, Clb, Cub

    def CoMStabilityFindConstraint(self, robot_model, robot_data, joint_config, dt):
        # update class parameters
        self.robot_model = robot_model
        self.robot_data = robot_data
        self.joint_config = joint_config

        # Find the constraint parameters
        C, Clb, Cub = self.CoMStabilityC(dt)
        self.fetch_Clb = Clb
        self.fetch_Cub = Cub

        return C, Clb, Cub
    
