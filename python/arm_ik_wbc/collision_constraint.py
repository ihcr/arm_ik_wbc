""" References:
[1] Carpentier, J., Valenza, F., and Mansard, N. Pinocchio (2.6.1). [Software]. 2021. [Accessed 23 June 2021].
[2] Harris, C. R., Millman, K. J., van der Walt, S. J. et al. Array Programming with NumPy. Nature. 2020, 585, pp.357-362
"""

import pinocchio as pin #[1]
import numpy as np #[2]
import math

class CollisionConstraint:
    """ Class Input:
        col_pair_frame_ids (list int): A list of the index values of the frames in the collision pair (in relation to the pinocchio model)
    """
    def __init__(self, col_pair_frame_ids):
        self.col_pair_frame_ids = col_pair_frame_ids
        print(self.col_pair_frame_ids)
        self.geom_model = None
        self.geom_data = None
        self.n_of_constraints = 3 # number of rows in the output matrices

    def CollisionInitialise(self, geom_model, geom_data):
        # define the geometry model and data of the pinocchio model robot model
        self.geom_model = geom_model
        self.geom_data = geom_data

        # Find collision id
        self.col_geom_name_1 = 0
        self.col_geom_id_1 = 0
        self.col_frame_id_1 = 0
        self.col_geom_name_2 = 0
        self.col_geom_id_2 = 0
        self.col_frame_id_2 = 0
        self.col_id = 0
        
        for i in range(len(self.geom_model.geometryObjects)):
            if self.geom_model.geometryObjects[i].parentFrame == self.col_pair_frame_ids[0]:
                self.col_geom_name_1 = self.geom_model.geometryObjects[i].name
            if self.geom_model.geometryObjects[i].parentJoint == self.col_pair_frame_ids[1]:
                self.col_geom_name_2 = self.geom_model.geometryObjects[i-1].name

        self.col_geom_id_1 = self.geom_model.getGeometryId(self.col_geom_name_1)
        self.col_geom_id_2 = self.geom_model.getGeometryId(self.col_geom_name_2)
        self.col_frame_id_1 = self.geom_model.geometryObjects[self.col_geom_id_1].parentFrame
        self.col_frame_id_2 = self.geom_model.geometryObjects[self.col_geom_id_2].parentFrame
        self.col_id = self.geom_model.findCollisionPair(pin.CollisionPair(self.col_geom_id_1, self.col_geom_id_2))

        # set only the collision pair of interest active
        self.geom_data.deactivateAllCollisionPairs()
        self.geom_data.activateCollisionPair(self.col_id)

    def CollisionC(self, robot_model, robot_data):
        # Update geometry model
        pin.updateGeometryPlacements(robot_model, robot_data, self.geom_model, self.geom_data)
        # Calculate distance data
        pin.computeDistance(self.geom_model, self.geom_data, self.col_id)

        # Set velocity damper parameters
        damping = 0.1
        ds = 0.05
        di = 0.08

        """ Calculate C, Clb and Cub """
        d = self.geom_data.distanceResults[self.col_id].min_distance
        if d < di:
            # apply velocity damper method
            n = self.geom_data.distanceResults[self.col_id].normal
            J1 = pin.getFrameJacobian(robot_model, robot_data, self.col_frame_id_1, pin.ReferenceFrame.LOCAL)[:3]
            J2 = pin.getFrameJacobian(robot_model, robot_data, self.col_frame_id_2, pin.ReferenceFrame.LOCAL)[:3]
            C = np.dot(n.reshape(1,3), (J1-J2))
            Clb = np.array([-damping * ((d - ds)/(di - ds))])
            Cub = np.array([math.inf])
            return C, Clb, Cub
        else:
            return [],[],[]

    def CollisionFindConstraint(self, robot_model, robot_data):
        # find C, Clb, and Cub based on the current pinocchio robot model and data
        C, Clb, Cub = self.CollisionC(robot_model, robot_data)

        return C, Clb, Cub

    def CollisionChangePair(new_col_pair_frame_ids):
        # update the collision pair of interest
        self.col_pair_frame_ids = new_col_pair_frame_ids
        self.CollisionInitialise(self.geom_model, self.geom_data)


        
            
