import pinocchio as pin #[1]
from scipy.spatial.transform import Rotation as R #[2]
import numpy as np #[3]
import math
import time
from arm_ik_wbc.QP_Wrapper import QP
from arm_ik_wbc.inverse_kinematics_task import InverseKinematicsTask
from arm_ik_wbc.joint_task import JointTask
from arm_ik_wbc.com_cartesian_task import CoMCartesianTask
from arm_ik_wbc.cartesian_constraint import CartesianConstraint
from arm_ik_wbc.com_stability_constraint import CoMStabilityConstraint
from arm_ik_wbc.collision_constraint import CollisionConstraint
from arm_ik_wbc.com_stability_task import CoMStabilityTask
from arm_ik_wbc.com_cartesian_task import CoMCartesianTask
from arm_ik_wbc.local_cartesian_constraint import LocalCartesianConstraint
import sys
import copy


class IkWBC:
    def __init__(self, urdf_path, mesh_dir_path, EE_frame_names, EE_joint_names, base_frame, init_joint_config, floating_base=False, foot_offset=False):
        """ Initialise Pinocchio model, data and geometry model """
        self.robot_model= pin.buildModelFromUrdf(urdf_path)
        #self.geom_model = pin.buildGeomFromUrdf(self.robot_model, urdf_path, mesh_dir_path, pin.GeometryType.COLLISION)
        #self.geom_model.addAllCollisionPairs()
        self.robot_data = self.robot_model.createData()
        #self.geom_data = pin.GeometryData(self.geom_model)
        self.joint_names = self.robot_model.names
        
        #!TODO! add foot contact stuff
        
        # model dimensions
        self.n_velocity_dimensions = self.robot_model.nv
        self.n_configuration_dimensions = self.robot_model.nq
        self.n_of_manip_joints = self.n_configuration_dimensions - init_joint_config.size
        print("len thing", self.n_of_manip_joints)
        self.n_of_EE = len(EE_frame_names)
        
        # base frame name and index
        self.base_frame_name = base_frame
        self.base_joint_id = self.robot_model.getJointId(base_frame)
        self.base_frame_index = self.robot_model.getFrameId(base_frame, pin.JOINT)
        
        # end-effector name(s) and index(s)
        self.EE_frame_names = EE_frame_names.copy()
        self.EE_joint_names = EE_joint_names.copy()
        self.EE_index_list_frame = [] 
        self.EE_index_list_joint = []
        for i in range(self.n_of_EE):
            ID = self.robot_model.getFrameId(self.EE_frame_names[i], pin.FIXED_JOINT)
            self.EE_index_list_frame.append(ID)
            ID = self.robot_model.getJointId(self.EE_joint_names[i])
            self.EE_index_list_joint.append(ID)
        
        # init data storing lists and variables
        self.base_frame_pos = 0
        self.base_frame_ori = 0
        self.prev_base_frame_pos = 0
        self.prev_base_frame_ori = 0
        self.EE_frame_pos = []
        self.EE_frame_ori = []
        self.prev_EE_pos = []
        self.prev_EE_ori = []
        self.prev_joint_vel = np.zeros(self.n_velocity_dimensions)
        for i in range(self.n_of_EE):
            self.EE_frame_pos.append(0)
            self.EE_frame_ori.append(0)
            self.prev_EE_pos.append(0)
            self.prev_EE_ori.append(0)
            
        # init logic parameters
        self.firstQP = True        
        
        #!TODO! add foot offset stuff
        
        #!TODO! add floating base hight stuff
        
        # time step parameters
        self.previous_time = 0
        self.dt = 0.002 #2ms
        self.step_time = 0.002
        
        #!TODO! maybe put task weights here?
        
        #!TODO! might need to put initial params here (e.g. ori and vel)
        
        # set inital pose of the robot
        self.init_joint_config = init_joint_config.copy()
        self.current_joint_config = init_joint_config.copy()
        self.updateState(init_joint_config)
        
        # enter general mode
        self.GeneralMode()
        
        # initialise WBC
        self.initialiseWBC()
        
        
    def initialiseWBC(self):
        # reset logic
        self.firstQP = True
        # initialise tasks and constraints
        for i in range(len(self.active_task_dict["task name"])):
            if self.active_task_dict["task type"][i] == "IK":
                self.active_task_dict["task instance"][i].IKInitialiseTask(self.robot_model, self.robot_data)
                
            if self.active_task_dict["task type"][i] == "CoM Cart":
                self.active_task_dict["task instance"][i].CoMCartesianInitialise(self.robot_model, self.robot_data, self.current_joint_config)

            if self.active_task_dict["task type"][i] == "CoM Stab":
                self.active_task_dict["task instance"][i].CoMStabilityInitialise(self.robot_model, self.robot_data, self.current_joint_config)
        
        
    def updateState(self, joint_config):
        #check size of joint array and adjust for manipulator joints
        if joint_config.size < self.n_configuration_dimensions:
            n_manip_joints = self.n_configuration_dimensions - joint_config.size
            manip_joints = np.zeros(n_manip_joints)
            joint_config = np.hstack((joint_config, manip_joints))
        # store previous joint config
        self.previous_joint_config = np.copy(self.current_joint_config)
        self.current_joint_config = np.copy(joint_config)
        # update model
        pin.framesForwardKinematics(self.robot_model, self.robot_data, joint_config)
        self.J = pin.computeJointJacobians(self.robot_model, self.robot_data, joint_config)
        pin.updateFramePlacements(self.robot_model, self.robot_data)
        pin.jacobianCenterOfMass(self.robot_model, self.robot_data, joint_config)
        # store frame placements
        self.base_frame_pos = np.copy(self.robot_data.oMf[self.base_frame_index].translation)
        self.base_frame_ori = np.copy(self.robot_data.oMf[self.base_frame_index].rotation)
        for i in range(self.n_of_EE):
            self.EE_frame_pos[i] = np.copy(self.robot_data.oMf[self.EE_index_list_frame[i]].translation)
            self.EE_frame_ori[i] = np.copy(self.robot_data.oMf[self.EE_index_list_frame[i]].rotation)
        
        
    def setTasks(self, task_dict): # task_dict format {"task name": list of strings, "task type": list of strings, "init parameters": list of lists}
        # add all active tasks to a dict
        self.active_task_names = []
        self.n_of_active_targets = 0
        self.active_task_dict = {"task name":[], "task type":[],"task instance":[]}
        for i in range(len(task_dict["task name"])):
            self.active_task_names.append(task_dict["task name"][i])
            self.active_task_dict["task name"].append(task_dict["task name"][i])
            self.active_task_dict["task type"].append(task_dict["task type"][i])
            if task_dict["task type"][i] == "IK":
                self.active_task_dict["task instance"].append(InverseKinematicsTask(task_dict["init parameters"][i][0], task_dict["init parameters"][i][1], task_dict["init parameters"][i][2], task_dict["init parameters"][i][3], task_dict["init parameters"][i][4]))
            if task_dict["task type"][i] == "Joint":
                self.active_task_dict["task instance"].append(JointTask(task_dict["init parameters"][i][0], task_dict["init parameters"][i][1], task_dict["init parameters"][i][2], task_dict["init parameters"][i][3], task_dict["init parameters"][i][4], task_dict["init parameters"][i][5]))
            if task_dict["task type"][i] == "CoM Cart":
                self.active_task_dict["task instance"].append(CoMCartesianTask(task_dict["init parameters"][i][0], task_dict["init parameters"][i][1]))
            if task_dict["task type"][i] == "CoM Stab":
                self.active_task_dict["task instance"].append(CoMStabilityTask(task_dict["init parameters"][i][0], task_dict["init parameters"][i][1]))
            self.n_of_active_targets = self.n_of_active_targets + self.active_task_dict["task instance"][i].n_of_targets
            
    
    
    def setConstraints(self, cstrnt_dict): # cstrnt_dict format {"constraint name": list of strings, "constraint type": list of strings, "init parameters", list of lists}
        # add all active constraints to a dictionary
        self.active_constraint_names = []
        self.n_of_active_constraints = 0
        self.active_constraint_dict = {"constraint name":[], "constraint type":[], "constraint instance":[]}
        for i in range(len(cstrnt_dict["constraint name"])):
            self.active_constraint_names.append(cstrnt_dict["constraint name"][i])
            self.active_constraint_dict["constraint name"].append(cstrnt_dict["constraint name"][i])
            self.active_constraint_dict["constraint type"].append(cstrnt_dict["constraint type"][i])
            if cstrnt_dict["constraint type"][i] == "Cart":
                self.active_constraint_dict["constraint instance"].append(CartesianConstraint(cstrnt_dict["init parameters"][i][0], cstrnt_dict["init parameters"][i][1], cstrnt_dict["init parameters"][i][2], cstrnt_dict["init parameters"][i][3], cstrnt_dict["init parameters"][i][4], cstrnt_dict["init parameters"][i][5], cstrnt_dict["init parameters"][i][6], cstrnt_dict["init parameters"][i][7], cstrnt_dict["init parameters"][i][8]))
            if cstrnt_dict["constraint type"][i] == "Local Cart":
                self.active_constraint_dict["constraint instance"].append(LocalCartesianConstraint(cstrnt_dict["init parameters"][i][0], cstrnt_dict["init parameters"][i][1], cstrnt_dict["init parameters"][i][2], cstrnt_dict["init parameters"][i][3], cstrnt_dict["init parameters"][i][4], cstrnt_dict["init parameters"][i][5], cstrnt_dict["init parameters"][i][6], cstrnt_dict["init parameters"][i][7], cstrnt_dict["init parameters"][i][8]))
            if cstrnt_dict["constraint type"][i] == "CoM Stability":
                self.active_constraint_dict["constraint instance"].append(CoMStabilityConstraint())
            if cstrnt_dict["constraint type"][i] == "Collision":
                self.active_constraint_dict["constraint instance"].append(CollisionConstraint(cstrnt_dict["init parameters"][i][0]))
            self.n_of_active_constraints = self.n_of_active_constraints + self.active_constraint_dict["constraint instance"][i].n_of_constraints
        
        

    def jointVelLimitsArray(self, initial_config=False):
        # returns an array for the upper and lower joint velocity limits which will be used for QP
        vel_lim = self.robot_model.velocityLimit
        lower_vel_lim = -vel_lim[np.newaxis]
        upper_vel_lim = vel_lim[np.newaxis]
        
        return lower_vel_lim, upper_vel_lim



    def jointPosLimitsArray(self, initial_config=False):
        # returns an array for the upper and lower joint position limits, these have been turned into velocity limits
            
        lower_pos_lim = np.copy(self.robot_model.lowerPositionLimit)
        upper_pos_lim = np.copy(self.robot_model.upperPositionLimit)
        K_lim = 0.1
        
        #!TODO! revisit when implementing for floating base 

        lower_pos_lim = (lower_pos_lim - self.current_joint_config)/self.step_time * K_lim
        upper_pos_lim = (upper_pos_lim - self.current_joint_config)/self.step_time * K_lim
        
        return lower_pos_lim.reshape(self.n_velocity_dimensions,), upper_pos_lim.reshape(self.n_velocity_dimensions,)


    def findJointConstraints(self):
        # find the joint constraints, ensuring that the upper and lower bounds do not conflict
        lb_pos, ub_pos = self.jointPosLimitsArray()
        lb_vel, ub_vel = self.jointVelLimitsArray()

        lb_vel = lb_vel.T
        ub_vel = ub_vel.T

        lb = np.zeros((lb_pos.shape[0],))
        ub = np.zeros((ub_pos.shape[0],))

        for i in range(len(lb)):
            if lb_pos[i] <= lb_vel[i]:
                lb[i] = lb_vel[i]
            else:
                lb[i] = lb_pos[i]
            if ub_pos[i] >= ub_vel[i]:
                ub[i] = ub_vel[i]
            else:
                ub[i] = ub_pos[i]
        """    
        for i in range(len(lb)):
            lb[i] = -3.142
            ub[i] = 3.142
        """        
        # remove manipulator joint limits
        for i in range(self.n_of_manip_joints):
            lb[-(i+1)] = 0
            ub[-(i+1)] = 0
            
        return lb, ub
    
        
        
    def findConstraints(self):
        # self.active_constraint_dict = {"contraint name":[], "constraint type":[], "constraint instance":[]}
        # based on the active constraints find C, Clb, and Cub required for the QP
        C = np.zeros((self.n_of_active_constraints, self.n_velocity_dimensions))
        Clb = np.zeros((self.n_of_active_constraints,))
        Cub = np.zeros((self.n_of_active_constraints,))
        append_index = 0
        for i in range(len(self.active_constraint_dict["constraint name"])):
            if self.active_constraint_dict["constraint type"][i] == "CoM Stability":
                C_tmp, Clb_tmp, Cub_tmp = self.active_constraint_dict["constraint instance"][i].CoMStabilityFindConstraint(self.robot_model, self.robot_data, self.current_joint_config, self.dt)
                C[append_index:(append_index+self.active_constraint_dict["constraint instance"][i].n_of_constraints)] = C_tmp
                Clb[append_index:(append_index+self.active_constraint_dict["constraint instance"][i].n_of_constraints)] = Clb_tmp
                Cub[append_index:(append_index+self.active_constraint_dict["constraint instance"][i].n_of_constraints)] = Cub_tmp
                append_index = append_index + self.active_constraint_dict["constraint instance"][i].n_of_constraints
            if self.active_constraint_dict["constraint type"][i] == "Cart":
                C_tmp, Clb_tmp, Cub_tmp = self.active_constraint_dict["constraint instance"][i].CartesianFindConstraint(self.robot_model, self.robot_data)
                C[append_index:(append_index+self.active_constraint_dict["constraint instance"][i].n_of_constraints)] = C_tmp
                Clb[append_index:(append_index+self.active_constraint_dict["constraint instance"][i].n_of_constraints)] = Clb_tmp
                Cub[append_index:(append_index+self.active_constraint_dict["constraint instance"][i].n_of_constraints)] = Cub_tmp
                append_index = append_index + self.active_constraint_dict["constraint instance"][i].n_of_constraints
            if self.active_constraint_dict["constraint type"][i] == "Local Cart":
                C_tmp, Clb_tmp, Cub_tmp = self.active_constraint_dict["constraint instance"][i].CartesianFindConstraint(self.robot_model, self.robot_data)
                C[append_index:(append_index+self.active_constraint_dict["constraint instance"][i].n_of_constraints)] = C_tmp
                Clb[append_index:(append_index+self.active_constraint_dict["constraint instance"][i].n_of_constraints)] = Clb_tmp
                Cub[append_index:(append_index+self.active_constraint_dict["constraint instance"][i].n_of_constraints)] = Cub_tmp
                append_index = append_index + self.active_constraint_dict["constraint instance"][i].n_of_constraints
            if self.active_constraint_dict["constraint type"][i] == "Collision":
                C_tmp, Clb_tmp, Cub_tmp = self.active_constraint_dict["constraint instance"][i].CollisionFindConstraint(self.robot_model, self.robot_data)
                if len(C_tmp) != 0:
                    C[append_index:(append_index+self.active_constraint_dict["constraint instance"][i].n_of_constraints)] = C_tmp
                    Clb[append_index:(append_index+self.active_constraint_dict["constraint instance"][i].n_of_constraints)] = Clb_tmp
                    Cub[append_index:(append_index+self.active_constraint_dict["constraint instance"][i].n_of_constraints)] = Cub_tmp
                    append_index = append_index + self.active_constraint_dict["constraint instance"][i].n_of_constraints

        return C.T, Clb, Cub
        
        
        
    def findAb(self, targets_dict): # target_dict = {"task name":[], "target pos":[], "target ori":[]}
        # self.active_task_dict = {"task name":[], "task type":[],"task instance":[]}
        # based off the active tasks find A and b required for the QP
        A = np.zeros((self.n_of_active_targets, self.n_velocity_dimensions))
        b = np.zeros((self.n_of_active_targets,1))
        append_index = 0
        for i in range(len(self.active_task_dict["task name"])):
            if self.active_task_dict["task type"][i] == "IK":
                target_index = targets_dict["task name"].index(self.active_task_dict["task name"][i])
                A_tmp, b_tmp = self.active_task_dict["task instance"][i].IKFindAb(self.robot_model, self.robot_data, targets_dict["target pos"][target_index], targets_dict["target ori"][target_index], self.dt)
                A[append_index:(append_index+self.active_task_dict["task instance"][i].n_of_targets)] = A_tmp
                b[append_index:(append_index+self.active_task_dict["task instance"][i].n_of_targets)] = b_tmp
                append_index = append_index + self.active_task_dict["task instance"][i].n_of_targets
            if self.active_task_dict["task type"][i] == "Joint":
                A_tmp, b_tmp = self.active_task_dict["task instance"][i].jointFindAb(self.robot_model, self.robot_data, self.current_joint_config, self.prev_joint_vel)
                A[append_index:(append_index+self.active_task_dict["task instance"][i].n_of_targets)] = A_tmp
                b[append_index:(append_index+self.active_task_dict["task instance"][i].n_of_targets)] = b_tmp
                append_index = append_index + self.active_task_dict["task instance"][i].n_of_targets
            if self.active_task_dict["task type"][i] == "CoM Cart":
                target_index = targets_dict["task name"].index(self.active_task_dict["task name"][i])
                A_tmp, b_tmp = self.active_task_dict["task instance"][i].CoMCartesianFindAb(self.robot_model, self.robot_data, self.current_joint_config, targets_dict["target pos"][target_index], self.dt)
                A[append_index:(append_index+self.active_task_dict["task instance"][i].n_of_targets)] = A_tmp
                b[append_index:(append_index+self.active_task_dict["task instance"][i].n_of_targets)] = b_tmp
                append_index = append_index + self.active_task_dict["task instance"][i].n_of_targets
            if self.active_task_dict["task type"][i] == "CoM Stab":
                A_tmp, b_tmp = self.active_task_dict["task instance"][i].CoMStabilityFindAb(self.robot_model, self.robot_data, self.current_joint_config, self.dt)
                A[append_index:(append_index+self.active_task_dict["task instance"][i].n_of_targets)] = A_tmp
                b[append_index:(append_index+self.active_task_dict["task instance"][i].n_of_targets)] = b_tmp
                append_index = append_index + self.active_task_dict["task instance"][i].n_of_targets

        return A, b.reshape(A.shape[0],)


    def autoConstraintConfig(self, targets_dict):
        # based on the current target position and the previous target position of a frame, if both are equal in a plane set a constraint for that plane to have zero velocity
        for i in range(len(self.active_constraint_dict["constraint name"])):
            if self.active_constraint_dict["constraint name"][i] in targets_dict["task name"] and self.active_constraint_dict["constraint instance"][i].auto == True:
                constraints_pos = [False, False, False]
                constraints_ori = [False, False, False]
                #current_constraints = self.active_constraint_dict["constraint instance"][i].constraints[3:]
                #constraints = constraints + current_constraints
                indx = targets_dict["task name"].index(self.active_constraint_dict["constraint name"][i])
                for ii in range(len(targets_dict["target pos"][indx])):
                    #if targets_dict["target pos"][indx][ii] == self.prev_target["target pos"][indx][ii]:
                        #constraints_pos[ii] = True
                    if targets_dict["target ori"][indx][ii] == self.prev_target["target ori"][indx][ii] and ii == 0:
                        constraints_ori[ii] = True      
                constraints = constraints_pos + constraints_ori
                self.active_constraint_dict["constraint instance"][i].updateConstraintParameters(self.active_constraint_dict["constraint instance"][i].frame_index, self.active_constraint_dict["constraint instance"][i].reference_frame, constraints)
        
    def runWBC(self, targets_dict, foot_force_fb=[]):
        if self.firstQP == True:
            self.prev_target = dict(targets_dict)
        # ensure sample time is maintained
        start_time = time.time()
        
        self.autoConstraintConfig(targets_dict)

        # find cartesian tasks (A and b)
        A, b = self.findAb(targets_dict)
        np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

        # find constraints
        C, Clb, Cub = self.findConstraints()
        lb, ub = self.findJointConstraints()

        # solve qp
        if self.firstQP == True:
            self.qp = QP(A, b, lb, ub, C, Clb, Cub, n_of_velocity_dimensions=self.n_velocity_dimensions)
            q_vel = self.qp.solveQP()
            self.firstQP = False
        else:
            self.qp = QP(A, b, lb, ub, C, Clb, Cub, n_of_velocity_dimensions=self.n_velocity_dimensions)
            q_vel = self.qp.solveQP()
            #q_vel = self.qp.solveQPHotstart(A, b, lb, ub, C, Clb, Cub)

        self.prev_joint_vel = q_vel

        joint_config_full = self.jointVelocitiestoConfig(q_vel)

        self.prev_target = copy.deepcopy(targets_dict)
        
        if self.n_of_manip_joints > 0:
            joint_config_no_manip = joint_config_full[:-self.n_of_manip_joints]
        else:
            joint_config_no_manip = joint_config_full
        
        while time.time() - start_time < self.step_time:
            pass
        return joint_config_no_manip
        
        
        
    def jointVelocitiestoConfig(self, joint_vel, update_model=False):
        # calculate the joint positions based on input joint velocities
        new_config = pin.integrate(self.robot_model, self.current_joint_config, joint_vel*self.dt)
        self.updateState(new_config)
        return new_config

        
        
    """ Control Modes """
    def GeneralMode(self, high_dof=True):
        self.K_pos = 0.1
        self.K_vel = 0.15

        # define tasks
        task_dict = {"task name":[], "task type":[], "init parameters":[]}
        task_names = []
        task_types = []
        task_parameters = []
        
        gain = np.identity(6)
        gain[3,3] = 0.00
        gain[4,4] = 0.00
        gain[5,5] = 0.00
        
        for i in range(self.n_of_EE):
            task_names.append(self.EE_joint_names[i])
            task_types.append("IK")
            task_parameters.append([self.EE_index_list_frame[i], 1, 1, gain, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED])
        
        for i in range(self.n_velocity_dimensions):
            task_names.append("joint " + str(i))
            task_types.append("Joint")
            task_parameters.append([i, self.n_velocity_dimensions, 0.001, "PREV", 0, self.init_joint_config])
        
        # for high dof arms, soft pose control may be required  
        if high_dof == True:
            for i in range(self.n_velocity_dimensions-self.n_of_manip_joints):
                task_names.append("joint pose" + str(i))
                task_types.append("Joint")
                task_parameters.append([i, self.n_velocity_dimensions, 0.5, "POSE", 0, self.init_joint_config])
            
        for i in range(len(task_names)):
            task_dict["task name"].append(task_names[i])
            task_dict["task type"].append(task_types[i])
            task_dict["init parameters"].append(task_parameters[i])
        
        # set tasks
        self.setTasks(task_dict)
        
        # define constraints
        cstrnt_dict = {"constraint name":[], "constraint type":[], "init parameters":[]}
        cstrnt_names = []
        cstrnt_types = []
        cstrnt_parameters = []
        for i in range(self.n_of_EE):
            cstrnt_names.append(self.EE_joint_names[i])
            cstrnt_types.append("Cart")
            cstrnt_parameters.append([self.EE_index_list_frame[i], pin.ReferenceFrame.LOCAL_WORLD_ALIGNED, False, False, False, False, False, False, False])
        for i in range(len(cstrnt_names)):
            cstrnt_dict["constraint name"].append(cstrnt_names[i])
            cstrnt_dict["constraint type"].append(cstrnt_types[i])
            cstrnt_dict["init parameters"].append(cstrnt_parameters[i])
        
        # set constraints
        self.setConstraints(cstrnt_dict)
        
        # initialise WBC in new mode
        self.initialiseWBC()
        
