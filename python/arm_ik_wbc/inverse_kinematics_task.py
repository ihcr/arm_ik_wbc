""" References:
[1] Carpentier, J., Valenza, F., and Mansard, N. Pinocchio (2.6.1). [Software]. 2021. [Accessed 23 June 2021].
[2] Virtanen, P., Gommer, R., Oliphant, T. E. et al. SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python. Nature Methods. 2020, 17, pp.261-272
[3] Harris, C. R., Millman, K. J., van der Walt, S. J. et al. Array Programming with NumPy. Nature. 2020, 585, pp.357-362
"""

import pinocchio as pin #[1]
from scipy.spatial.transform import Rotation as R #[2]
import numpy as np #[3]
import math
import time
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
        target_ori_local = np.copy(target_ori)
        # Find the position of the frame through forward kinematics
        fk_pos = np.copy(self.robot_data.oMf[self.frame_index].translation).reshape(3,1)
        #fk_ori = R.from_matrix(np.copy(self.robot_data.oMf[self.frame_index].rotation))
        # Find base desired velocity not accounting for residual
        ref_vel = (target_pos - self.prev_pos)/dt
        pos_vel = ref_vel + (np.dot(self.b_pos_gain, ((target_pos - fk_pos)/dt)))

        # Find the angular velocity
        # Check for Euler wrapping
        #for i in range(len(target_ori)):
        #    if target_ori_local[i] >= math.pi/2:
        #        target_ori_local[i] = target_ori_local[i] - (math.pi/2)
        
        fk_ori = R.from_matrix(np.copy(self.robot_data.oMf[self.frame_index].rotation)) # rot mat at t
        
        """ quat method """
        #fk_ori = fk_ori.as_quat() # q at t
        #target_ori_q = R.from_euler('xyz', target_ori.reshape(3,)) # euler at t+1
        #target_ori_q = target_ori_q.as_quat().reshape(4,) # q at t+1
        #prev_ori = R.from_euler('xyz', self.prev_ori.reshape(3,))
        #prev_ori_q = prev_ori.as_quat()
        #w = self.ang_vel_from_quat(fk_ori, target_ori_q, 1)
        #w = np.dot(w, self.b_ori_gain)
        #w_vel = self.ang_vel_from_quat(prev_ori_q, target_ori_q, dt)
        #ori_vel = (w_vel + w).reshape(3,1)

        """ euler method """
        #fk_ori_euler = fk_ori.as_euler('xyz')
        #ori_error = ((target_ori.reshape(3,) - fk_ori_euler)).reshape(3,)
        #w = np.dot(ori_error, self.b_ori_gain)
        #w_vel = (target_ori.reshape(3,) - self.prev_ori.reshape(3,))/dt
        #w_vel = np.dot(w_vel, self.b_ori_gain)
        #ori_vel = (w_vel+w).reshape(3,1)
        
        """ rotation mat method """
        fk_ori_mat = fk_ori.as_matrix()
        target_ori_mat = R.from_euler('xyz', target_ori.reshape(3,))
        target_ori_mat = target_ori_mat.as_matrix()
        ori_vel = self.rotationPD(target_ori_mat, fk_ori_mat).reshape(3,1)
        

        # Store previous values
        self.prev_pos = target_pos
        self.prev_ori = target_ori

        # Find overall target velocity
        target_vel = np.concatenate((pos_vel, ori_vel.reshape(3,1)), axis=0)
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
            
    def quat_conj(self, quat):
        # in the order of x, y, z, w
        return np.array([-quat[0], -quat[1], -quat[2], quat[3]])
    
    def quat_mag(self, quat):
        return math.sqrt(pow(quat[0],2) + pow(quat[1],2) + pow(quat[2],2) + pow(quat[3],2))
    
    def quat_inv(self, quat):
        return self.quat_conj(quat) / self.quat_mag(quat)
        
    def ang_vel_from_quat(self, q1, q2, dt):
        return (2 / dt) * np.array([
        q1[3]*q2[0] - q1[0]*q2[3] - q1[1]*q2[2] + q1[2]*q2[1],
        q1[3]*q2[1] + q1[0]*q2[2] - q1[1]*q2[3] - q1[2]*q2[0],
        q1[3]*q2[2] - q1[0]*q2[1] + q1[1]*q2[0] - q1[2]*q2[3]])
        
    
    def vec_to_so3(self, omg):
        """Converts a 3-vector to an so(3) representation
        Args:
            omg (ndarray): A 3-vector
        Returns:
            A 3x3 skew-symmetric matrix
        Example Input:
            omg = np.array([1, 2, 3])
        Output:
            np.array([[ 0, -3,  2],
                      [ 3,  0, -1],
                      [-2,  1,  0]])
        """
        return np.array([[0,      -omg[2],  omg[1]],
                         [omg[2],       0, -omg[0]],
                         [-omg[1], omg[0],       0]])
    
    def so3_to_vec(self, so3mat):
        """Converts an so(3) representation to a 3-vector
        
        Args:
            so3mat: A 3x3 skew-symmetric matrix
        Returns: 
            The 3-vector corresponding to so3mat
        Example Input:
            so3mat = np.array([[ 0, -3,  2],
                               [ 3,  0, -1],
                               [-2,  1,  0]])
        Output:
            np.array([1, 2, 3])
        """
        return np.array([so3mat[2][1], so3mat[0][2], so3mat[1][0]])

    def matrix_log3(self, R):
        """Computes the matrix logarithm of a rotation matrix
        
        Args:
            mat: A 3x3 rotation matrix
        Returns:
            The matrix logarithm of R
        Example Input:
            R = np.array([[0, 0, 1], 
                          [1, 0, 0], 
                          [0, 1, 0]])
        Output: 
            np.array([[          0, -1.20919958,  1.20919958],
                      [ 1.20919958,           0, -1.20919958],
                      [-1.20919958,  1.20919958,           0]])
        """
        acosinput = (np.trace(R) - 1) / 2.0
        if acosinput >= 1:
            return np.zeros((3, 3))
        elif acosinput <= -1:
            if not near_zero(1 + R[2][2]):
                omg = (1.0 / np.sqrt(2 * (1 + R[2][2]))) \
                      * np.array([R[0][2], R[1][2], 1 + R[2][2]])
            elif not near_zero(1 + R[1][1]):
                omg = (1.0 / np.sqrt(2 * (1 + R[1][1]))) \
                      * np.array([R[0][1], 1 + R[1][1], R[2][1]])
            else:
                omg = (1.0 / np.sqrt(2 * (1 + R[0][0]))) \
                      * np.array([1 + R[0][0], R[1][0], R[2][0]])
            return vec_to_so3(np.pi * omg)
        else:
            theta = np.arccos(acosinput)
            return theta / 2.0 / np.sin(theta) * (R - np.array(R).T)
        
        
    def rotationPD(self, des_rot, cur_rot, 
                   des_omega=np.zeros(3), cur_omega=np.zeros(3), 
                   des_omega_dot=np.zeros(3), 
                   kp=np.array([100, 100, 100]), kd=np.array([0.0, 0.0, 0.0])):
        return (kp * cur_rot.dot(self.so3_to_vec(self.matrix_log3(cur_rot.T.dot(des_rot)))) + 
            kd * (des_omega - cur_omega) + des_omega_dot)
        
        
