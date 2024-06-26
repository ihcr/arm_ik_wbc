import os
import sys
import inspect
import time
import numpy as np
import math
import copy

from scipy.spatial.transform import Rotation as R

from commutils.yaml_parser import load_yaml
import pybullet as p
import pybullet_data
from arm_ik_wbc.ik_wbc import IkWBC
from arm_ik_wbc.trajectory_planner import TrajectoryPlanner

# absolute directory of this package
rootdir = os.path.dirname(os.path.dirname(
        os.path.abspath(inspect.getfile(inspect.currentframe())))) #/home/joeyh/rob_ws/src/arm_ik_wbc

TIME_STEP = 0.002  # 500 Hz
MAX_TIME_SECS = 100  # maximum time to run the robot.

def main(argv):
    # Load configuration file
    if len(argv) == 1:
        cfg_file = argv[0]
    else:
        raise RuntimeError("Usage: python3 ./ik_wbc_arm_demo.py /<config file within root folder>")
    
    configs = load_yaml(rootdir + cfg_file)
    
    # ! Create a PyBullet simulation environment before any robots !
    p.connect(p.GUI)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0,0,-9.8)
    p.setTimeStep(1./500)
    urdfFlags = p.URDF_USE_SELF_COLLISION
    plane = p.loadURDF(rootdir+"/models/plane/plane.urdf")
    robot_path = configs["sim_robot_variables"]["urdf_filename"]
    arm = p.loadURDF(rootdir+robot_path, [0,0,0], [0.,0.,0.,1.], useFixedBase=True)
    p.resetDebugVisualizerCamera(1.20,63.65,-31.4,[0.04,0.03,0.13])
    
    # fetch robot params
    joint_des_pos = np.array(configs["control_variables"]["joint_des_pos"])
    joint_names = configs["control_variables"]["joint_names"]
    arm_urdf_path = rootdir + configs["sim_robot_variables"]["urdf_filename"]
    arm_mesh_path = rootdir + configs["sim_robot_variables"]["mesh_foldername"]
    EE_frame_name = configs["sim_robot_variables"]["limb_endeff_frame_names"]
    EE_joint_name = configs["sim_robot_variables"]["limb_endeff_joint_names"]
    base_frame_name = configs["sim_robot_variables"]["base_name"]
    
    # Initialise ik wbc
    controller = IkWBC(arm_urdf_path, arm_mesh_path, EE_frame_name, EE_frame_name, base_frame_name, joint_des_pos)
    
    # Initialise targect dict
    EE_target_pos = [0]
    EE_target_ori = [0]
    for i in range(len(controller.EE_index_list_frame)):
        EE_target_pos[i] = controller.EE_frame_pos[i].reshape(3,1)
        rot_mat = R.from_matrix(controller.EE_frame_ori[i])
        rot_euler = rot_mat.as_euler('xyz')
        EE_target_ori[i] = rot_euler.reshape(3,1)
    target_dict = {"task name":[], "target pos":[], "target ori":[]}
    target_dict["task name"] = EE_frame_name
    target_dict["target pos"] = copy.deepcopy(EE_target_pos)
    target_dict["target ori"] = copy.deepcopy(EE_target_ori)
    
    # Initialise planner
    planner = TrajectoryPlanner(copy.deepcopy(target_dict))
    jointIds = []
    for j in range(p.getNumJoints(arm)):
        p.changeDynamics(arm, j, linearDamping=0, angularDamping=0)
        info = p.getJointInfo(arm, j)
        jointName = info[1]
        jointName = jointName.decode('UTF-8')
        jointType = info[2]
        if (jointType==p.JOINT_PRISMATIC or jointType==p.JOINT_REVOLUTE):
            jointIds.append(j)
    #print(jointIds)
    while True:
        p.stepSimulation()
        
        #0
        end_target_dict = copy.deepcopy(target_dict)
        end_target_dict["target pos"][0][0] = end_target_dict["target pos"][0][0] + 0.
        #end_target_dict["target ori"][0][1] = end_target_dict["target ori"][0][1] - 0.5
        interval, traj_interval = planner.generate_trajectory(target_dict, end_target_dict)
        for i in traj_interval:
            #print(0)
            # Fetch new target dict
            target_dict = planner.run_stored_trajectory(target_dict, i)
            start_time = time.time()
            # Run WBC
            joint_des_pos = controller.runWBC(target_dict)
            #print(time.time()-start_time)
            for ii in range(len(joint_des_pos)):
                p.setJointMotorControl2(arm, jointIds[ii], p.POSITION_CONTROL, joint_des_pos[ii], force=100)
            p.stepSimulation()
            print(joint_des_pos)
        
        end_target_dict = copy.deepcopy(target_dict)
        end_target_dict["target pos"][0][0] = end_target_dict["target pos"][0][0] + 0.1
        #end_target_dict["target ori"][0][1] = end_target_dict["target ori"][0][1] + 0.5
        interval, traj_interval = planner.generate_trajectory(target_dict, end_target_dict)
        for i in traj_interval:
            # Fetch new target dict
            target_dict = planner.run_stored_trajectory(target_dict, i)
            start_time = time.time()
            # Run WBC
            joint_des_pos = controller.runWBC(target_dict)
            #print(time.time()-start_time)
            for ii in range(len(joint_des_pos)):
                p.setJointMotorControl2(arm, jointIds[ii], p.POSITION_CONTROL, joint_des_pos[ii], force=100)
            p.stepSimulation()
            print(joint_des_pos)
        
        #1
        end_target_dict = copy.deepcopy(target_dict)
        #end_target_dict["target pos"][0][0] = end_target_dict["target pos"][0][0] - 0.5
        #end_target_dict["target pos"][0][2] = end_target_dict["target pos"][0][2] + 0.5
        end_target_dict["target ori"][0][2] = end_target_dict["target ori"][0][2] - math.pi/2
        interval, traj_interval = planner.generate_trajectory(target_dict, end_target_dict)
        for i in traj_interval:
            # Fetch new target dict
            target_dict = planner.run_stored_trajectory(target_dict, i)
            
            # Run WBC
            joint_des_pos = controller.runWBC(target_dict)
        
            for ii in range(len(joint_des_pos)):
                p.setJointMotorControl2(arm, jointIds[ii], p.POSITION_CONTROL, joint_des_pos[ii], force=50)
            p.stepSimulation()
            #print(joint_des_pos)
       
        interval, traj_interval = planner.generate_homing_trajectory(target_dict)
        for i in traj_interval:
            # Fetch new target dict
            target_dict = planner.run_stored_trajectory(target_dict, i)
            start_time = time.time()
            # Run WBC
            joint_des_pos = controller.runWBC(target_dict)
            #print(time.time()-start_time)
            for ii in range(len(joint_des_pos)):
                p.setJointMotorControl2(arm, jointIds[ii], p.POSITION_CONTROL, joint_des_pos[ii], force=100)
            p.stepSimulation()
        
        #0
        end_target_dict = copy.deepcopy(target_dict)
        end_target_dict["target pos"][0][0] = end_target_dict["target pos"][0][0] + 0.
        #end_target_dict["target ori"][0][1] = end_target_dict["target ori"][0][1] - 0.5
        interval, traj_interval = planner.generate_trajectory(target_dict, end_target_dict)
        for i in traj_interval:
            #print(0)
            # Fetch new target dict
            target_dict = planner.run_stored_trajectory(target_dict, i)
            start_time = time.time()
            # Run WBC
            joint_des_pos = controller.runWBC(target_dict)
            #print(time.time()-start_time)
            for ii in range(len(joint_des_pos)):
                p.setJointMotorControl2(arm, jointIds[ii], p.POSITION_CONTROL, joint_des_pos[ii], force=100)
            p.stepSimulation()
            print(joint_des_pos)
        
        end_target_dict = copy.deepcopy(target_dict)
        end_target_dict["target pos"][0][0] = end_target_dict["target pos"][0][0] + 0.1
        #end_target_dict["target ori"][0][1] = end_target_dict["target ori"][0][1] + 0.5
        interval, traj_interval = planner.generate_trajectory(target_dict, end_target_dict)
        for i in traj_interval:
            # Fetch new target dict
            target_dict = planner.run_stored_trajectory(target_dict, i)
            start_time = time.time()
            # Run WBC
            joint_des_pos = controller.runWBC(target_dict)
            #print(time.time()-start_time)
            for ii in range(len(joint_des_pos)):
                p.setJointMotorControl2(arm, jointIds[ii], p.POSITION_CONTROL, joint_des_pos[ii], force=100)
            p.stepSimulation()
            print(joint_des_pos)
            
        
        #1
        end_target_dict = copy.deepcopy(target_dict)
        end_target_dict["target pos"][0][1] = end_target_dict["target pos"][0][1] + 0.5
        end_target_dict["target ori"][0][2] = end_target_dict["target ori"][0][2] + math.pi/4
        interval, traj_interval = planner.generate_trajectory(target_dict, end_target_dict)
        for i in traj_interval:
            # Fetch new target dict
            target_dict = planner.run_stored_trajectory(target_dict, i)
            
            # Run WBC
            joint_des_pos = controller.runWBC(target_dict)
        
            for ii in range(len(joint_des_pos)):
                p.setJointMotorControl2(arm, jointIds[ii], p.POSITION_CONTROL, joint_des_pos[ii], force=50)
            p.stepSimulation()
            #print(joint_des_pos)
            
        
        #1
        end_target_dict = copy.deepcopy(target_dict)
        end_target_dict["target pos"][0][0] = end_target_dict["target pos"][0][0] - 0.5
        end_target_dict["target ori"][0][2] = end_target_dict["target ori"][0][2] + math.pi/2
        interval, traj_interval = planner.generate_trajectory(target_dict, end_target_dict)
        for i in traj_interval:
            # Fetch new target dict
            target_dict = planner.run_stored_trajectory(target_dict, i)
            
            # Run WBC
            joint_des_pos = controller.runWBC(target_dict)
        
            for ii in range(len(joint_des_pos)):
                p.setJointMotorControl2(arm, jointIds[ii], p.POSITION_CONTROL, joint_des_pos[ii], force=50)
            p.stepSimulation()
            #print(joint_des_pos)
        
        """
        #2
        end_target_dict = copy.deepcopy(target_dict)
        end_target_dict["target pos"][0][2] = end_target_dict["target pos"][0][2] + 0.1
        interval, traj_interval = planner.generate_trajectory(target_dict, end_target_dict)
        for i in traj_interval:
            # Fetch new target dict
            target_dict = planner.run_stored_trajectory(target_dict, i)
            
            # Run WBC
            joint_des_pos = controller.runWBC(target_dict)
        
            for ii in range(len(joint_des_pos)):
                p.setJointMotorControl2(arm, jointIds[ii], p.POSITION_CONTROL, joint_des_pos[ii], force=50)
            p.stepSimulation()
            #print(joint_des_pos)
        
        #3
        end_target_dict = copy.deepcopy(target_dict)
        end_target_dict["target pos"][0][1] = end_target_dict["target pos"][0][1] -0.4
        end_target_dict["target ori"][0][2] = end_target_dict["target ori"][0][2] - math.pi/4
        interval, traj_interval = planner.generate_trajectory(target_dict, end_target_dict)
        for i in traj_interval:
            # Fetch new target dict
            target_dict = planner.run_stored_trajectory(target_dict, i)
            
            # Run WBC
            joint_des_pos = controller.runWBC(target_dict)
        
            for ii in range(len(joint_des_pos)):
                p.setJointMotorControl2(arm, jointIds[ii], p.POSITION_CONTROL, joint_des_pos[ii], force=50)
            p.stepSimulation()
            #print(joint_des_pos)
        
        #4
        end_target_dict = copy.deepcopy(target_dict)
        end_target_dict["target pos"][0][2] = end_target_dict["target pos"][0][2] - 0.1
        interval, traj_interval = planner.generate_trajectory(target_dict, end_target_dict)
        for i in traj_interval:
            # Fetch new target dict
            target_dict = planner.run_stored_trajectory(target_dict, i)
            
            # Run WBC
            joint_des_pos = controller.runWBC(target_dict)
        
            for ii in range(len(joint_des_pos)):
                p.setJointMotorControl2(arm, jointIds[ii], p.POSITION_CONTROL, joint_des_pos[ii], force=50)
            p.stepSimulation()
            #print(joint_des_pos)
         
        #5
        end_target_dict = copy.deepcopy(target_dict)
        end_target_dict["target pos"][0][1] = end_target_dict["target pos"][0][1] + 0.2
        end_target_dict["target ori"][0][2] = end_target_dict["target ori"][0][2] + math.pi/8
        interval, traj_interval = planner.generate_trajectory(target_dict, end_target_dict)
        for i in traj_interval:
            # Fetch new target dict
            target_dict = planner.run_stored_trajectory(target_dict, i)
            
            # Run WBC
            joint_des_pos = controller.runWBC(target_dict)
        
            for ii in range(len(joint_des_pos)):
                p.setJointMotorControl2(arm, jointIds[ii], p.POSITION_CONTROL, joint_des_pos[ii], force=50)
            p.stepSimulation()
            #print(joint_des_pos)
            
        end_target_dict = copy.deepcopy(target_dict)
        end_target_dict["target pos"][0][0] = end_target_dict["target pos"][0][0] - 0.1
        """
        interval, traj_interval = planner.generate_homing_trajectory(target_dict)
        for i in traj_interval:
            # Fetch new target dict
            target_dict = planner.run_stored_trajectory(target_dict, i)
            start_time = time.time()
            # Run WBC
            joint_des_pos = controller.runWBC(target_dict)
            #print(time.time()-start_time)
            for ii in range(len(joint_des_pos)):
                p.setJointMotorControl2(arm, jointIds[ii], p.POSITION_CONTROL, joint_des_pos[ii], force=100)
            p.stepSimulation()
            
        
        end_target_dict = copy.deepcopy(target_dict)
        end_target_dict["target pos"][0][0] = end_target_dict["target pos"][0][0] + 0.1
        #end_target_dict["target ori"][0][1] = end_target_dict["target ori"][0][1] + 0.5
        interval, traj_interval = planner.generate_trajectory(target_dict, end_target_dict)
        for i in traj_interval:
            # Fetch new target dict
            target_dict = planner.run_stored_trajectory(target_dict, i)
            start_time = time.time()
            # Run WBC
            joint_des_pos = controller.runWBC(target_dict)
            #print(time.time()-start_time)
            for ii in range(len(joint_des_pos)):
                p.setJointMotorControl2(arm, jointIds[ii], p.POSITION_CONTROL, joint_des_pos[ii], force=100)
            p.stepSimulation()
            print(joint_des_pos)
            
        
        #1
        end_target_dict = copy.deepcopy(target_dict)
        end_target_dict["target pos"][0][1] = end_target_dict["target pos"][0][1] - 0.5
        end_target_dict["target ori"][0][2] = end_target_dict["target ori"][0][2] - math.pi/4
        interval, traj_interval = planner.generate_trajectory(target_dict, end_target_dict)
        for i in traj_interval:
            # Fetch new target dict
            target_dict = planner.run_stored_trajectory(target_dict, i)
            
            # Run WBC
            joint_des_pos = controller.runWBC(target_dict)
        
            for ii in range(len(joint_des_pos)):
                p.setJointMotorControl2(arm, jointIds[ii], p.POSITION_CONTROL, joint_des_pos[ii], force=50)
            p.stepSimulation()
            #print(joint_des_pos)
            
        
        #1
        end_target_dict = copy.deepcopy(target_dict)
        end_target_dict["target pos"][0][0] = end_target_dict["target pos"][0][0] - 0.8
        end_target_dict["target ori"][0][2] = end_target_dict["target ori"][0][2] - math.pi/2
        interval, traj_interval = planner.generate_trajectory(target_dict, end_target_dict)
        for i in traj_interval:
            # Fetch new target dict
            target_dict = planner.run_stored_trajectory(target_dict, i)
            
            # Run WBC
            joint_des_pos = controller.runWBC(target_dict)
        
            for ii in range(len(joint_des_pos)):
                p.setJointMotorControl2(arm, jointIds[ii], p.POSITION_CONTROL, joint_des_pos[ii], force=50)
            p.stepSimulation()
            #print(joint_des_pos)
        
        interval, traj_interval = planner.generate_homing_trajectory(target_dict)
        for i in traj_interval:
            # Fetch new target dict
            target_dict = planner.run_stored_trajectory(target_dict, i)
            start_time = time.time()
            # Run WBC
            joint_des_pos = controller.runWBC(target_dict)
            #print(time.time()-start_time)
            for ii in range(len(joint_des_pos)):
                p.setJointMotorControl2(arm, jointIds[ii], p.POSITION_CONTROL, joint_des_pos[ii], force=100)
            p.stepSimulation()
            

       
    
if __name__ == "__main__":
    main(sys.argv[1:])
