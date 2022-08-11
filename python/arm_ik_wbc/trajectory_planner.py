import time
from klampt.model import trajectory
import copy
import numpy as np

class TrajectoryPlanner:
    def __init__(self, init_dict):
        self.init_dict = copy.deepcopy(init_dict)
        self.init_pos = copy.deepcopy(init_dict["target pos"])
        self.init_ori = copy.deepcopy(init_dict["target ori"])
        self.n_of_control_frames = len(init_dict["task name"])
        
    def set_init_dict(self, current_dict):
        self.init_dict = copy.deepcopy(current_dict)
        self.init_pos = copy.deepcopy(self.init_dict["target pos"])
        self.init_ori = copy.deepcopy(self.init_dict["target ori"])
        
    def fetch_init_dict(self):
        return copy.deepcopy(self.init_dict)
        
    def fetch_init_pos(self):
        return copy.deepcopy(self.init_pos)
        
    def fetch_init_ori(self):
        return copy.deepcopy(self.init_ori)
        
    def generate_trajectory(self, current_dict, target_dict):
        milestones_list_pos = []
        milestones_list_ori = []
        self.traj_list_pos = []
        self.traj_list_ori = []
        interval = 0.005
        
        
        # find milestones for each frame
        for i in range(self.n_of_control_frames):
            milestones_list_pos.append([current_dict["target pos"][i], target_dict["target pos"][i]])
            milestones_list_ori.append([current_dict["target ori"][i], target_dict["target ori"][i]])
            
        # generate trajectories for each frame
        for i in range(self.n_of_control_frames):
            self.traj_list_pos.append(trajectory.Trajectory(milestones=milestones_list_pos[i]))
            self.traj_list_ori.append(trajectory.Trajectory(milestones=milestones_list_ori[i]))
            
        traj_interval = np.arange(0, 2, interval).tolist()
            
        return interval, traj_interval
        
    def generate_multi_trajectory(self, current_dict, target_dict_list):
        milestones_list_pos = []
        milestones_list_ori = []
        self.traj_list_pos = []
        self.traj_list_ori = []
        interval = 0.008
        
        # set milestones for each frame
        for i in range(self.n_of_control_frames):
            milestones_list_pos.append([current_dict["target pos"][i]])
            milestones_list_ori.append([current_dict["target ori"][i]])
            for ii in range(len(target_dict_list)):
                milestones_list_pos[i].append(target_dict_list[ii]["target pos"][i])
                milestones_list_ori[i].append(target_dict_list[ii]["target ori"][i])
        
        # generate trajectories for each frame
        for i in range(self.n_of_control_frames):
            self.traj_list_pos.append(trajectory.Trajectory(milestones=milestones_list_pos[i]))
            self.traj_list_ori.append(trajectory.Trajectory(milestones=milestones_list_ori[i]))
            
        traj_interval = np.arange(0, (len(target_dict_list)+1), interval).tolist()
            
        return interval, traj_interval
            
    
    def run_stored_trajectory(self, target_dict, step):
        # find new target dict
        for i in range(self.n_of_control_frames):
            target_dict["target pos"][i] = np.array(self.traj_list_pos[i].eval(step)).reshape(3,1)
            target_dict["target ori"][i] = np.array(self.traj_list_ori[i].eval(step)).reshape(3,1)
        
        return target_dict
        
    def generate_homing_trajectory(self, current_dict):
        local_current_dict = copy.deepcopy(current_dict)
        interval, traj_interval = self.generate_trajectory(local_current_dict, self.init_dict)
        
        return interval, traj_interval
        
        
        
        
        
        
        
        
