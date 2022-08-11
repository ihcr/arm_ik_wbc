import pybullet as p
import numpy as np
from arm_ik_wbc.ik_wbc import IkWBC

p.connect(p.GUI)
plane = p.loadURDF("/home/joeyh/wbc_ws/src/TeLeMan-teleman_final_demo/Robot_Descriptions/urdf/plane.urdf")
p.setGravity(0,0,-9.8)
p.setTimeStep(1./500)
urdfFlags = p.URDF_USE_SELF_COLLISION

# urdf params
arm_urdf_path = "/home/joeyh/rob_ws/src/arm_ik_wbc/models/arm/dofarm.urdf"
arm_mesh_path = "/home/joeyh/rob_ws/src/arm_ik_wbc/models/arm/dofarm"
EE_frame_name = ["wrist2"]
EE_joint_name = ["wrist_wrist2"]
base_frame_name = "base_waist"
init_joint_config = np.array([0,0,0,0,0,0])

LeggedRobot_bullet = p.loadURDF(arm_urdf_path,[0,0,0],[0,0,0,1], flags=urdfFlags, useFixedBase=True)
jointIds = []
for j in range(p.getNumJoints(LeggedRobot_bullet)):
    p.changeDynamics(LeggedRobot_bullet, j, linearDamping=0, angularDamping=0)
    info = p.getJointInfo(LeggedRobot_bullet, j)
    jointName = info[1]
    jointName = jointName.decode('UTF-8')
    jointType = info[2]
    if (jointType==p.JOINT_PRISMATIC or jointType==p.JOINT_REVOLUTE):
        jointIds.append(j)
    
joints_py = np.array([0,0,0,0,0,0])
    
#for i in range(len(joints_py)):
#    p.setJointMotorControl2(LeggedRobot_bullet, jointIds[i], p.POSITION_CONTROL, joints_py[i])

p.setJointMotorControlArray(LeggedRobot_bullet, jointIds, p.POSITION_CONTROL, joints_py)

   
p.setRealTimeSimulation(1)

# wbc setup
controller = IkWBC(arm_urdf_path, arm_mesh_path, EE_frame_name, EE_joint_name, base_frame_name, init_joint_config)

while(1):
    #for i in range(len(joints_py)):
    #    p.setJointMotorControl2(LeggedRobot_bullet, jointIds[i], p.POSITION_CONTROL, joints_py[i])
    p.setJointMotorControlArray(LeggedRobot_bullet, jointIds, p.POSITION_CONTROL, joints_py)
    
p.disconnect()
