arm_ik_wbc
------

### What it is
An inverse kinematics whole-body controller (WBC) for robot arms. Any arm can be used with this controller but requires the creation of its own .yaml file, urdf, and mesh files. Follow the examples to further understand the code.

### Dependency Installation

1. Install Pinocchio if you have not done so already.
  You can use any of the ways here: https://stack-of-tasks.github.io/pinocchio/download.html
  
2. Install Klampt: https://github.com/krishauser/Klampt
  

### Package Usage
Replace `<work_folder>` with a specific workspace name, such as rob_ws.
```
mkdir -p <work_folder>/src
cd <work_folder>/src
git clone https://github.com/ihcr/commutils.git
git clone https://github.com/ihcr/limbsim.git
git clone https://github.com/ihcr/yamlutils.git
cd ..
colcon build
```
Once the code has been compiled, you can source .bash file in `install/setup.bash`
```
. install/setup.bash
```

### Running demos
For checking the package has been set up correctly:
```
cd <work_folder>/
python3 ./src/arm_ik_wbc/demos/ik_wbc_arm_static_demo.py /configs/vx300.yaml
```

To see the WBC running a simple trajectory:
```
cd <work_folder>/
python3 ./src/arm_ik_wbc/demos/ik_wbc_arm_demo.py /configs/vx300.yaml
```

### License and Copyrights

Copyright (c) 2021, University of Leeds and Harbin Institute of Technology.
BSD 3-Clause License
