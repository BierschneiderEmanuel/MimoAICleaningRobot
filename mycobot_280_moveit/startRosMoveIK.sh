#!/bin/bash
roscore
cd ./scripts
rosrun mycobot_280_moveit sync_plan.py
cd ./scripts
rosrun mycobot_280_moveit ServerIK.py 
cd ./launch
roslaunch mycobot_280_moveit demo.launch
