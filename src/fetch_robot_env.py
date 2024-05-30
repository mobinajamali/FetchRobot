#! /usr/bin/env python

import numpy as np
import rospy
from openai_ros import robot_gazebo_env_goal
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
from fetch_moveit_config.fetch_commander import FetchCommander
from geometry_msgs.msg import Pose

class FetchEnv(robot_gazebo_env_goal.RobotGazeboEnv):

    def __init__(self):
        """
        Initialize a new Fetch environment
        """
        self.joints = JointState()
        self.joint_states_sub = rospy.Subscriber('/joint_states', JointState, self.joints_callback)

        # interface with moveit
        self.fetch_commander_obj = FetchCommander()

        self.controllers_list = []
        self.robot_name_space = ""
        
        # launch the init function of the Parent Class robot_gazebo_env_goal.RobotGazeboEnv
        super(FetchEnv, self).__init__(controllers_list=self.controllers_list,
                                                robot_name_space=self.robot_name_space,
                                                reset_controls=False)

    def joints_callback(self, data):
        self.joints = data

    def get_joints(self):
        return self.joints

    def _check_all_systems_ready(self):
        """
        Check the systems are operational
        """
        self._check_all_sensors_ready()
        return True

    def _check_all_sensors_ready(self):

        self._check_joint_states_ready()
        rospy.logdebug("ALL SENSORS READY")

    def _check_joint_states_ready(self):

        self.joints = None
        while self.joints is None and not rospy.is_shutdown():
            try:
                self.joints = rospy.wait_for_message("/joint_states", JointState, timeout=1.0)
                rospy.logdebug("Current /joint_states READY=>" + str(self.joints))

            except:
                rospy.logerr("Current /joint_states not ready yet, retrying for getting joint_states")
        return self.joints

    def get_ee_pose(self):
        """
        get x,y,z position of end effector
        """
        gripper_pose = self.fetch_commander_obj.get_ee_pose()
        return gripper_pose
        
    def get_ee_rpy(self):
        """
        get roll,pitch, yaw of end effector
        """
        gripper_rpy = self.fetch_commander_obj.get_ee_rpy()
        return gripper_rpy

    def set_trajectory_ee(self, action):
        """
        Set the end effector position and orientation
        """       
        pose = Pose()
        pose.position.x = action[0]
        pose.position.y = action[1]
        pose.position.z = action[2]
        pose.orientation.x = 0.0
        pose.orientation.y = 0.0
        pose.orientation.z = 0.0
        pose.orientation.w = 1.0

        try:
            self.fetch_commander_obj.move_ee_to_pose(pose)
            result = True
        except Exception as e:
            print(e)
            result = False
        return result
        
    def set_trajectory_joints(self, initial_qpos):
        """
        Set the joint position
        """
        position = [None] * 7
        position[0] = initial_qpos["joint0"]
        position[1] = initial_qpos["joint1"]
        position[2] = initial_qpos["joint2"]
        position[3] = initial_qpos["joint3"]
        position[4] = initial_qpos["joint4"]
        position[5] = initial_qpos["joint5"]
        position[6] = initial_qpos["joint6"]

        try:
            self.fetch_commander_obj.move_joints_traj(position)
            result = True
        except Exception as e:
            print(e)
            result = False
        return result
        

    
    # ABSTRACT METHODS FOR TASK ENV TO BE DEFINED 
    def _init_env_variables(self):
        raise NotImplementedError()

    def _compute_reward(self, observations, done):
        raise NotImplementedError()

    def _set_action(self, action):
        raise NotImplementedError()

    def _get_obs(self):
        raise NotImplementedError()

    def _is_done(self, observations):
        raise NotImplementedError()