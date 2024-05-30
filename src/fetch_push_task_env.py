#! /usr/bin/env python

from gym import utils
import math
import rospy
from gym import spaces
from fetch_robot_env import FetchEnv
from gym.envs.registration import register
import numpy as np
from cube_positions import Obj_Pos

max_episode_steps = 1000 

register(
        id='FetchPush-v0',
        entry_point='fetch_push_task_env:FetchPushEnv',
        max_episode_steps=max_episode_steps)

class FetchPushEnv(FetchEnv, utils.EzPickle):
    """
    task env object for teaching fetch how to push the cube obj
    """
    def __init__(self):
        
        self.obj_positions = Obj_Pos(object_name="demo_cube")
        self.get_params()
        FetchEnv.__init__(self)
        utils.EzPickle.__init__(self)
        self.gazebo.unpauseSim()

        # action space has 7 dim corresponding to each arm joint
        self.action_space = spaces.Box(
            low=self.position_joints_min,
            high=self.position_joints_max, shape=(self.n_actions,),
            dtype=np.float32
        )

        # observation space has 3 dim (distance of obj from ee, speed of obj, ee z position)
        observations_high_dist = np.array([self.max_distance])
        observations_low_dist = np.array([0.0])

        observations_high_speed = np.array([self.max_speed])
        observations_low_speed = np.array([0.0])

        observations_ee_z_max = np.array([self.ee_z_max])
        observations_ee_z_min = np.array([self.ee_z_min])

        high = np.concatenate([observations_high_dist, observations_high_speed, observations_ee_z_max])
        low = np.concatenate([observations_low_dist, observations_low_speed, observations_ee_z_min])
        self.observation_space = spaces.Box(low, high)

        obs = self.get_obs()
        

    def get_params(self):
        """
        get configuration parameters

        """
        self.sim_time = rospy.get_time()
        self.n_actions = 7
        self.position_ee_max = 10.0
        self.position_ee_min = -10.0
        self.position_joints_max = 2.16
        self.position_joints_min = -2.16

        self.init_pos = {
                "joint0": 0.0,
                "joint1": -0.8,
                "joint2": 0.0,
                "joint3": 1.6,
                "joint4": 0.0,
                "joint5": 0.8,
                "joint6": 0.0}
        
        self.setup_ee_pos = {"x": 0.598,
                            "y": 0.005,
                            "z": 0.9}


        self.position_delta = 0.1
        self.step_punishment = -1
        self.closer_reward = 10
        self.impossible_movement_punishement = -100
        self.reached_goal_reward = 100

        self.max_distance = 3.0
        self.max_speed = 1.0
        self.ee_z_max = 1.0
        self.ee_z_min = 0.3



    def set_init_pose(self):
        """
        Set the Robot in its init pose
        """
        self.gazebo.unpauseSim()
        if not self.set_trajectory_joints(self.init_pos):
            assert False, "Initialisation is failed...."

    def init_env_variables(self):
        """
        Init variables needed to be initialised each reset time 
        """
        rospy.logdebug("Init Env Variables...")
        rospy.logdebug("DONE")

    def set_action(self, action):

        self.new_pos = {
                "joint0": action[0],
                "joint1": action[1],
                "joint2": action[2],
                "joint3": action[3],
                "joint4": action[4],
                "joint5": action[5],
                "joint6": action[6]}
        # call set_trajectory method from fetch_robot_env
        self.movement_result = self.set_trajectory_joints(self.new_pos)

    def calc_dist(self,p1,p2):
        """
        helper func for get_obs distance calculations
        """
        x_d = math.pow(p1[0] - p2[0],2)
        y_d = math.pow(p1[1] - p2[1],2)
        z_d = math.pow(p1[2] - p2[2],2)
        d = math.sqrt(x_d + y_d + z_d)

        return d

    def get_obs(self):
        """
        return the observations
        """
        self.gazebo.unpauseSim()
        grip_pose = self.get_ee_pose()
        ee_array_pose = [grip_pose.position.x, grip_pose.position.y, grip_pose.position.z]

        # the position of the cube on a table        
        object_data = self.obj_positions.get_states()
        # speed of the cube
        object_pos = object_data[3:]
        distance_from_cube = self.calc_dist(object_pos,ee_array_pose)
        object_velp = object_data[-3:]
        speed = np.linalg.norm(object_velp)

        # We state as observations the distance form cube, the speed of cube and the z postion of the end effector
        observations_obj = np.array([distance_from_cube,
                             speed, ee_array_pose[2]])
        return  observations_obj
    
    def get_elapsed_time(self):
        """
        Return the elapsed time since the beginning of the simulation
        """
        current_time = rospy.get_time()
        dt = self.sim_time - current_time
        self.sim_time = current_time
        return dt

    def is_done(self, observations):

        # get current observations
        speed = observations[1]

        # define fail and success of the action
        done_fail = not(self.movement_result)
        done_sucess = speed >= self.max_speed
        done = done_fail or done_sucess
        return done

    def compute_reward(self, observations, done):
        """
        Reward for moving the cube, Punish for moving to unreachable positions
        Calculate the reward: binary => 1 for success, 0 for failure
        """
        distance = observations[0]
        speed = observations[1]
        ee_z_pos = observations[2]

        done_fail = not(self.movement_result)
        done_sucess = speed >= self.max_speed

        if done_fail:
            reward = self.impossible_movement_punishement
        else:
            if done_sucess:
                reward = -1*self.impossible_movement_punishement
            else:
                if ee_z_pos < self.ee_z_min or ee_z_pos >= self.ee_z_max:
                    print("Punish, ee z too low or high")
                    reward = self.impossible_movement_punishement / 4.0
                else:
                    # It didnt move the cube. We reward it by getting closer
                    print("Reward for getting closer")
                    reward = 1.0 / distance
        return reward