#!/usr/bin/env python

import numpy as np
import rospy
from gazebo_msgs.srv import GetWorldProperties, GetModelState

class Obj_Pos(object):
    """
    get info of the position of the cube to calculate rewards
    """

    def __init__(self, object_name):
        self.object_name = object_name
        world_specs = rospy.ServiceProxy(
            '/gazebo/get_world_properties', GetWorldProperties)()
        self.time = 0
        self.model_names = world_specs.model_names
        self.get_model_state = rospy.ServiceProxy(
            '/gazebo/get_model_state', GetModelState)

    def get_states(self):
        """
        Return the ndarray of poseition  and rotation of the cube
        """
        for model_name in self.model_names:
            if model_name == self.object_name:
                data = self.get_model_state(
                    model_name, "world")  
                return np.array([
                    data.pose.position.x,
                    data.pose.position.y,
                    data.pose.position.z,
                    data.twist.linear.x,
                    data.twist.linear.y,
                    data.twist.linear.z
                ])


if __name__ == "__main__":
    rospy.init_node("Cube")
    obj_positions = Obj_Pos(object_name="demo_cube")
    st = obj_positions.get_states()
    r = rospy.Rate(5.0)
    while not rospy.is_shutdown():
        st = obj_positions.get_states()
        print(st)
        r.sleep()
        