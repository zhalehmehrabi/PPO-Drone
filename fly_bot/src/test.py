#! /usr/bin/env python

from gazebo_msgs.srv import GetModelState
from geometry_msgs.msg import Point
import rospy
import numpy as np

class Block:
    def __init__(self, name, relative_entity_name):
        self._name = name
        self._relative_entity_name = relative_entity_name

class Tutorial:

    _blockListDict = {
        'block_a': Block('Kwad', 'base_link2'),
    }

    def show_gazebo_models(self):
        # try:
        #     model_coordinates = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        #     resp_coordinates = model_coordinates('Kwad', 'plate1')
        #     print('\n')
        #     print('Status.success = ', resp_coordinates.success)
        #     print(resp_coordinates.pose.position)
        # except rospy.ServiceException as e:
        #     rospy.loginfo("Get Model State service call failed:  {0}".format(e))
        rospy.logwarn("Start Getting observation")
        odom = None
        try:
            model_coordinates = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
            odom = model_coordinates("Kwad", "")
            print(odom)
        except rospy.ServiceException as e:
            print(e)


if __name__ == '__main__':
    tuto = Tutorial()
    tuto.show_gazebo_models()