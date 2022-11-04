#! /usr/bin/env python
# ---------------------------------------------------
from stable_baselines3 import PPO

import gym
import numpy
import time
from gym import wrappers
# ROS packages required
import rospy
import rospkg
import droneTest
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Point
from tf.transformations import quaternion_from_euler


def set_init_pose(pub_init_pos):  # done

    state_msg = ModelState()
    state_msg.model_name = 'Kwad'
    pose = Point()
    pose.x = 1
    pose.y = 1
    pose.z = 0
    state_msg.pose.position = pose
    normal_orientation = [0, 0, 0]
    denormalized_roll = normal_orientation[0]
    denormalized_pitch = normal_orientation[1]
    denormalized_yaw = normal_orientation[2]
    orientation = quaternion_from_euler(denormalized_roll, denormalized_pitch, denormalized_yaw)
    state_msg.pose.orientation.x = orientation[0]
    state_msg.pose.orientation.y = orientation[1]
    state_msg.pose.orientation.z = orientation[2]
    state_msg.pose.orientation.w = orientation[3]

    # rospy.wait_for_service('/gazebo/set_model_state')
    # try:
    #     set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
    #     resp = set_state(state_msg)
    #     rospy.logerr(f"status of set init pose: {resp}")
    # except rospy.ServiceException as e:
    #     print("Service call failed: %s" % e)
    try:
        pub_init_pos.publish(state_msg)
    except rospy.ServiceException as e:
        print("publish on set model states call failed: %s" % e)


if __name__ == '__main__':
    # env = gym.make("DroneTest-v0")
    # rospy.loginfo("Gym environment done")
    #
    # rospy.init_node('droneTest_ppo', anonymous=True, log_level=rospy.WARN)
    # rospack = rospkg.RosPack()
    # pkg_path = rospack.get_path('fly_bot')
    # outdir = pkg_path + '/training_results'
    # # env = wrappers.Monitor(env, outdir, force=True)
    # rospy.loginfo("Monitor Wrapper started")
    #
    # last_time_steps = numpy.ndarray(0)
    #
    # # Loads parameters from the ROS param server
    # # Parameters are stored in a yaml file inside the config directory
    # # They are loaded at runtime by the launch file
    # Alpha = rospy.get_param("/drone/alpha")
    # Epsilon = rospy.get_param("/drone/epsilon")
    # Gamma = rospy.get_param("/drone/gamma")
    # epsilon_discount = rospy.get_param("/drone/epsilon_discount")
    # nepisodes = rospy.get_param("/drone/nepisodes")
    # nsteps = rospy.get_param("/drone/nsteps")
    #
    #
    # # model = PPO("MlpPolicy", env, verbose=1)
    # # model.learn(total_timesteps=20_000)
    # #
    # # obs = env.reset()
    # # for i in range(1000):
    # #     action, _states = model.predict(obs, deterministic=True)
    # #     obs, reward, done, info = env.step(action)
    # #     env.render()
    # #     if done:
    # #         obs = env.reset()
    #
    # # model.save(outdir)
    # env.close()
    rospy.init_node('set_pose')
    pub_init_pos = rospy.Publisher("/gazebo/set_model_state", ModelState, queue_size=1)
    set_init_pose(pub_init_pos)
