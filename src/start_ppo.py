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

if __name__ == '__main__':

    env = gym.make("DroneTest-v0")
    rospy.loginfo("Gym environment done")

    rospy.init_node('droneTest_qlearn', anonymous=True, log_level=rospy.WARN)
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('fly_bot')
    outdir = pkg_path + '/training_results'
    env = wrappers.Monitor(env, outdir, force=True)
    rospy.loginfo("Monitor Wrapper started")

    last_time_steps = numpy.ndarray(0)

    # Loads parameters from the ROS param server
    # Parameters are stored in a yaml file inside the config directory
    # They are loaded at runtime by the launch file
    Alpha = rospy.get_param("/drone/alpha")
    Epsilon = rospy.get_param("/drone/epsilon")
    Gamma = rospy.get_param("/drone/gamma")
    epsilon_discount = rospy.get_param("/drone/epsilon_discount")
    nepisodes = rospy.get_param("/drone/nepisodes")
    nsteps = rospy.get_param("/drone/nsteps")

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=20_000)
    #
    # obs = env.reset()
    # for i in range(1000):
    #     action, _states = model.predict(obs, deterministic=True)
    #     obs, reward, done, info = env.step(action)
    #     env.render()
    #     if done:
    #         obs = env.reset()

    model.save(outdir)
    env.close()
