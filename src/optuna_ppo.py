#! /usr/bin/env python
# ---------------------------------------------------
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
import gym
import numpy
import time
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
# ROS packages required
import rospy
import rospkg
import droneTest
import torch as th
import os
import optuna

import os

import optuna
from optuna.trial import TrialState
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

DEVICE = torch.device("cuda")
BATCHSIZE = 128
CLASSES = 10
DIR = os.getcwd()
EPOCHS = 10
N_TRAIN_EXAMPLES = BATCHSIZE * 30
N_VALID_EXAMPLES = BATCHSIZE * 10


def define_model(trial):
    # We optimize the number of layers, hidden units and dropout ratio in each layer.
    n_layers_policy = trial.suggest_int("n_layers_policy", 1, 3)
    n_layers_value = trial.suggest_int("n_layers_value", 1, 3)
    policy = []
    value_function = []
    for i in range(n_layers_policy):
        neurons = trial.suggest_int("n_units_policy{}".format(i), 4, 256, log=True)
        policy.append(neurons)
    for i in range(n_layers_value):
        neurons = trial.suggest_int("n_units_value{}".format(i), 4, 256, log=True)
        value_function.append(neurons)

    gamma = trial.suggest_float("gamma", 0.9, 0.99999, log=True)
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 5.0, log=True)
    gae_lambda = trial.suggest_float("gae_lambda", 0.8, 0.999, log=True)
    # from 2**3 = 8 to 2**10 = 1024
    n_steps = 2 ** trial.suggest_int("exponent_n_steps", 3, 10, log=True)
    learning_rate = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    ent_coef = trial.suggest_float("ent_coef", 0.00000001, 0.1, log=True)
    ortho_init = trial.suggest_categorical("ortho_init", [False, True])

    # activation_fn = trial.suggest_categorical("activation_fn", [nn.Tanh, nn.ReLU])

    policy_kwargs = dict(activation_fn=nn.ReLU,
                         ortho_init=ortho_init,
                         net_arch=[dict(pi=policy, vf=value_function)])

    hyperparams = dict(
        n_steps=n_steps,
        learning_rate=learning_rate,
        gamma=gamma,  # discount factor
        gae_lambda=gae_lambda,  # Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        # Equivalent to classic advantage when set to 1.
        max_grad_norm=max_grad_norm,  # The maximum value for the gradient clipping
        ent_coef=ent_coef,  # Entropy coefficient for the loss calculation
    )
    # Create the agent
    model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1, **hyperparams)
    return model


def objective(trial):
    # Generate the model.
    model = define_model(trial)

    for epoch in range(EPOCHS):
        rospy.logwarn("start learning")
        model.learn(100)
        rospy.logwarn("learning finished, evaluation start")
        mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=1, deterministic=True)
        rospy.logwarn("evaluation finished")
        trial.report(mean_reward, epoch)
        rospy.logwarn(f"Epoch {epoch} finished")
        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    rospy.logwarn("Trial finished")
    return mean_reward


if __name__ == '__main__':

    env = gym.make("DroneTest-v0")
    rospy.loginfo("Gym environment done")

    rospy.init_node('droneTest_ppo', anonymous=True, log_level=rospy.WARN)
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('fly_bot')
    outdir = pkg_path + '/training_results'

    os.makedirs(outdir, exist_ok=True)
    env = Monitor(env, outdir)
    rospy.loginfo("Monitor Wrapper started")

    last_time_steps = numpy.ndarray(0)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100, timeout=600)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
