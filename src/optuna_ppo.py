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
import pickle as pkl

import os

import optuna
from optuna.trial import TrialState
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.visualization import plot_optimization_history, plot_param_importances
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

DEVICE = torch.device("cuda")

N_TRIALS = 100
N_STARTUP_TRIALS = 5
N_EVALUATIONS = 2
N_TIMESTEPS = int(2e4)
EVAL_FREQ = int(N_TIMESTEPS / N_EVALUATIONS)
N_EVAL_EPISODES = 5
N_EVAL_ENVS = 1
TIMEOUT = int(60 * 15)  # 15 minutes


EPOCHS = 2
ENV_ID = "DroneTest-v0"


class TrialEvalCallback(EvalCallback):
    """Callback used for evaluating and reporting a trial."""

    def __init__(
            self,
            env,
            trial: optuna.Trial,
            n_eval_episodes: int = 5,
            eval_freq: int = 10000,
            deterministic: bool = True,
            verbose: int = 0,
    ):

        super().__init__(
            eval_env=env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super()._on_step()
            self.eval_idx += 1
            self.trial.report(self.last_mean_reward, self.eval_idx)
            # Prune trial if need
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True


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
    model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=0, **hyperparams)
    return model


def objective(trial):
    print(1)
    model = define_model(trial)
    print(2)
    eval_callback = TrialEvalCallback(
        env,
        trial,
        n_eval_episodes=N_EVAL_EPISODES,
        eval_freq=EVAL_FREQ,
        deterministic=True,
    )
    print(3)
    nan_encountered = False
    print(4)
    try:
        model.learn(N_TIMESTEPS, callback=eval_callback)
    except AssertionError as e:
        # Sometimes, random hyperparams can generate NaN
        print(e)
        nan_encountered = True
    print(5)
    # Tell the optimizer that the trial failed
    if nan_encountered:
        return float("nan")
    print(6)
    if eval_callback.is_pruned:
        raise optuna.exceptions.TrialPruned()
    print(7)
    return eval_callback.last_mean_reward

# def objective(trial):
#     # Generate the model.
#     model = define_model(trial)
#
#     for epoch in range(EPOCHS):
#         rospy.logwarn("start learning")
#         model.learn(1000)
#         rospy.logwarn("learning finished, evaluation start")
#         mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=1, deterministic=True)
#         rospy.logwarn("evaluation finished")
#         trial.report(mean_reward, epoch)
#         rospy.logwarn(f"Epoch {epoch} finished")
#         # Handle pruning based on the intermediate value.
#         if trial.should_prune():
#             raise optuna.exceptions.TrialPruned()
#     rospy.logwarn("Trial finished")
#     return mean_reward

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

    sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS)
    # Do not prune before 1/3 of the max budget is used
    pruner = MedianPruner(
        n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=N_EVALUATIONS // 3
    )
    study = optuna.create_study(sampler=sampler, pruner=pruner, direction="maximize")
    study.optimize(objective, n_trials=N_TRIALS, timeout=TIMEOUT)

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
    # Write report
    study.trials_dataframe().to_csv("study_results_a2c_cartpole.csv")

    with open("study.pkl", "wb+") as f:
        pkl.dump(study, f)

    fig1 = plot_optimization_history(study)
    fig2 = plot_param_importances(study)

    fig1.show()
    fig2.show()
