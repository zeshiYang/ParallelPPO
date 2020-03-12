'''
unit test for vector environment and GAE runner
'''

import gym
from env.subproc_vec_env import SubprocVecEnv
from env.clEnv import CLEnv
from algorithm.model import *
from algorithm.runner import GAERunner
from algorithm.data_tools import PPO_Dataset
from algorithm.PPO_vec import PPO_vec


def make_env(env_name):
    return lambda : CLEnv(gym.make(env_name))

def make_vec_env(env_name, num_env):
    return  SubprocVecEnv([make_env(env_name) for i in range(num_env)])




if __name__ == "__main__":
    import json
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="training configrature file path")
    args = parser.parse_args()
    config = args.config
    with open(config, "r") as f:
        args = json.load(f)
    
    env_name = args["env_name"]
    num_envs = args["num_envs"]
    lam = args["lam"]
    gamma = args["gamma"]
    v_max = args["v_max"]
    v_min = args["v_min"]
    exp_id = args["exp_id"]
    save_dir = args["save_dir"]
    time_limit = args["time_limit"]

    vec_env = make_vec_env(env_name, num_envs)
    s_norm = Normalizer(vec_env.observation_space.shape[0])
    actor = Actor(vec_env.observation_space.shape[0], vec_env.action_space.shape[0], vec_env.action_space, hidden=[128, 64])
    critic = Critic(vec_env.observation_space.shape[0], 0/(1-gamma), v_max/(1-gamma), hidden =[128, 64])

    PPO_vec(actor=actor, critic = critic, s_norm = s_norm, vec_env = vec_env, exp_id = exp_id, save_dir= save_dir, v_max=v_max, v_min=v_min, time_limit= time_limit)
        