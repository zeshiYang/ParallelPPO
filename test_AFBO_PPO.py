from algorithm.PPO_vec import PPO_vec
from env.subproc_vec_env import SubprocVecEnv
from env.clEnv import CLEnv
import gym
from envs_coadapt.pybullet.evoenvs import HalfCheetahEnv
'''
unit test for PPO_vec class
'''


def make_env(env_name):
    return lambda : CLEnv(HalfCheetahEnv())

def make_vec_env(env_name, num_env):
    return  SubprocVecEnv([make_env(env_name) for i in range(num_env)])




if __name__ == "__main__":
    from IPython import embed
    import json
    import argparse
    import time

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
    vec_env.set_new_design([1,1,1,1,1,1])
    ppo = PPO_vec(vec_env, exp_id = exp_id, save_dir= save_dir, v_max= v_max, v_min= v_min, time_limit=time_limit)
    start_time = time.time()
    ppo.learn(num_iter = 1000)
    end_time = time.time()
    print(end_time - start_time)
    print("evaluate model:{}".format(ppo.evaluate()))