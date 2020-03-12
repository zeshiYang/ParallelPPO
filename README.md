# ParallelPPO
Parallel implementation of PPO algorithm for deep reinforcement learning(PyTorch)
## Getting started
  Use command python test.py --config config.json to run code
  You need to specify parameters in config.json:
    env_name: environment name in openai mujoco tasks, if not in openai gym, please change make_env() in test.py
    num_env: # of parallel environments to sample data
    lam: parameter in GAE
    gamma: parameter in GAE
    v_max: maximum of reward in a single step
    v_min: miminum of reward in a single step
    exp_id: id of experiments
    time_limit: maximum of steps in environment, designed for curriculum learning
    save_dir: directory of saved models and logs
