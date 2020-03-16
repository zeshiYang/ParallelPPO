from gym import spaces
import numpy as np
from .pybullet_api.gym_locomotion_envs import HalfCheetahBulletEnv
import copy


class HalfCheetahEnv(object):
    #todo sha: why need this wrapper?
    #todo: a key project feature here
    #todo step(self, a): the state returned by this function is original halfcheetach concate with robot design
    def __init__(self, config = {'env' : {'render' : False, 'reward_func' : 'Normal', 'record_video': False}}):
        self._config = config
        self._render = self._config['env']['render']
        self._current_design = [1.5]*6
        self._config_numpy = np.array(self._current_design)
        self.design_params_bounds = [(0.2, 8)] * 6
        self._env = HalfCheetahBulletEnv(render=self._render, design=self._current_design)
        '''
        if self._config['ini_design_manner'] == 'original':
            self.init_sim_params = [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]
        elif self._config['ini_design_manner'] == 'random':
            self.init_sim_params = [list(np.random.uniform(low=0.8, high=2.0, size=6))]
        elif self._config['ini_design_manner'] == 'given':
            self.init_sim_params = self._config['ini_design_given']
        else:
            raise NotImplementedError
        '''
        '''
        self.init_sim_params = [
            [1.0] * 6,
            [1.41, 0.96, 1.97, 1.73, 1.97, 1.17],
            [1.52, 1.07, 1.11, 1.97, 1.51, 0.99],
            [1.08, 1.18, 1.39, 1.76 , 1.85, 0.92],
            [0.85, 1.54, 0.97, 1.38, 1.10, 1.49],
        ]
        '''
        # if self._config['rl_algorithm_config']['concate_design_2_state']:
        #     self.observation_space = spaces.Box(-np.inf, np.inf, shape=[self._env.observation_space.shape[0] + self._config['design_dim']], dtype=np.float32)#env.observation_space
        # else:
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=[self._env.observation_space.shape[0]], dtype=np.float32)
        self.action_space = self._env.action_space
        self._initial_state = self._env.reset()

    def render(self):
        pass

    def step(self, a):
        info = {}
        state, reward, done, _ = self._env.step(a)
        #if self._config['rl_algorithm_config']['concate_design_2_state']:
        #    state = np.append(state, self._config_numpy)
        info['orig_action_cost'] = 0.1 * np.mean(np.square(a))
        info['orig_reward'] = reward
        return state, reward, False, info

    def reset(self):
        state = self._env.reset()
        self._initial_state = state
        #if self._config['rl_algorithm_config']['concate_design_2_state']:
        #    state = np.append(state, self._config_numpy)
        return state

    def set_new_design(self, vec):
        #assert type(vec) == int,'env can only receive one design per once'
        print(vec)
        self._env.reset_design(vec)
        self._current_design = vec
        self._config_numpy = np.array(vec)

    def get_random_design(self):
        optimized_params = np.random.uniform(low=0.8, high=2.0, size=6)#todo, another hand-engineered constrain
        return optimized_params

    def get_current_design(self):
        return copy.copy(self._current_design)

    def get_obj_f(self, max_steps_per_episode):
        def objective(design, policy):
            pass

        return objective

