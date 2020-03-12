'''
time based curriculum environment wrapper
'''
from gym import spaces
import numpy as np
import copy


class CLEnv(object):
    '''
    curriculum learning environment for locomotion tasks
    '''
    def __init__(self, env):
        self._env = env
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        self._initial_state = self._env.reset()
        self.current_step = 0
        self._task_t = 1000 #standard settings for gym based tasks

    def render(self):
        pass

    def step(self, a):
        state, reward, done, info = self._env.step(a)
        self.current_step +=1
        info['fail'] = done
        if(self.current_step > self._task_t):
            done = True
        return state, reward, done, info

    def reset(self):
        state = self._env.reset()
        self._initial_state = state
        self.current_step = 0
        return state
    def set_task_t(self, t):
        """ Set the max t an episode can have under training mode for curriculum learning
        """
        self._task_t = min(t, 1000)

