import numpy as np
from multiprocessing import Process, Pipe
from env import VecEnv, CloudpickleWrapper
'''
implementation of basic vector environment for sampling
'''
def worker(remote, parent_remote, env_fn_wrapper):

  parent_remote.close()
  env = env_fn_wrapper.x()

  try:
    while True:
      cmd, data = remote.recv()
      if cmd == 'step':
        ob, reward, done, info = env.step(data)
        if done:
          info["end_ob"] = ob
          ob = env.reset()
        remote.send((ob, reward, done, info))
      elif cmd == 'reset':
        ob = env.reset()
        remote.send(ob)
      elif cmd == 'observation_space':
        remote.send(env.observation_space)
      elif cmd == 'action_space':
        remote.send(env.action_space)
      elif cmd == 'close':
        remote.close()
        break
      elif cmd == 'set_task_t':
        env.set_task_t(data)
        remote.send(True)
      elif cmd == 'calc_sub_rewards':
        err_vec = env.calc_sub_rewards()
        remote.send(err_vec)
      else:
        raise NotImplementedError
  #except KeyboardInterrupt:
  #  print('SubprocVecEnv worker: got KeyboardInterrupt')
  finally:
    env.close()

class SubprocVecEnv(VecEnv):
  """
  VecEnv that runs multiple environments in parallel in subproceses and communicates with them via pipes.
  Recommended to use when num_envs > 1 and step() can be a bottleneck.
  """
  def __init__(self, env_fns):
    """
    Arguments:

    env_fns: iterable of callables -  functions that create environments to run in subprocesses. Need to be cloud-pickleable make_env() function
    """
    self.waiting = False
    self.closed = False
    self.num_envs = len(env_fns)
    self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(self.num_envs)])
    self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
           for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
    for p in self.ps:
      p.daemon = True  # if the main process crashes, we should not cause things to hang
      p.start()
    for remote in self.work_remotes:
      remote.close()

    self.set_task_t(1000)# set initial maximum of episodes to 1000, which is regular setting for gym task
    self.viewer = None
    #self.specs = [f().spec for f in env_fns]
    #VecEnv.__init__(self, len(env_fns), observation_space, action_space)
    VecEnv.__init__(self, len(env_fns))

    self.remotes[0].send(('observation_space', None))
    self.observation_space = self.remotes[0].recv()

    self.remotes[0].send(('action_space', None))
    self.action_space = self.remotes[0].recv()



  def step_async(self, actions):
    self._assert_not_closed()
    for remote, action in zip(self.remotes, actions):
      remote.send(('step', action))
    self.waiting = True

  def step_wait(self):
    self._assert_not_closed()
    results = [remote.recv() for remote in self.remotes]
    self.waiting = False
    obs, rews, dones, infos = zip(*results)
    return _flatten_obs(obs), np.stack(rews), np.stack(dones), infos

  def reset(self):
    self._assert_not_closed()
    for remote in self.remotes:
      remote.send(('reset', None))
    return _flatten_obs([remote.recv() for remote in self.remotes])


  def close_extras(self):
    self.closed = True
    if self.waiting:
      for remote in self.remotes:
        remote.recv()
    for remote in self.remotes:
      remote.send(('close', None))
    for p in self.ps:
      p.join()

  def _assert_not_closed(self):
    assert not self.closed, "Trying to operate on a SubprocVecEnv after calling close()"

  def set_task_t(self, t):
    self.task_t = t
    self._assert_not_closed()
    for remote in self.remotes:
      remote.send(('set_task_t', t))
    for remote in self.remotes:
      remote.recv()

  def calc_sub_rewards(self):
    self._assert_not_closed()
    for remote in self.remotes:
      remote.send(('calc_sub_rewards', None))
    self.waiting = True
    results = [remote.recv() for remote in self.remotes]
    self.waiting = False
    return _flatten_obs(results)

def _flatten_obs(obs):
  assert isinstance(obs, list) or isinstance(obs, tuple)
  assert len(obs) > 0

  if isinstance(obs[0], dict):
    import collections
    assert isinstance(obs, collections.OrderedDict)
    keys = obs[0].keys()
    return {k: np.stack([o[k] for o in obs]) for k in keys}
  else:
    return np.stack(obs)

