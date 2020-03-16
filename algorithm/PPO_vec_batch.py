from .PPO_vec import PPO_vec



class PPOVecBatch(object):

    def __init__(self, config, batch_index, vec_env, batch_size):
        self.batch_size = batch_size
        self._algorithm_inds = dict()
        for i in range(self.batch_size):
            self._algorithm_inds[i] = PPO_vec(
                vec_env = vec_env,
                exp_id = self.batch_size * batch_index+i,
                save_dir = config['save_dir'],
                v_max = config['v_max'],
                v_min = config['v_min'],
                time_limit= config['time_limit']
            )

    def single_train_step(self, design_index):
        self._algorithm_inds[design_index]._try_to_train()
