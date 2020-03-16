import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import torch.optim as optim
import torch
from tensorboardX import SummaryWriter
from .runner import GAERunner
from .data_tools import PPO_Dataset
import numpy as np
import time
from model import *

class PPO_vec(object):

    def __init__(self,vec_env, exp_id,
        save_dir="./experiments",
        sample_size=4096,
        epoch_size=10,
        batch_size=256,
        checkpoint_batch=10,
        test_batch=10,
        gamma=0.99,
        lam=0.95,
        clip_threshold=0.2,
        actor_lr=3e-4,
        critic_lr=3e-4,
        max_grad_norm=0.5,
        v_max = 1,
        v_min = 0,
        time_limit = 1000,
        use_gpu_model=False):

        #log parameters
        self.save_dir = save_dir
        self.checkpoint_batch = checkpoint_batch
        self.test_batch = test_batch
        self.exp_id = exp_id
        self.initLog()

        #rl parameters
        self.sample_size = sample_size
        self.epoch_size = epoch_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.lam = lam
        self.clip_threshold = clip_threshold
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.max_grad_norm = max_grad_norm
        self.v_max = v_max
        self.v_min = v_min
        self.time_limit = time_limit
        self.use_gpu_model = use_gpu_model

        #network settings
        self.s_norm = Normalizer(vec_env.observation_space.shape[0])
        self.actor = Actor(vec_env.observation_space.shape[0], vec_env.action_space.shape[0], vec_env.action_space, hidden=[128, 64])
        self.critic = Critic(vec_env.observation_space.shape[0], self.v_min/(1-self.gamma), self.v_max/(1-self.gamma), hidden =[128, 64])
        
        #env setting
        self.vec_env = vec_env
        self.runner = GAERunner(self.vec_env, self.s_norm, self.actor, self.critic, self.sample_size, self.gamma, self.lam, self.v_max, self.v_min,
            use_gpu_model = self.use_gpu_model)


        # optimizer
        self.actor_optim = optim.Adam(self.actor.parameters(), self.actor_lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), self.critic_lr)

        if use_gpu_model:
            self.T = lambda x: torch.cuda.FloatTensor(x)
        else:
            self.T = lambda x: torch.FloatTensor(x)

        #idx of iter
        self.it = 0


    def _try_to_train(self):
        '''
        train one iter
        '''
        self.vec_env.set_task_t(self.time_limit)
        #collect data
        data = self.runner.run()
        dataset = PPO_Dataset(data)

        atarg = dataset.advantage
        atarg = (atarg - atarg.mean()) / (atarg.std() + 1e-5) # trick: standardized advantage function

        adv_clip_rate = np.mean(np.abs(atarg) > 4)
        adv_max = np.max(atarg)
        adv_min = np.min(atarg)
        val_min = self.v_min;
        val_max = self.v_max / (1-self.gamma);
        vtarg = dataset.vtarget
        vtarg_clip_rate = np.mean(np.logical_or(vtarg < val_min, vtarg > val_max))
        vtd_max = np.max(vtarg)
        vtd_min = np.min(vtarg)

        atarg = np.clip(atarg, -4, 4)
        vtarg = np.clip(vtarg, val_min, val_max)

        dataset.advantage = atarg
        dataset.vtarget = vtarg

        # logging interested variables
        N = np.clip(data["news"].sum(), a_min=1, a_max=None) # prevent divding 0
        avg_rwd = data["rwds"].sum()/N
        avg_step = data["samples"]/N
        rwd_per_step = avg_rwd / avg_step
    
        fail_rate = sum(data["fails"])/N
        self.total_sample += data["samples"]
        self.writer.add_scalar("sample/avg_rwd",     avg_rwd,        self.it)
        self.writer.add_scalar("sample/avg_step",    avg_step,       self.it)
        self.writer.add_scalar("sample/rwd_per_step",rwd_per_step,   self.it)
        self.writer.add_scalar("anneal/total_samples", self.total_sample, self.it)
     
        # special handling of bound channels
  

        if (self.it % self.test_batch == 0):
            test_step, test_rwd = self.runner.test()
            self.writer.add_scalar("sample/test_step", test_step,      self.it)
            self.writer.add_scalar("sample/test_rwd",  test_rwd,       self.it)

            print("\n===== iter %d ====="% self.it)
            #print("avg_rwd       = %f" % avg_rwd)
            #print("avg_step      = %f" % avg_step)
            #print("rwd_per_step  = %f" % rwd_per_step)
            print("test_rwd      = %f" % test_rwd)
            print("test_step     = %f" % test_step)
            #print("fail_rate     = %f" % fail_rate)
            #print("total_samples = %d" % total_sample)
            #print("adv_clip_rate = %f, (%f, %f)" % (adv_clip_rate, adv_min, adv_max))
            #print("vtd_clip_rate = %f, (%f, %f)" % (vtarg_clip_rate, vtd_min, vtd_max))

            #writer.add_scalar("debug/adv_clip",     adv_clip_rate,      it)
            #writer.add_scalar("debug/vtarg_clip",   vtarg_clip_rate,    it)

   

        # start training
        pol_loss_avg    = 0
        pol_surr_avg    = 0
        pol_abound_avg  = 0
        vf_loss_avg     = 0
        clip_rate_avg   = 0

        actor_grad_avg  = 0
        critic_grad_avg = 0

        for epoch in range(self.epoch_size):
        #print("iter %d, epoch %d" % (it, epoch))
            for bit, batch in enumerate(dataset.batch_sample(self.batch_size)):
                # prepare batch data
                ob, ac, atarg, tdlamret, log_p_old = batch
                ob = self.T(ob)
                ac = self.T(ac)
                atarg = self.T(atarg)
                tdlamret = self.T(tdlamret).view(-1, 1)
                log_p_old = self.T(log_p_old)

                # clean optimizer cache
                self.actor_optim.zero_grad()
                self.critic_optim.zero_grad()

                # calculate new log_pact
                ob_normed = self.s_norm(ob)
                m = self.actor.act_distribution(ob_normed)
                vpred = self.critic(ob_normed)
                log_pact = m.log_prob(ac)
                if log_pact.dim() == 2:
                    log_pact = log_pact.sum(dim=1)

                # PPO object, clip advantage object
                ratio = torch.exp(log_pact - log_p_old)
                surr1 = ratio * atarg
                surr2 = torch.clamp(ratio, 1.0 - self.clip_threshold, 1.0 + self.clip_threshold) * atarg
                pol_surr = -torch.mean(torch.min(surr1, surr2))
                pol_loss = pol_surr 
                pol_loss_avg += pol_loss.item()


                # critic vpred loss
                vf_criteria = nn.MSELoss()
                vf_loss = vf_criteria(vpred, tdlamret) / (self.critic.v_std**2) # trick: normalize v loss
               

                vf_loss_avg += vf_loss.item()

                if (not np.isfinite(pol_loss.item())):
                    print("pol_loss infinite")
                    assert(False)
                    from IPython import embed; embed()

                if (not np.isfinite(vf_loss.item())):
                    print("vf_loss infinite")
                    assert(False)
                    from IPython import embed; embed()

                pol_loss.backward()
                vf_loss.backward()

                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)

                self.actor_optim.step()
                self.critic_optim.step()


        batch_num = (self.sample_size // self.batch_size)
        pol_loss_avg    /= batch_num
        vf_loss_avg     /= batch_num
      
        self.writer.add_scalar("train/pol_loss",  pol_loss_avg,      self.it)
        self.writer.add_scalar("train/vf_loss",   vf_loss_avg,       self.it)

        #print("pol_loss      = %f" % pol_loss_avg)
        #print("vf_loss       = %f" % vf_loss_avg)

        # save checkpoint
        if (self.it % self.checkpoint_batch == 0):
            print("save check point ...")
            self.actor.cpu()
            self.critic.cpu()
            self.s_norm.cpu()
            data = {"actor": self.actor.state_dict(),
                    "critic": self.critic.state_dict(),
                    "s_norm": self.s_norm.state_dict()}
            if self.use_gpu_model:
                self.actor.cuda()
                self.critic.cuda()
                self.s_norm.cuda()

            torch.save(data, "%s/%s/checkpoint_%d.tar" % (self.save_dir, self.exp_id, self.it))

        self.it +=1


    
    def setExpid(self,idx):
        '''
        set id of the experiment(idx of batch in BayesianOpt)
        '''
        self.exp_id = idx

    def initLog(self):
        '''
        initialize tensorboard log settings
        '''
        self.writer = SummaryWriter("%s/%s" % (self.save_dir, self.exp_id))
        self.total_sample = 0
        self.train_sample = 0

    def killLog(self):
        self.writer.close()
        self.total_sample = 0
        self.train_sample = 0
        self.it = 0

    def learn(self, num_iter):
        '''
        train policy for num_iter iterations
        '''
        for i in range(num_iter):
            self._try_to_train()


    def evaluate(self):
        '''
        evaluate trained policy
        '''
        test_step, test_rwd = self.runner.test()

        return test_rwd




# def PPO_vec(actor, critic, s_norm, vec_env, exp_id,
#         save_dir="./experiments",
#         iter_num=1000,
#         sample_size=4096,
#         epoch_size=10,
#         batch_size=256,
#         checkpoint_batch=10,
#         test_batch=10,
#         gamma=0.99,
#         lam=0.95,
#         clip_threshold=0.2,
#         actor_lr=3e-4,
#         critic_lr=3e-4,
#         max_grad_norm=0.5,
#         v_max = 1,
#         v_min = 0,
#         time_limit = 1000,
#         use_gpu_model=False):
#     """
#     PPO algorithm of reinforcement learning
#     Inputs:
#         actor       policy network, with following methods:
#                     m = actor.distributions(ob)
#                     ac = actor.act_deterministic()

#         critic

#         s_norm

#         vec_env     vectorized environment, with following methods:
#                     ob = env.reset()
#                     ob, rew, done, _ = env.step(ac)
#         exp_id      string, experiment id, checkpoints amd training monitor data
#                     will be saved to ./${exp_id}/
#     """


#     #actor_optim = optim.Adam(actor.parameters(), actor_lr, momentum=actor_momentum, weight_decay=actor_wdecay)
#     #critic_optim = optim.Adam(critic.parameters(), critic_lr, momentum=critic_momentum, weight_decay=critic_wdecay)
#     actor_optim = optim.Adam(actor.parameters(), actor_lr)
#     critic_optim = optim.Adam(critic.parameters(), critic_lr)

#     # set up environment and data generator
#     runner = GAERunner(vec_env, s_norm, actor, critic, sample_size, gamma, lam, v_max, v_min,
#             use_gpu_model=use_gpu_model)

#     if use_gpu_model:
#         T = lambda x: torch.cuda.FloatTensor(x)
#     else:
#         T = lambda x: torch.FloatTensor(x)

#     writer = SummaryWriter("%s/%s" % (save_dir, exp_id))
#     total_sample = 0
#     train_sample = 0
   
#     for it in range(iter_num + 1):
#         # sample data with gae estimated adv and vtarg
#         # currculum learning settings
#         vec_env.set_task_t(time_limit)
#         #collect data
#         # t0 = time.time()
#         data = runner.run()
#         # t1 = time.time()
#         # print("sampling time :{}".format(t1 -t0))
#         dataset = PPO_Dataset(data)

#         atarg = dataset.advantage
#         atarg = (atarg - atarg.mean()) / (atarg.std() + 1e-5) # trick: standardized advantage function

#         adv_clip_rate = np.mean(np.abs(atarg) > 4)
#         adv_max = np.max(atarg)
#         adv_min = np.min(atarg)
#         val_min = v_min;
#         val_max = v_max / (1-gamma);
#         vtarg = dataset.vtarget
#         vtarg_clip_rate = np.mean(np.logical_or(vtarg < val_min, vtarg > val_max))
#         vtd_max = np.max(vtarg)
#         vtd_min = np.min(vtarg)

#         atarg = np.clip(atarg, -4, 4)
#         vtarg = np.clip(vtarg, val_min, val_max)

#         dataset.advantage = atarg
#         dataset.vtarget = vtarg

#         # logging interested variables
#         N = np.clip(data["news"].sum(), a_min=1, a_max=None) # prevent divding 0
#         avg_rwd = data["rwds"].sum()/N
#         avg_step = data["samples"]/N
#         rwd_per_step = avg_rwd / avg_step
    
#         fail_rate = sum(data["fails"])/N
#         total_sample += data["samples"]
#         writer.add_scalar("sample/avg_rwd",     avg_rwd,        it)
#         writer.add_scalar("sample/avg_step",    avg_step,       it)
#         writer.add_scalar("sample/rwd_per_step",rwd_per_step,   it)
#         writer.add_scalar("anneal/total_samples", total_sample, it)
     
#         # special handling of bound channels
  

#         if (it % test_batch == 0):
#             test_step, test_rwd = runner.test()
#             writer.add_scalar("sample/test_step", test_step,      it)
#             writer.add_scalar("sample/test_rwd",  test_rwd,       it)

#             print("\n===== iter %d ====="% it)
#             #print("avg_rwd       = %f" % avg_rwd)
#             #print("avg_step      = %f" % avg_step)
#             #print("rwd_per_step  = %f" % rwd_per_step)
#             print("test_rwd      = %f" % test_rwd)
#             print("test_step     = %f" % test_step)
#             #print("fail_rate     = %f" % fail_rate)
#             #print("total_samples = %d" % total_sample)
#             #print("adv_clip_rate = %f, (%f, %f)" % (adv_clip_rate, adv_min, adv_max))
#             #print("vtd_clip_rate = %f, (%f, %f)" % (vtarg_clip_rate, vtd_min, vtd_max))

#             writer.add_scalar("debug/adv_clip",     adv_clip_rate,      it)
#             writer.add_scalar("debug/vtarg_clip",   vtarg_clip_rate,    it)

   

#         # start training
#         pol_loss_avg    = 0
#         pol_surr_avg    = 0
#         pol_abound_avg  = 0
#         vf_loss_avg     = 0
#         clip_rate_avg   = 0

#         actor_grad_avg  = 0
#         critic_grad_avg = 0
#         # t0 = time.time()
#         for epoch in range(epoch_size):
#         #print("iter %d, epoch %d" % (it, epoch))
#             for bit, batch in enumerate(dataset.batch_sample(batch_size)):
#                 # prepare batch data
#                 ob, ac, atarg, tdlamret, log_p_old = batch
#                 ob = T(ob)
#                 ac = T(ac)
#                 atarg = T(atarg)
#                 tdlamret = T(tdlamret).view(-1, 1)
#                 log_p_old = T(log_p_old)

#                 # clean optimizer cache
#                 actor_optim.zero_grad()
#                 critic_optim.zero_grad()

#                 # calculate new log_pact
#                 ob_normed = s_norm(ob)
#                 m = actor.act_distribution(ob_normed)
#                 vpred = critic(ob_normed)
#                 log_pact = m.log_prob(ac)
#                 if log_pact.dim() == 2:
#                     log_pact = log_pact.sum(dim=1)

#                 # PPO object, clip advantage object
#                 ratio = torch.exp(log_pact - log_p_old)
#                 surr1 = ratio * atarg
#                 surr2 = torch.clamp(ratio, 1.0 - clip_threshold, 1.0 + clip_threshold) * atarg
#                 pol_surr = -torch.mean(torch.min(surr1, surr2))
#                 pol_loss = pol_surr 
#                 pol_loss_avg += pol_loss.item()


#                 # critic vpred loss
#                 vf_criteria = nn.MSELoss()
#                 vf_loss = vf_criteria(vpred, tdlamret) / (critic.v_std**2) # trick: normalize v loss
#                 #vf_loss = 0.5*((vpred - tdlamret).pow(2)).mean()

#                 vf_loss_avg += vf_loss.item()

#                 if (not np.isfinite(pol_loss.item())):
#                     print("pol_loss infinite")
#                     assert(False)
#                     from IPython import embed; embed()

#                 if (not np.isfinite(vf_loss.item())):
#                     print("vf_loss infinite")
#                     assert(False)
#                     from IPython import embed; embed()

#                 pol_loss.backward()
#                 vf_loss.backward()

#                 nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
#                 nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)

#                 actor_optim.step()
#                 critic_optim.step()
#         # t1 = time.time()
#         # print("training time :{}".format(t1 -t0))

#         batch_num = (sample_size // batch_size)
#         pol_loss_avg    /= batch_num
#         vf_loss_avg     /= batch_num
      
#         writer.add_scalar("train/pol_loss",  pol_loss_avg,      it)
#         writer.add_scalar("train/vf_loss",   vf_loss_avg,       it)

#         #print("pol_loss      = %f" % pol_loss_avg)
#         #print("vf_loss       = %f" % vf_loss_avg)

#         # save checkpoint
#         if (it % checkpoint_batch == 0):
#             print("save check point ...")
#             actor.cpu()
#             critic.cpu()
#             s_norm.cpu()
#             data = {"actor": actor.state_dict(),
#                     "critic": critic.state_dict(),
#                     "s_norm": s_norm.state_dict()}
#             if use_gpu_model:
#                 actor.cuda()
#                 critic.cuda()
#                 s_norm.cuda()

#             torch.save(data, "%s/%s/checkpoint_%d.tar" % (save_dir, exp_id, it))