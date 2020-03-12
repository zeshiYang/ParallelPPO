import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import numpy as np

######
INIT_ACTOR_SCALE = 0.01
NOISE = 0.1
USE_ELU = False

NORM_SAMPLES    = 10000000

######

class Normalizer(nn.Module):
  '''
    the class of state normalization modeule
  '''
  def __init__(self, in_dim, sample_lim=NORM_SAMPLES):
    super(Normalizer, self).__init__()

    self.mean    = nn.Parameter(torch.zeros([in_dim]))
    self.std     = nn.Parameter(torch.ones([in_dim]))
    self.mean_sq = nn.Parameter(torch.ones([in_dim]))
    self.num     = nn.Parameter(torch.zeros([1]))


    self.sum_new    = torch.zeros([in_dim])
    self.sum_sq_new = torch.zeros([in_dim])
    self.num_new    = torch.zeros([1])

    for param in self.parameters():
      param.requires_grad = False

    self.sample_lim = sample_lim

  def forward(self, x):
    return np.clip(((x - self.mean) / self.std), -4,4)

  def unnormalize(self, x):
    return x * self.std + self.mean

  def set_mean_std(self, mean, std):
    self.mean.data = torch.Tensor(mean)
    self.std.data = torch.Tensor(std)

  def record(self, x):
    if (self.num + self.num_new >= self.sample_lim):
      return
    
    if x.dim() == 1:
      self.num_new += 1
      self.sum_new += x
      #print(x)
      self.sum_sq_new += torch.pow(x, 2)
    elif x.dim() == 2:
      self.num_new += x.shape[0]
     
      #print(x)
      self.sum_new += torch.sum(x, dim=0)
      self.sum_sq_new += torch.sum(torch.pow(x, 2), dim=0)
    else:
      assert(False and "normalizer record more than 2 dim")

  def update(self):
    if self.num >= self.sample_lim or self.num_new == 0:
      return

    # update mean, mean_sq and std
    total_num = self.num + self.num_new;
    self.mean.data *= (self.num / total_num)
    self.mean.data += self.sum_new / total_num
    self.mean_sq.data *= (self.num / total_num)
    self.mean_sq.data += self.sum_sq_new / total_num
    self.std.data = torch.sqrt(torch.abs(self.mean_sq.data - torch.pow(self.mean.data, 2)))
    self.std.data += 0.01 # in case of divide by 0
    self.num.data += self.num_new

    # clear buffer
    self.sum_new.data.zero_()
    self.sum_sq_new.data.zero_()
    self.num_new.data.zero_()

    return

# initialize fc layer using xavier uniform initialization
def xavier_init(module):
  nn.init.xavier_uniform_(module.weight.data, gain=1)
  nn.init.constant_(module.bias.data, 0)
  return module

def orthogonal_init(module, gain=1):
  nn.init.orthogonal_(module.weight.data, gain)
  nn.init.constant_(module.bias.data, 0)
  return module

class Actor(nn.Module):
  '''
  the class of actor neural network
  '''

  def __init__(self, s_dim, a_dim, a_bound, hidden=[1024, 512]):
    '''
    s_dim: state dimentions
    a_dim: action dimentions
    a_bound: action bounds
    hidden : dimentions of differnet layers
    '''
    super(Actor, self).__init__()
    self.fc = []
    input_dim = s_dim
    for h_dim in hidden:
      self.fc.append(orthogonal_init(nn.Linear(input_dim, h_dim)))
      input_dim = h_dim

    self.fc = nn.ModuleList(self.fc)

    # initialize final layer weight to be [INIT, INIT]
    self.fca = orthogonal_init(nn.Linear(input_dim, a_dim), 0.01)

    # set a_norm not trainable
    self.a_min = torch.Tensor(a_bound.low)
    self.a_max = torch.Tensor(a_bound.high)
    self.a_mean = nn.Parameter((self.a_max + self.a_min) / 2)
    self.a_std  = nn.Parameter(torch.Tensor([np.log(0.1)]*a_bound.shape[0]))

    self.a_min.requires_grad = False
    self.a_max.requires_grad = False
    self.a_mean.requires_grad = False
    self.a_std.requires_grad = True

    self.activation = F.elu if USE_ELU else F.tanh

  def forward(self, x):
    # normalize x first
    layer = x
    for fc_op in self.fc:
      layer = self.activation(fc_op(layer))

    # unnormalize action
    layer_a = self.fca(layer)
    a_mean = layer_a
    return a_mean

  def act_distribution(self, x):
    '''
    compute action distributions
    '''
    a_mean = self.forward(x)
    m = D.Normal(a_mean, torch.exp(self.a_std))
    return m

  def act_deterministic(self, x):
    '''
    compute determnistic actions
    '''
    return self.forward(x)

  def act_stochastic(self, x):
    m = self.act_distribution(x)
    ac = m.sample()
    return ac

class Critic(nn.Module):
      
  '''
  the class of critic network
  '''

  def __init__(self, s_dim, val_min, val_max, hidden=[1024, 512]):
    super(Critic, self).__init__()
    self.fc = []
    input_dim = s_dim
    for h_dim in hidden:
      self.fc.append(orthogonal_init(nn.Linear(input_dim, h_dim)))
      input_dim = h_dim

    self.fc = nn.ModuleList(self.fc)
    self.fcv = orthogonal_init(nn.Linear(input_dim, 1))

    # value normalizer
    self.v_min = torch.Tensor([val_min])
    self.v_max = torch.Tensor([val_max])
    self.v_mean = nn.Parameter((self.v_max + self.v_min) / 2)
    self.v_std  = nn.Parameter((self.v_max - self.v_min) / 2)
    self.v_min.requires_grad = False
    self.v_max.requires_grad = False
    self.v_mean.requires_grad = False
    self.v_std.requires_grad = False

    self.activation = F.elu if USE_ELU else F.tanh

  def forward(self, x):
    '''
    compute state values
    '''
    layer = x
    for fc_op in self.fc:
      layer = self.activation(fc_op(layer))

    # unnormalize value
    value = self.fcv(layer)
    value = self.v_std * value + self.v_mean

    return value

def load_model(ckpt):
  '''
  load the saved model

  ckpt : path of loaded model
  return :
        s_norm: state normalization module
        actor: actor neural network
        critic : critic neural network
  '''
  data = torch.load(ckpt)

  s_dim = data["s_norm"]["mean"].shape[0]
  a_dim = data["actor"]["fca.bias"].shape[0]
  a_min = data["actor"]["a_mean"].numpy()
  a_max = data["actor"]["a_mean"].numpy()
  import gym.spaces
  a_bound = gym.spaces.Box(a_min, a_max)

  a_hidden = list(map(lambda i: data["actor"]["fc.%d.bias" % i].shape[0], [0, 1]))
  c_hidden = list(map(lambda i: data["critic"]["fc.%d.bias" % i].shape[0], [0, 1]))

  s_norm = Normalizer(s_dim, sample_lim=-1)
  actor = Actor(s_dim, a_dim, a_bound, a_hidden)
  critic = Critic(s_dim, 0, 1, c_hidden)

  s_norm.load_state_dict(data["s_norm"])
  actor.load_state_dict(data["actor"])
  critic.load_state_dict(data["critic"])

  return s_norm, actor, critic
