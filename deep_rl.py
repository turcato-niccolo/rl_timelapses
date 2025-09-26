import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
from torch.nn.init import calculate_gain

class DeterministicActor(nn.Module):
    def __init__(self, state_dim, action_dim, device, max_action, depth=2, num_neurons=512):
        super(DeterministicActor, self).__init__()

        self.device = device
        self.layers = []

        self.layers.append(nn.Linear(state_dim, num_neurons, device=device, dtype=torch.float32))
        for _ in range(depth-1):
            self.layers.append(nn.Linear(num_neurons, num_neurons, device=device, dtype=torch.float32))
        self.layers.append(nn.Linear(num_neurons, action_dim, device=device, dtype=torch.float32))

        self.layers = nn.ParameterList(self.layers)
        self.depth = depth
        self.num_neurons = num_neurons
        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.layers[0](state))
        for k in range(self.depth-1):
            a = F.relu(self.layers[k+1](a))

        return self.max_action * torch.tanh(self.layers[-1](a))

    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return torch.clip(action, -self.max_action, self.max_action), torch.tensor(0.), mean

    def save_to_csv_files(self, path):
        np.savetxt(path+'l0_w.csv', self.layers[0].weight.detach().cpu().numpy(), delimiter=',')
        np.savetxt(path+'l0_b.csv', self.layers[0].bias.detach().cpu().numpy(), delimiter=',')
        for k in range(self.depth - 1):
            np.savetxt(path + f'l{k+1}_w.csv', self.layers[k+1].weight.detach().cpu().numpy(), delimiter=',')
            np.savetxt(path + f'l{k+1}_b.csv', self.layers[k+1].bias.detach().cpu().numpy(), delimiter=',')
        np.savetxt(path + f'l{self.depth}_w.csv', self.layers[self.depth].weight.detach().cpu().numpy(), delimiter=',')
        np.savetxt(path + f'l{self.depth}_b.csv', self.layers[self.depth].bias.detach().cpu().numpy(), delimiter=',')


class Actor_with_uncertainty(nn.Module):
    def __init__(self, state_dim, action_dim, device, max_action, depth=2, num_neurons=512):
        super(Actor_with_uncertainty, self).__init__()
        self.device = device
        self.layers = []

        self.layers.append(nn.Linear(state_dim+1, num_neurons, device=device, dtype=torch.float32))
        for _ in range(depth - 1):
            self.layers.append(nn.Linear(num_neurons, num_neurons, device=device, dtype=torch.float32))
        self.layers.append(nn.Linear(num_neurons, action_dim, device=device, dtype=torch.float32))

        self.layers = nn.ParameterList(self.layers)
        self.depth = depth
        self.num_neurons = num_neurons
        self.max_action = max_action

    def forward(self, state, uncertain):
        su = torch.cat([state, uncertain], 1)  # .unsqueeze(0).expand(state.size(dim=0), -1, -1)
        a = F.relu(self.layers[0](su))
        for k in range(self.depth-1):
            a = F.relu(self.layers[k+1](a))
        return self.max_action * torch.tanh(self.layers[-1](a))

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, device, depth=2, num_neurons=512):
        super(Critic, self).__init__()

        # Q architecture
        self.layers = []

        self.layers.append(nn.Linear(state_dim+action_dim, num_neurons, device=device))
        for _ in range(depth - 1):
            self.layers.append(nn.Linear(num_neurons, num_neurons, device=device))
        self.layers.append(nn.Linear(num_neurons, 1, device=device))

        self.layers = nn.ParameterList(self.layers)
        self.depth = depth
        self.num_neurons = num_neurons


    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        sa = F.relu(self.layers[0](sa))
        for k in range(self.depth-1):
            sa = F.relu(self.layers[k+1](sa))

        return self.layers[-1](sa)

class Critic_ensemble(nn.Module):

    def __init__(self, state_dim, action_dim, device, depth=2, num_neurons=512, ensemble_size=10):
        super(Critic_ensemble, self).__init__()
        self.depth = depth
        self.num_neurons = num_neurons

        # Q architecture
        self.layers = []

        self.layers.append(EnsembleFC(state_dim + action_dim + 1, num_neurons, ensemble_size=ensemble_size, device=device))
        for _ in range(depth - 1):
            self.layers.append(EnsembleFC(num_neurons, num_neurons, ensemble_size=ensemble_size, device=device))
        self.layers.append(EnsembleFC(num_neurons, 1, ensemble_size=ensemble_size, device=device))

        self.layers = nn.ParameterList(self.layers)

        self.ensemble_size = ensemble_size
        self.device = device

    def forward(self, state, action, uncertain):
        q = self.get_Q_estimates(state, action, uncertain)

        mean = q.mean(0)
        std = q.std(0)

        return mean, std

    def get_Q_estimates(self, state, action, uncertain):
        # print('state.size():', state.size())
        # print('action.size():', action.size())
        # print('uncertain.size():', uncertain.size())

        sau = torch.cat([state, action, uncertain], -1).unsqueeze(0).expand(self.ensemble_size, -1, -1)
        q = F.relu(self.layers[0](sau))
        for k in range(self.depth-1):
            q = F.relu(self.layers[k+1](q))

        return q

    def get_Q_estimate(self, state, action, uncertain, k):
        Qs = self.get_Q_estimates(state, action, uncertain)
        return Qs[k, :, :]

class EnsembleFC(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    ensemble_size: int
    weight: torch.Tensor

    def __init__(self, in_features: int, out_features: int, ensemble_size: int, device: torch.device, bias: bool = True) -> None:
        super(EnsembleFC, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size
        self.device = device
        self.weight = nn.Parameter(torch.Tensor(ensemble_size, in_features, out_features).to(self.device))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(ensemble_size, out_features).to(self.device))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        w_times_x = torch.bmm(input, self.weight)
        return torch.add(w_times_x, self.bias[:, None, :])  # w times x + b

def kaiming_uniform_(tensor, a=0., mode='fan_in', nonlinearity='leaky_relu'):
    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)


def _calculate_correct_fan(tensor, mode):
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out']
    if mode not in valid_modes:
        raise ValueError("Mode {} not supported, please use one of {}".format(mode, valid_modes))

    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    return fan_in if mode == 'fan_in' else fan_out


def _calculate_fan_in_and_fan_out(tensor):
    dimensions = tensor.dim()
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")
    num_input_fmaps = tensor.size(-2)
    num_output_fmaps = tensor.size(-1)
    receptive_field_size = 1
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


import torch
import torch.nn.functional as F
from torch.optim import Adam
import torch.nn as nn
from torch.distributions import Normal

# Implementation of SAC taken from: https://github.com/pranz24/pytorch-soft-actor-critic/
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6


# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class TwinCritic(nn.Module):
    def __init__(self, state_dim, action_dim, device, depth=2, num_neurons=512):
        super(TwinCritic, self).__init__()
        # Q1 architecture
        self.q1 = Critic(state_dim=state_dim, action_dim=action_dim, device=device, depth=depth, num_neurons=num_neurons)

        # Q2 architecture
        self.q2 = Critic(state_dim=state_dim, action_dim=action_dim, device=device, depth=depth, num_neurons=num_neurons)

    def forward(self, state, action):
        return self.q1(state, action), self.q2(state, action)

    def Q1(self, state, action):
        return self.q1(state, action)


class SAC(object):
    def __init__(self, state_dim, action_dim, max_action,
                 discount=0.99,
                 tau=0.005,
                 alpha=0.2,
                 policy="Gaussian",
                 automatic_entropy_tuning=True,
                 num_neurons=256,
                 policy_freq=2,
                 depth=2,
                 lr=3e-4,
                 device=None,
                 *args,
                 **kargs):

        self.discount = discount
        self.tau = tau
        self.alpha = alpha

        self.policy_type = policy
        self.automatic_entropy_tuning = automatic_entropy_tuning

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.critic = TwinCritic(state_dim, action_dim, device=device, depth=depth, num_neurons=num_neurons)
        self.critic_optim = Adam(self.critic.parameters(), lr=lr)

        self.critic_target = TwinCritic(state_dim, action_dim, device=device, depth=depth, num_neurons=num_neurons)
        hard_update(self.critic_target, self.critic)
        self.updates = 0

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor([action_dim]).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=lr)

            self.policy = GaussianPolicy(state_dim, action_dim, num_neurons, max_action).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=lr)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicActor(state_dim, action_dim, device=device, depth=depth, num_neurons=num_neurons)
            self.policy_optim = Adam(self.policy.parameters(), lr=lr)

        self.max_action = max_action
        self.total_it = 0
        self.policy_freq = policy_freq

    def reset(self):
        self.policy_optim = torch.optim.Adam(self.policy.parameters(),
                                             lr=self.policy_optim.param_groups[0]['lr'])

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate:
            _, _, action = self.policy.sample(state)
        else:
            action, _, _ = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def train(self, memory, batch_size):
        self.updates += 1
        # Sample a batch from memory
        state_batch, action_batch, next_state_batch, reward_batch, mask_batch = memory.sample(batch_size=batch_size)

        """state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device)"""

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.discount * (min_qf_next_target)
        qf1, qf2 = self.critic(state_batch,
                               action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            pi, log_pi, _ = self.policy.sample(state_batch)

            qf1_pi, qf2_pi = self.critic(state_batch, pi)
            policy_loss = ((
                                       self.alpha * log_pi) - qf1_pi).mean()  # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

            self.policy_optim.zero_grad()
            policy_loss.backward()
            self.policy_optim.step()

            if self.automatic_entropy_tuning:
                alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

                self.alpha_optim.zero_grad()
                alpha_loss.backward()
                self.alpha_optim.step()

                self.alpha = self.log_alpha.exp()
                alpha_tlogs = self.alpha.clone()  # For TensorboardX logs
            else:
                alpha_loss = torch.tensor(0.).to(self.device)
                alpha_tlogs = torch.tensor(self.alpha)  # For TensorboardX logs

            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

    # Save model parameters
    def save(self, filename):
        print('Saving models to {}'.format(filename))
        torch.save({'policy_state_dict': self.policy.state_dict(),
                    'critic_state_dict': self.critic.state_dict(),
                    'critic_target_state_dict': self.critic_target.state_dict(),
                    'critic_optimizer_state_dict': self.critic_optim.state_dict(),
                    'policy_optimizer_state_dict': self.policy_optim.state_dict()}, filename)

    # Load model parameters
    def load(self, filename, evaluate=False, load_critic=True):
        print('Loading models from {}'.format(filename))

        checkpoint = torch.load(filename, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.policy_optim.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        if load_critic:
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.critic_optim.load_state_dict(checkpoint['critic_optimizer_state_dict'])

        if evaluate:
            self.policy.eval()
            self.critic.eval()
            self.critic_target.eval()
        else:
            self.policy.train()
            self.critic.train()
            self.critic_target.train()


class GaussianPolicy(DeterministicActor):
    def __init__(self, state_dim, action_dim, device, max_action, depth=2, num_neurons=512):
        super(GaussianPolicy, self).__init__(state_dim=state_dim, action_dim=action_dim, device=device,
                                             max_action=max_action, depth=depth, num_neurons=num_neurons)

        self.mean_linear = nn.Linear(num_neurons, action_dim, device=device)
        self.log_std_linear = nn.Linear(num_neurons, action_dim, device=device)
        self.apply(weights_init_)

        # action rescaling
        self.action_scale = max_action
        self.action_bias = 0.

    def forward(self, state):
        x = F.relu(self.layers[0](state))
        for k in range(self.depth - 1):
            x = F.relu(self.layers[k + 1](x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


class DeterministicPolicy(DeterministicActor):
    def __init__(self, state_dim, action_dim, device, max_action, depth=2, num_neurons=512):
        super(DeterministicPolicy, self).__init__(state_dim=state_dim, action_dim=action_dim, device=device,
                                                  max_action=max_action, depth=depth, num_neurons=num_neurons)
        self.mean = nn.Linear(num_neurons, action_dim, device=device)
        self.noise = torch.Tensor(action_dim, device=device)
        self.apply(weights_init_)
        # action rescaling
        self.action_scale = torch.tensor(max_action, device=device)
        self.action_bias = torch.tensor(0., device=device)

    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6), device=None):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )

