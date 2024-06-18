#from typing import Tuple
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import time
import gymnasium as gym
from tqdm import trange
import numpy as np

import torch
from torch import nn
from torch.optim import Adam

DTYPE = torch.float32
#RENDER_MODE = 'human'

### TEST ENVS
# import press_the_light
# ENVIRONMENT_NAME = "PTLG-v0"
# RENDER_MODE = None
#ENVIRONMENT_NAME = "MountainCar-v0"
ENVIRONMENT_NAME = "LunarLander-v2"
#ENVIRONMENT_NAME = 'CartPole-v1'
#~~ TEST ENVS

# HYPERPARAMETER LAND
plot = 0
PLOT_INTERVAL=20
BS = 512
MAX_REPLAY_BUFFER = 4000
HIDDEN_STATE_SZ = 32
EPISODES = 500
LEARNING_RATE= 3e-3
VALUE_SCALE = 1
ENTROPY_SCALE = 0.001 #0.001 #
DISCOUNT_FACTOR = 0.99
TRAIN_STEPS=5
PPO_EPSILON=0.2
APPROX_KL_THRESHOLD_BREAK=0.2


class Agent(nn.Module):
    def __init__(self, in_features:int, out_features:int, hidden_state:int=HIDDEN_STATE_SZ,final_activation:bool=False):
        super(Agent, self).__init__()
        self.l1 = nn.Linear(in_features,hidden_state)
        self.l2 = nn.Linear(hidden_state,hidden_state)
        self.l3 = nn.Linear(hidden_state,out_features)

        self.c1 = nn.Linear(in_features,hidden_state)
        self.c2 = nn.Linear(hidden_state,hidden_state)
        self.c3 = nn.Linear(hidden_state,1)

        self.activation = nn.GELU()
        self.final_activation = final_activation

    def forward(self,obs:torch.Tensor):

        # Actor
        x = self.activation(self.l1(obs))
        x = self.activation(self.l2(x))
        #x = x[:,-1,:]
        x = self.l3(x)
        
        if self.final_activation:
            return x.log_softmax(dim=-1)#
        return x


def evaluate(actor_model:nn.Module,test_env:gym.Env):
    (obs,_), terminated, truncated = test_env.reset(), False, False
    total_rew = 0.0
    while not terminated and not truncated:
        act = actor_model(torch.tensor(obs,dtype=DTYPE)).exp().argmax().item()

        obs, rew, terminated, truncated, _ = test_env.step(act)
        total_rew += float(rew)
    return total_rew


if __name__ == '__main__':
    env = gym.make(ENVIRONMENT_NAME)
    obs_shape = env.unwrapped.observation_space.shape[0]
    act_shape = int(env.action_space.n)

    actor  = Agent(obs_shape,act_shape, HIDDEN_STATE_SZ,final_activation=True)
    critic = Agent(obs_shape,1, HIDDEN_STATE_SZ)

    actor_opt = Adam(actor.parameters(), lr=LEARNING_RATE)
    critic_opt = Adam(critic.parameters(), lr=LEARNING_RATE)

    def train_step(x,selected_action,reward,old_log_dist):
        break_signal = 0

        actor.train();critic.train();
        log_dist, value = actor(x),critic(x)
        action_mask = selected_action.reshape(-1,1) == torch.arange(log_dist.shape[1]).reshape(1,-1).expand(selected_action.shape[0],-1).float()
        
        # Advantage
        advantage = reward.reshape(-1,1) - value
        #advantage = (advantage - advantage.mean()) / (advantage.numpy().std() + 1e-10)
        
        # with advantage
        masked_advantage = action_mask * advantage.detach()

        # no advantage
        # masked_advantage = action_mask * reward.reshape(-1,1)

        # PPO
        logratios = log_dist - old_log_dist
        ratios = logratios.exp()
        unclipped_ratio = masked_advantage * ratios
        clipped_ratio   = masked_advantage * ratios.clip(1-PPO_EPSILON,1+PPO_EPSILON)
        action_loss = -unclipped_ratio.minimum(clipped_ratio).sum(-1).mean()
        approx_kl = ((ratios - 1) - logratios).mean()
        
        # VPG
        # action_loss = -(log_dist.exp() * masked_advantage).sum(-1).mean()

        entropy_loss = (log_dist.exp() * log_dist).sum(-1).mean() # encourages diversity/exploring
        critic_loss  = advantage.square().mean()
        actor_opt.zero_grad()
        critic_opt.zero_grad()
        (action_loss + entropy_loss*ENTROPY_SCALE + critic_loss*VALUE_SCALE).backward()
        actor_opt.step()
        critic_opt.step()

        if approx_kl > APPROX_KL_THRESHOLD_BREAK:
            break_signal=1
        return action_loss.item(),entropy_loss.item(),critic_loss.item(),break_signal

    def get_action(obs):#temperature=TEMPERATURE
        log_dist = actor(obs)
        action = log_dist.exp().multinomial(1)#,replacement=True
        return action.detach().item(), log_dist.detach()

    if plot:
        import matplotlib.pyplot as plt
        plt.ioff()
        plt.figure(2)

    st,steps = time.perf_counter(),0
    Xn,An,Rn = [],[],[]
    old_logs = []
    returns,action_losses,critic_losses,entropy_losses=[],[],[],[]
    episode_lengths = []

    for episode_number in (t:=trange(EPISODES)):
        obs = env.reset()[0]
        rews,terminated,truncated = [], False, False

        while not terminated and not truncated:
            act,log_dist = get_action(torch.tensor(obs,dtype=DTYPE))

            Xn.append(np.copy(obs))
            An.append(act)
            old_logs.append(log_dist)

            obs,rew,terminated,truncated,_ = env.step(act)
            rews.append(float(rew))
        steps+=len(rews)

        # For plot/misc
        reward_print = np.sum(rews)
        returns.append(reward_print)#(sum(rews))
        episode_lengths.append(len(rews)/700)

        #reward to go
        discounts = np.power(DISCOUNT_FACTOR, np.arange(len(rews)))
        #rtgs=[np.sum(rews[i:] * discounts[:len(rews)-i]) for i in range(len(rews))]
        #rtgs = rtgs / (np.std(rtgs)+1e-8)
        #rtgs = np.clip(rtgs, -10, 10)
        #Rn += rtgs.tolist()
        Rn += [np.sum(rews[i:] * discounts[:len(rews)-i]) for i in range(len(rews))]
        
        Xn, An, Rn = Xn[-MAX_REPLAY_BUFFER:],An[-MAX_REPLAY_BUFFER:],Rn[-MAX_REPLAY_BUFFER:]
        old_logs = old_logs[-MAX_REPLAY_BUFFER:]
        X,A,R = torch.tensor(np.array(Xn), dtype=DTYPE),torch.tensor(np.array(An), dtype=DTYPE),torch.tensor(np.array(Rn), dtype=DTYPE)

        old_log_dist = torch.tensor(np.array(old_logs), dtype=DTYPE).detach().squeeze()
        for j in range(TRAIN_STEPS):
            samples = torch.randperm(len(X))[:BS]
            action_loss, entropy_loss, critic_loss, break_signal = train_step(X[samples],A[samples],R[samples],old_log_dist[samples])
            if break_signal:
                #print(f'KL break on {j}')
                break
        t.set_description(f"sz:{len(Xn):5d} steps/s: {steps/(time.perf_counter() - st):.2f}  action_loss: {action_loss:.2f} critic_loss: {critic_loss:.2f} entropy_loss: {entropy_loss:.2f} reward: {reward_print:.2f}")

        action_losses.append(min(action_loss,1000)/1000)
        critic_losses.append(min(critic_loss,1000)/1000)
        entropy_losses.append(entropy_loss)

        
        if plot and episode_number>0 and episode_number%PLOT_INTERVAL==0:
            plt.clf()
            plt.subplot(211)
            plt.plot(returns,label='returns')
            plt.plot(episode_lengths,label='episode_lengths')
            plt.legend(loc="upper left")
            plt.subplot(212)
            plt.plot(action_losses,label='actionloss')
            plt.plot(entropy_losses,label='entropyloss')
            plt.plot(critic_losses,label='criticloss')
            plt.legend(loc="upper left")
            plt.pause(0.5)

    test_env = gym.make(ENVIRONMENT_NAME,render_mode='human')
    test_rew = evaluate(actor,test_env)
    print(f'Test reward : {test_rew}')
    pytorch_actor_params = sum(p.numel() for p in actor.parameters() if p.requires_grad)
    pytorch_critic_params = sum(p.numel() for p in critic.parameters() if p.requires_grad)
    print('\ntrainable params',pytorch_actor_params+pytorch_critic_params)
    if plot:
        plt.show()    
    env.close()

    
    


