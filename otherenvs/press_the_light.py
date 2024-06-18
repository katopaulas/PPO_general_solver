import random
import gymnasium as gym
import numpy as np

from stable_baselines3 import A2C
from gymnasium.envs.registration import register

class PTLG(gym.Env):
    """
    """
    metadata = {'render_modes':[]}
    def __init__(self,render_mode=None, size=16, game_length=10,hard_mode=False):
        self.size = size
        self.g_length = game_length
        self.done = True
        self.hard_mode = hard_mode
        self.current_step = 0

        self.observation_space = gym.spaces.Box(0,1,shape=(self.size,),dtype=np.float32)
        self.action_space      = gym.spaces.Discrete(self.size)


    def _get_obs(self):
        obs = [0] * self.size
        if self.current_step < len(self.state):
            obs[self.state[self.current_step]] = 1
        return np.array(obs,dtype=np.float32)

    def reset(self,seed=None,options=None):
        super().reset(seed=seed)
        self.state = np.random.randint(0, self.size, size = self.g_length)
        self.done = False
        self.current_step = 0
        return self._get_obs(),{}

    def step(self, action):
        #print(self.current_step,self.state, action)
        target = ((action + self.current_step) % self.size) if self.hard_mode else action 
        reward = int(target == self.state[self.current_step])
        self.current_step+=1
        if not reward:
            self.done = True

        if self.current_step>=self.g_length:
            self.done=True

        return self._get_obs(), reward, self.done, self.current_step>=self.g_length, {}
    
register(
id="PTLG-v0",
entry_point=PTLG,
max_episode_steps=None,
)

# if __name__ == '__main__':

#     env = PTLG(size=10, game_length=10)
#     #env = gym.make("LunarLander-v2", render_mode="rgb_array")
   

#     model = A2C('MlpPolicy',env,verbose=1)
#     model.learn(total_timesteps=1000)


#     ######################################
#     MAX_STEPS = 100
#     rewards = 0
#     v_env = model.get_env()
#     for step in range(MAX_STEPS):
#         print(f'Playing game {step}')

#         done = False
#         obs = v_env.reset()
#         while not done:
#             action, _ = model.predict(obs.squeeze(), deterministic = True)
#             #print(obs.squeeze(),action)
            
            
#             obs, reward, done, truncated, info = env.step(action)
#             rewards+=reward
            

#             s_out = f'Step:    {step}/{MAX_STEPS}, reward:    {reward},  acc_rewards:    {rewards}'
#             print(s_out,end='\r')

#             #v_env.render('human')

