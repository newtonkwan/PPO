import torch
from torch.distributions import MultivariateNormal
from network import FeedForwardNN

class PPO: 

    def __init__(self, env):
        # init hyperparameters 
        self._init_hyperparameters()

        # Extract environment information
        self.env = env 
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action.shape[0]

        # ALG STEP 1 
        # initialize actor and critic networks 
        self.actor = FeedForwardNN(self.obs_dim, self.act_dim)
        self.critic = FeedForwardNN(self.obs_dim, 1) 

        # create variables for multivariate distribution 
        self.cov_var = torch.full(size=(self.act_dim, ), fill_value=0.5) # 0.5 is arbitrary
        self.cov_mat = torch.diag(self.cov_var)

    def learn(self, total_timesteps):
        # will increment later

        t_so_far = 0 
        while t_so_far < total_timesteps: # ALG STEP 2 
            # ALG STEP 3 
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()

    
    def _init_hyperparameters(self):
        # defeault values for hyperparameters, will need to change later 

        self.timesteps_per_batch = 4800         # timesteps per batch (number of episodes = rollouts = tau = trajectories)
        self.max_timesteps_per_episode = 1600   # timesteps per episode (FYI: trajectories = episodes = tau = rollouts) 
        self.gamma = 0.95                       # discount factor 

    def get_action(self, obs):
        # gets the action from the observation 

        # Query the actor network for a mean action 
        # same thing as calling self.actor.forward.obs()
        mean = self.actor(obs) 

        # create mv distribution 
        dist = MultivariateNormal(mean, self.cov_mat)

        # sample an action from the distribution and get its log prob 
        action = dist.sample()
        log_prob = dist.log_prob(action)

        # return the sampled action and the log prob of that action 
        # note that i'm calling detach() since the action and the log_prob
        # are tensors with computation graphs, so I want to get rid 
        # of the graph and just convert the action to numpy array.
        # log prob as tensor is fine. Our computation graph will
        # start later down the line 
        return action.detach().numpy(), log_prob.detach()

    def compute_rtgs(self, batch_rews):
        # the rewards to go (rtg) per episode per batch to return. 
        # the shape will be the (num timesteps per episode) 
        batch_rtgs = []

        # iterate through each epsiode backwards to maintain the same order in batch_rtgs 
        for ep_rews in reversed(batch_rews):
            discounted_reward = 0 # the discounted reward so far 

            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma 
                batch_rtgs.insert(0, discounted_reward)
            
        # convert the rewards to go to a tensor 
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)

        return batch_rtgs
    
    def rollout(self):
        # Batch data 
        batch_obs = []           # batch observations         shape = (# of timesteps per batch, dim of observation)
        batch_acts = []          # batch actions              shape = (# of timesteps per batch, dim of action)
        batch_log_probs = []     # log probs of each action   shape = (# of timesteps per batch)
        batch_rews = []          # batch rewards              shape = (# of episodes, number of timesteps per episode) 
        batch_rtgs = []          # batch rewards-to-go        shape = (# of timesteps per batch)
        batch_lens = []          # episodic lengths in batch  shape = (# of episodes)

        t = 0 # Timesteps simulated so far 

        while t < self.timesteps_per_batch: 
            '''
            # Generic gym rollout on one episode
            obs = self.env.reset()
            done = False 

            for ep_t in range(self.max_timesteps_per_episode):

                action = self.env.action_space.sample()
                obvs, rew, done, _ = self.env.step(action)

                if done:
                    break            
            '''
            # rewards thie episode 
            ep_rews = []

            obs = self.env.reset()
            done = False 

            for ep_t in range(self.max_timesteps_per_episode):
                # increment timesteps ran this batch so far 
                t += 1 

                # collect observation 
                batch_obs.append(obs)

                action, log_prob = self.get_action(obs)
                obs, rew, done, _ = self.env.step(action)

                # collect reward, action, and log prob 
                ep_rews.append(rew)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                if done:
                    break 

            # collect episodic length and rewards 
            batch_lens.append(ep_t + 1 ) # plus 1 bc timestep starts at 0 
            batch_rews.append(ep_rews)
        
        # reshape the data as tensors in the shape specified before returning 
        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)

        # ALG STEP 4 
        batch_rtgs = self.compute_rtgs(batch_rews)

        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens
                
            
                




