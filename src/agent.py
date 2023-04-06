import numpy as np
import random
from collections import namedtuple, deque
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.optim as optim
from model import QNetwork


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, 
                 state_size, 
                 action_size,
                 t_step=0,
                 alpha=0.1,
                 model_path=None,
                 buffer_size=int(1e5),
                 batch_size=64,
                 gamma=0.99,
                 tau=1e-3,
                 learning_rate=5e-4,
                 update_every=4,
                 seed=0):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Q-Network
        if not model_path:
            print(f"Creating new models for local and target networks.")
            self.qnetwork_local = QNetwork(state_size, action_size, seed).to(self.device)
            self.qnetwork_target = QNetwork(state_size, action_size, seed).to(self.device)
            self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=learning_rate)
            self.training = True
        else:
            print(f"Loading model from file {model_path}")
            self.qnetwork_local = QNetwork(state_size, action_size, seed).to(self.device)
            self.qnetwork_target = QNetwork(state_size, action_size, seed).to(self.device)
            self.qnetwork_local.load_state_dict(torch.load(model_path))
            self.qnetwork_target.load_state_dict(torch.load(model_path))
            self.training = False
        
        # Replay memory
        self.memory = PrioritizedReplayBuffer(action_size, buffer_size, batch_size, seed, self.device)
        
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = t_step
        self.alpha = alpha
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.learning_rate = learning_rate
        self.update_every = update_every
        

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Compute target values: # reward + gamma * max(Q(s', a')) * (1-done)
        td_targets_next_states = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        td_targets = rewards + (gamma*td_targets_next_states * (1-dones))
#         print(f"td_targets_next_states: {td_targets_next_states.shape}. Should be {rewards.shape[0]}")
        
        # Current Q values
        q_values_local = self.qnetwork_local(states).gather(1, actions) # Q
        
        # compute loss
        loss = F.mse_loss(q_values_local, td_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- # # Only update the target network after we pass on the samples
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)                     


    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
            
            
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)

                
    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
            
        if self.training:
            self.qnetwork_local.train()
        
        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))


class PrioritizedReplayBuffer: # TODO: Implement Prioritized Experience Replay    
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed, device):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.device = device
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
