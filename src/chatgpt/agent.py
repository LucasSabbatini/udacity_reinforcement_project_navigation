import numpy as np
import random
from collections import namedtuple, deque
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, 
                 state_size, 
                 action_size,
                 t_step=0,
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
#         self.memory = PrioritizedReplayBuffer(action_size, buffer_size, batch_size, seed, self.device)
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, seed, self.device)
        
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = t_step
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

#         # Compute target values: # reward + gamma * max(Q(s', a')) * (1-done)
#         td_targets_next_states = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
#         td_targets = rewards + (gamma*td_targets_next_states * (1-dones))
        
        """
        TODO: IMPLEMENT DOUBLE DQN: INSTEAD OF CALCULATING THE MAX VALUE USING THE TARGET NETWORK, 
        WE WILL GET THE INDEX OF THE ACTION USING THE LOCAL NETWORK. WITH THIS INDEX, WE WILL GET THE VALUE CALCULATED
        USING THE TARGET NETWORK
        
        From the paper:
        "In the original Double Q-learning algorithm, two value
        functions are learned by assigning each experience randomly to update one of the two value functions, such that
        there are two sets of weights, θ and θ`. For each update, one
        set of weights is used to determine the greedy policy and the
        other to determine its value. For a clear comparison, we can
        first untangle the selection and evaluation in Q-learning..."
        
        PSEUDOCODE
        td_targets_next_states_indexes = np.argmax(self.qnetwork_local(next_states), axis=1)
        td_targets_next_states = self.qnetwork_target(next_states).gather(td_targets_next_states_indexes)
        
        """
#         td_targets_next_states_indexes = np.argmax(self.qnetwork_local(next_states).detach(), axis=1).unsqueeze(1) # Get the action index using the local network
#         td_targets_next_states = self.qnetwork_target(next_states).detach().gather(1, td_targets_next_states_indexes) # use the indexes to get the values from the target network
#         td_targets = rewards + (gamma*td_targets_next_states * (1-dones)) # calculate the target just as before
    
        # Get index of max Q values of next states from local model
        next_state_actions = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1) # DOES THIS RETURN THE INDEX OF THE MAX VALUE????

        # Get max Q values of next states from target model
        td_targets_next_states = self.qnetwork_target(next_states).gather(1, next_state_actions)

        # Compute Q targets for current states 
        td_targets = rewards + (gamma * td_targets_next_states * (1 - dones))    
    
        
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

            
    def calculate_error_delta(self, state, action, reward, next_state, done):
        """
        TODO: compute the error delta for the current state, action, reward and next_stat set
        """
        # Compute target values: # reward + gamma * max(Q(s', a')) * (1-done)
        # transform input state to torch structure
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        next_state = torch.from_numpy(next_state).float().unsqueeze(0).to(self.device)
        action = np.asarray(action)
        action = torch.from_numpy(action).long().unsqueeze(0).to(self.device)
        action = action.expand(1,1)
#         print(f"State: {state.size()}. Next state: {next_state.size()}. Action: {action.size()}")
        
        # VANILA DQN. DO DOUBLE DQN LATER
#         td_targets_next_state = self.qnetwork_target(next_state).detach().max(1)[0].unsqueeze(1)
#         td_target = reward + (gamma*td_targets_next_state * (1-done)) # Target
        
        # Using double DQN
        td_targets_next_states_indexes = np.argmax(self.qnetwork_local(next_state).detach(), axis=1).unsqueeze(1) # Get the action index using the local network
        td_targets_next_states = self.qnetwork_target(next_state).detach().gather(1, td_targets_next_states_indexes) # use the indexes to get the values from the target network
        td_target = reward + (gamma*td_targets_next_states * (1-done)) # calculate the target just as before
        
        # CHATGPT VERSION
#         # Get index of max Q values of next states from local model
#         next_state_actions = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1) # DOES THIS RETURN THE INDEX OF THE MAX VALUE????

#         # Get max Q values of next states from target model
#         td_targets_next_states = self.qnetwork_target(next_states).gather(1, next_state_actions)

#         # Compute Q targets for current states 
#         td_targets = rewards + (gamma * td_targets_next_states * (1 - dones))  
        
        q_value_local = self.qnetwork_local(state).gather(1, action) # Q current
        return abs(td_target - q_value_local)

            
    def step(self, state, action, reward, next_state, done):
        
        """
        TODO: add error delta to the memory along each entry
        """
#         error = self.calculate_error_delta(state, action, reward, next_state, done)
#         # Save experience in replay memory
#         self.memory.add(state, action, reward, next_state, done, error)

        
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


class PrioritizedReplayBuffer:
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
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done", "error"])
        self.seed = random.seed(seed)
        self.a = 1
    
    def add(self, state, action, reward, next_state, done, error):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done, error)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        
        # Choose samples based on the probability, which is proportional to the error 
        priorities = np.asarray([e.error for e in self.memory], dtype=np.float64)
        with torch.no_grad():
            probabilities = priorities / sum(priorities) # It gives out a torch grad error, but this has never been in the graph, so why????? 
        experiences = random.choices(self.memory, weights=probabilities)
        
#         # now it is choosing the samples randomly. Change it to use the prioritized formula
#         experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
  
    
class ReplayBuffer: 
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