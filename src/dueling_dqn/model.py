import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module): # TODO
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        
                
        """
        TODO: IMPLEMENT DUELING DQNS: SEPARATE VALUE FROM ACTION
        
        
        ONE NET OUTPUT WILL BE A SINGLE VALUE FOR THE STATE
        THE OTHER OUTPUT WILL BE ONE VALUE FOR EACH ACTION ???
        
        """
        self.action_size = action_size
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 64)
        self.ac1 = nn.ReLU()
        
        # Advantage values
        self.fc2_a = nn.Linear(64, 64)
        self.ac2_a = nn.ReLU()
        self.fc3_a = nn.Linear(64, action_size)
#         self.ac3_a = nn.ReLU()
        
        # State value
        self.fc2_s = nn.Linear(64, 64)
        self.ac2_s = nn.ReLU()
        self.fc3_s = nn.Linear(64, 1)
#         self.ac3_s = nn.ReLU()

    def preprocess_input(self, state):
        return state

    def forward(self, state): # TODO
        """Build a network that maps state -> action values."""
        # perform the forward pass
        # x = self.preprocess_input(state)
        x = self.ac1(self.fc1(state))
        
        # advantage values
        ad = self.ac2_a(self.fc2_a(x))
        ad = self.fc3_a(ad)
        
        # state value 
        sv = self.ac2_s(self.fc2_s(x))
        sv = self.fc3_s(sv)
        
# #         print(f"Advantage tensor shape: {ad.shape}. State value: {sv.shape}")
#         print(f"Advantage tensor: {ad}. State value: {sv}")
#         print(f"Sum: {ad+sv}")
        return ad + sv