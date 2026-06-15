from torch import optim
import torch
from torch import nn

from core.network import PolicyValueNetwork

class Trainer:
    def __init__(self, network: PolicyValueNetwork, optimizer: optim.Adam):
        self.network = network
        self.optimizer = optimizer
    
    def policy_loss(self, policy_pred, policy_true, epsilon=1e-15):
        a = policy_true * torch.log(policy_pred + epsilon)
        return - torch.mean(a.sum(dim=1))
    
    def value_loss(self, value_pred, value_true):
        loss_fn = nn.MSELoss()
        loss = loss_fn(value_pred, value_true)
        return loss


    def step(self, s, pi, z):         
        self.optimizer.zero_grad(set_to_none=True)  
        self.network.train()

        policy_head, value_head = self.network(s)

        loss = self.policy_loss(policy_head, pi) + self.value_loss(value_head, z)
        loss.backward()
        
        self.optimizer.step()



