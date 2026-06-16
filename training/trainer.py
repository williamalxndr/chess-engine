from torch import optim
import torch
from torch import nn
import torch.optim.lr_scheduler as lr_scheduler

from core.network import PolicyValueNetwork

class Trainer:
    def __init__(self, network: PolicyValueNetwork, optimizer: optim.Adam, T_max=100):
        self.network = network
        self.optimizer = optimizer
        self.scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)

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

        s = s.unsqueeze(1)
        policy_head, value_head = self.network(s)

        loss = self.policy_loss(policy_head, pi) + self.value_loss(value_head, z)
        loss.backward()

        print(f"Loss: {loss}", end="\r", flush=True)     

        self.optimizer.step()
    


