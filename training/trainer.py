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
        self.device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else "cpu"

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

        s = s.to(self.device)
        pi = pi.to(self.device)
        z = z.to(self.device)

        policy_head, value_head = self.network(s)
        
        p_loss = self.policy_loss(policy_head, pi)
        v_loss = self.value_loss(value_head, z)
        loss = p_loss + v_loss

        loss.backward()

        self.optimizer.step()

        return loss, p_loss, v_loss
    


