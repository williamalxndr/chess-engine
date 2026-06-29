from torch import optim
import torch
from torch import nn
import torch.optim.lr_scheduler as lr_scheduler
import time

from core.network import PolicyValueNetwork
from profiler import timing as prof

class Trainer:
    def __init__(self, network: PolicyValueNetwork, optimizer: optim.Adam, T_max=100):
        self.network = network
        self.optimizer = optimizer
        self.scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max or 100)
        
    def policy_loss(self, policy_logits, policy_true, epsilon=1e-15):
        log_probs = torch.nn.functional.log_softmax(policy_logits, dim=1)
        a = policy_true * log_probs
        return -torch.mean(a.sum(dim=1))    
    
    def value_loss(self, value_pred, value_true):
        loss_fn = nn.MSELoss()
        loss = loss_fn(value_pred, value_true)
        return loss


    def step(self, s, pi, z):
        start = time.time()

        self.optimizer.zero_grad(set_to_none=True)  
        self.network.train()

        device = next(self.network.parameters()).device

        pi = pi.to(device)
        z  = z.to(device)

        policy_head, value_head = self.network(s)
        
        p_loss = self.policy_loss(policy_head, pi)
        v_loss = self.value_loss(value_head, z)
        loss = p_loss + v_loss

        loss.backward()

        self.optimizer.step()

        end = time.time()
        network_optimize_time = end - start
        prof.add(network_optimize=network_optimize_time) 

        return loss, p_loss, v_loss
    


