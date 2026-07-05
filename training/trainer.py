from torch import optim
import torch
from torch import nn
import torch.optim.lr_scheduler as lr_scheduler
import time

from core.network import PolicyValueNetwork
from profiler import timing as prof
from core import factory
from selfplay.generator import SelfPlayGenerator

class Trainer:
    def __init__(self, network: PolicyValueNetwork, optimizer: optim.Optimizer, T_max=100):
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

    def step(self, s, pi, mask, z):
        start = time.time()

        self.optimizer.zero_grad(set_to_none=True)
        self.network.train()

        device = next(self.network.parameters()).device

        pi = pi.to(device)
        z  = z.to(device)
        mask = mask.to(device)

        policy_head, value_head = self.network(s)

        # masking illegal action
        policy_head = policy_head.masked_fill(~mask, -1e-9)

        p_loss = self.policy_loss(policy_head, pi)
        v_loss = self.value_loss(value_head, z)
        loss = p_loss + v_loss

        loss.backward()

        self.optimizer.step()

        end = time.time()
        network_optimize_time = end - start
        prof.add(network_optimize=network_optimize_time)

        return loss, p_loss, v_loss


if __name__ == "__main__":
    GAME        = "chess"
    NUM_GAMES   = 2   
    NUM_ROLLOUT = 16  
    MCTS_BATCH  = 16  
    TRAIN_BATCH = 8    

    network = factory.build_network(GAME)
    buffer  = factory.build_buffer()

    gen   = SelfPlayGenerator(game=GAME, network=network,
                              num_rollout=NUM_ROLLOUT, batch_size=MCTS_BATCH)
    games = gen.generate(num_games=NUM_GAMES)
    for trajectory, result in games:
        buffer.add(trajectory, result)

    print(f"games={len(games)} | "
          f"moves={[len(t) for t, _ in games]} | "
          f"results={[r for _, r in games]} | "
          f"buffer_len={len(buffer)}")

    batch = min(TRAIN_BATCH, len(buffer))
    s, pi, mask, z = buffer.sample(batch)
    print(f"sampled s={tuple(s.shape)} pi={tuple(pi.shape)} mask={tuple(mask.shape)} z={tuple(z.shape)} "
          f"| pi.sum(dim=1)[:3]={pi.sum(dim=1)[:3].tolist()} "
          f"| z[:3]={z.squeeze(-1)[:3].tolist()}")

    # One step.
    optimizer = optim.Adam(network.parameters(), lr=1e-3)
    trainer   = Trainer(network, optimizer)
    loss, p_loss, v_loss = trainer.step(s, pi, mask, z)
    print(f"loss={loss.item():.4f} | policy={p_loss.item():.4f} | value={v_loss.item():.4f}")