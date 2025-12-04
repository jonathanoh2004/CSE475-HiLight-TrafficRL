import os
import torch
import torch.nn as nn

from world.world_cityflow import World
from common.metrics import Metrics
from common.registry import Registry
import agent  # ensure models get registered


def build_world_and_agent():
    """
    Build the CityFlow world + HiLight agent
    using the same config paths as your tests.
    """
    ROOT = os.path.dirname(os.path.abspath(__file__))

    cfg_path  = os.path.join(ROOT, "data", "raw_data", "grid4x4hl", "cityflow_config.json")
    flow_path = os.path.join(ROOT, "data", "raw_data", "grid4x4hl", "flow.json")

    world = World(cfg_path, thread_num=1)

    metric = Metrics(
        lane_metrics=['queue'],
        world_metrics=[],
        world=world,
        agents=[]
    )

    HilightAgent = Registry.get_model('hilight')
    hi_agent = HilightAgent(world, metric, flow_path=flow_path)

    return world, hi_agent


def train_hilight(num_steps: int = 300, lr: float = 1e-3):
    world, hi_agent = build_world_and_agent()

    optimizer = torch.optim.Adam(hi_agent.parameters(), lr=lr)
    print("Optimizer initialized with", len(hi_agent.parameters()), "params")

    param_count = sum(p.numel() for p in hi_agent.parameters())
    # Hyperparameters for clipping
    R_CLIP = 5.0  # reward clipping range [-R_CLIP, R_CLIP]
    V_CLIP = 10.0  # value clipping range [-V_CLIP, V_CLIP]
    MAX_GRAD_NORM = 0.5  # gradient clipping for stability

    print("Starting HiLight training loop...")

    for t in range(num_steps):
        # 1) Roll environment one step & collect training stats
        actions, log_prob_mean, value_mean, reward = hi_agent.step_train()

        #reward = max(min(reward, 5.0), -5.0)

        # 2) Build scalar tensors for losses
        reward_tensor = torch.tensor(
            reward,
            dtype=torch.float32,
            device=value_mean.device
        )

        value_clamped = torch.clamp(value_mean, -V_CLIP, V_CLIP)

        # Advantage for policy loss (no gradient through reward/value here)
        advantage = reward_tensor - value_mean.detach()

        policy_loss = -log_prob_mean * advantage
        value_loss = (value_mean - reward_tensor) ** 2

        # Combine losses; .mean() to ensure scalar if tensors are not scalars
        policy_loss = policy_loss.mean()
        value_loss = value_loss.mean()

        loss = policy_loss + 0.1 * value_loss


        # 3) Backprop + update
        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(hi_agent.parameters(), MAX_GRAD_NORM)

        optimizer.step()

        # 4) Logging
        if (t + 1) % 10 == 0:
            print(
                f"Step {t+1:04d} | "
                f"reward={reward:.3f} | "
                f"policy_loss={float(policy_loss.item()):.4f} | "
                f"value_loss={float(value_loss.item()):.4f}"
            )

    print("Training finished.")


if __name__ == "__main__":
    # tweak num_steps / lr as needed
    train_hilight(num_steps=300, lr=1e-3)
