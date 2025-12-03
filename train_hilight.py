import os
import torch

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


def train_hilight(num_steps: int = 1000, lr: float = 1e-4):
    world, hi_agent = build_world_and_agent()

    optimizer = torch.optim.Adam(hi_agent.parameters(), lr=lr)

    print("Starting HiLight training loop...")

    for t in range(num_steps):
        # 1) Roll environment one step & collect training stats
        actions, log_prob_mean, value_mean, reward = hi_agent.step_train()

        # 2) Build scalar tensors for losses
        reward_tensor = torch.tensor(
            reward,
            dtype=torch.float32,
            device=value_mean.device
        )

        # Advantage for policy loss (no gradient through reward/value here)
        advantage = reward_tensor - value_mean.detach()

        policy_loss = -log_prob_mean * advantage
        value_loss  = (value_mean - reward_tensor) ** 2

        loss = policy_loss + 0.5 * value_loss

        # 3) Backprop + update
        optimizer.zero_grad()
        loss.backward()
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
    train_hilight(num_steps=300, lr=1e-4)
