"""
Minimal HiLight rollout test.
Runs the full integrated architecture (local GAC + meta-policy + sub-policy)
for multiple timesteps in the CityFlow world.

Run with:
    (hilight) python -m tests.test_hilight_rollout
"""

import os
import torch
import numpy as np

from train_hilight import train_hilight
from world.world_cityflow import World
from common.metrics import Metrics
from common.registry import Registry
import agent


def build_world_and_agent():
    # Use the SAME ROOT computation as your working script
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # These are the paths you confirmed working
    cfg  = os.path.join(ROOT, "data", "raw_data", "grid4x4hl", "cityflow_config.json")
    flow = os.path.join(ROOT, "data", "raw_data", "grid4x4hl", "flow.json")

    # Build CityFlow world
    world = World(cfg, thread_num=1)

    # Build metrics object
    metric = Metrics(
        lane_metrics=['queue'],
        world_metrics=[],
        world=world,
        agents=[]
    )

    # Build HiLight agent from registry
    HilightAgent = Registry.get_model('hilight')
    agent = HilightAgent(world, metric, flow_path=flow)

    return world, agent


def run_rollout(num_steps=30):
    world, agent = build_world_and_agent()

    print("\n=== Starting HiLight Rollout ===\n")

    # Optional: clear meta-policy window if it exists
    if hasattr(agent, "regional_window"):
        agent.regional_window.clear()

    train_hilight()

    for t in range(num_steps):

        # 1) Compute full HiLight action (meta-policy + GAC + sub-policy)
        actions = agent.get_raw_action()

        # 2) Step the CityFlow engine with the chosen actions
        world.step(actions)

        # 3) Get immediate reward
        reward_vec = agent.get_reward()  # numpy array shape (16,)

        # Logging only the first few intersections to avoid 80-line prints
        inter_ids = world.intersection_ids
        sample_actions = [(inter_ids[i], actions[i]) for i in range(min(3, len(inter_ids)))]

        average_delay = agent.metric.delay()
        average_travel_time = agent.metric.real_average_travel_time()
        print(f"Step {t:03d} | actions: {sample_actions} | "
              f"reward[0]={float(reward_vec[0]):.3f} | ADT={average_delay:.3f} | ATT={average_travel_time:.3f}")



    print("\n=== Rollout Finished ===\n")


if __name__ == "__main__":
    run_rollout(num_steps=300)