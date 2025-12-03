import os
import torch
from world.world_cityflow import World
from common.metrics import Metrics
import agent
from common.registry import Registry

# 1. Build world
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cfg = os.path.join(ROOT, "data", "raw_data", "grid4x4hl", "cityflow_config.json")
flow = os.path.join(ROOT, "data", "raw_data", "grid4x4hl", "flow.json")

world = World(cfg, thread_num=1)

metric = Metrics(
    lane_metrics=['queue'],
    world_metrics=[],
    world=world,
    agents=[]
)

# 2. Build HiLight agent
HilightAgent = Registry.get_model('hilight')
agent = HilightAgent(world, metric, flow_path=flow)

# 3. Run forward pass for ~25 steps
print("Running forward test...")
for t in range(25):
    actions = agent.get_action()
    print(f"Step {t}, sample action:", list(actions.items())[:3])
