# scripts/test_get_ob.py

from world.world_cityflow import World
from common.metrics import Metrics
from common.registry import Registry
import agent
import numpy as np
import torch
import torch.nn as nn
from agent.hilight import LocalEncoderMLP

def main():
    print("1. Load grid4x4 CityFlow config")

    # This is the same file you just checked with os.path.exists
    cityflow_cfg = "data/raw_data/grid4x4/config.json"

    # World creates its own CityFlow engine internally
    world = World(cityflow_cfg, thread_num=1)

    print("2. Build metric")
    lane_metrics = ['queue']
    world_metrics = []  # nothing global needed for this test
    agents = []         # Metrics usually expects a list; empty is fine if it
                    
    metric = Metrics(lane_metrics, world_metrics, world, agents)

    print("3. Create Hilight agent")
    HilightAgent = Registry.get_model('hilight')
    hilight_agent = HilightAgent(world, metric, flow_path="data/raw_data/grid4x4/flow.json")

    print("4. Inspect get_ob()")
    obs = hilight_agent.get_ob()
    print("----- get_ob() output -----")
    for k, v in obs.items():
        print(k, type(v), getattr(v, "shape", None))
    
    print("Sample features for intersection 0, lane 0:", obs["sub_obs"][0, 0, :])

    sub_obs = obs["sub_obs"]          # (16, 12, 9)
    gac_input = hilight_agent.build_gac_input(sub_obs)
    print("sub_obs shape:", sub_obs.shape)
    print("gac_input shape:", gac_input.shape)
    print("approach lanes for inter 0:", hilight_agent.inter_approach_lanes[0])
    print("pressures for inter 0:", gac_input[0,0,48:52])

    iid0 = world.intersection_ids[0]
    print("inter 0 lane_ids:", hilight_agent.inter_in_lanes[iid0])

    # testing local observation encoder
    device = torch.device("cpu") 
    gac_input_t = torch.from_numpy(gac_input).float().to(device)
    local_encoder = LocalEncoderMLP(in_dim=56, hidden_dim=128, out_dim=56)
    gac_embeddings = local_encoder(gac_input_t)

    print("local_embeddings shape:", gac_embeddings.shape)
    print("sample embedding for intersection 0:", gac_embeddings[0,0,:8])


    
if __name__ == "__main__":
    main()

