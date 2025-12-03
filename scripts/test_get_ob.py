# scripts/test_get_ob.py

import os
import numpy as np
import torch
import torch.nn as nn

from world.world_cityflow import World
from common.metrics import Metrics
from common.registry import Registry

from agent.hilight import LocalEncoderMLP
from agent.hilight_gac import GraphAttentionConcat


def project_root():
    """
    Get the project root directory regardless of where this file is run from.
    """
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():

    ROOT = project_root()

    # ---------------------------------------------------------------
    # 1. Load CityFlow config safely (NO HARDCODED ABSOLUTE PATHS!)
    # ---------------------------------------------------------------
    print("1. Load grid4x4hl CityFlow config")

    cityflow_cfg = os.path.join(
        ROOT, "data", "raw_data", "grid4x4hl", "cityflow_config.json"
    )

    if not os.path.exists(cityflow_cfg):
        raise FileNotFoundError(f"CityFlow config not found: {cityflow_cfg}")

    world = World(cityflow_cfg, thread_num=1)

    # ---------------------------------------------------------------
    # 2. Build metrics object
    # ---------------------------------------------------------------
    print("2. Build metric")

    lane_metrics = ['queue']
    world_metrics = []
    agents = []

    metric = Metrics(lane_metrics, world_metrics, world, agents)

    # ---------------------------------------------------------------
    # 3. Initialize HiLight agent (flow.json also root-safe)
    # ---------------------------------------------------------------
    print("3. Create Hilight agent")

    flow_path = os.path.join(
        ROOT, "data", "raw_data", "grid4x4hl", "flow.json"
    )

    HilightAgent = Registry.get_model('hilight')
    hilight_agent = HilightAgent(world, metric, flow_path=flow_path)

    # ---------------------------------------------------------------
    # 4. Inspect get_ob()
    # ---------------------------------------------------------------
    print("4. Inspect get_ob()")

    obs = hilight_agent.get_ob()

    print("----- get_ob() output -----")
    for k, v in obs.items():
        print(k, type(v), getattr(v, "shape", None))

    # Show one sample lane feature from intersection 0
    print("Sample features for intersection 0, lane 0:", obs["sub_obs"][0, 0, :])

    # ---------------------------------------------------------------
    # 5. Build GAC input
    # ---------------------------------------------------------------
    sub_obs = obs["sub_obs"]
    gac_input = hilight_agent.build_gac_input(sub_obs)

    print("sub_obs shape:", sub_obs.shape)
    print("gac_input shape:", gac_input.shape)
    print("approach lanes for inter 0:", hilight_agent.inter_approach_lanes[0])
    print("pressures for inter 0:", gac_input[0,0,48:52])

    # Utility info
    iid0 = world.intersection_ids[0]
    print("inter 0 lane_ids:", hilight_agent.inter_in_lanes[iid0])

    # ---------------------------------------------------------------
    # 6. Local feature encoder test (LocalEncoderMLP)
    # ---------------------------------------------------------------
    device = torch.device("cpu")
    gac_input_t = torch.from_numpy(gac_input).float().to(device)

    local_encoder = LocalEncoderMLP(in_dim=56, hidden_dim=128, out_dim=56)
    gac_embeddings = local_encoder(gac_input_t)

    print("local_embeddings shape:", gac_embeddings.shape)
    print("sample embedding for intersection 0:", gac_embeddings[0,0,:8])

    # ---------------------------------------------------------------
    # 7. Test GraphAttentionConcat
    # ---------------------------------------------------------------
    neighbor_index = hilight_agent.neighbor_index.to(device)

    gac = GraphAttentionConcat(in_dim=56, num_neighbors=4).to(device)
    z = gac(gac_embeddings, neighbor_index)

    print("GAC output shape:", z.shape)
    print("sample GAC feature for inter 0:", z[0, 0, :10])


if __name__ == "__main__":
    main()
