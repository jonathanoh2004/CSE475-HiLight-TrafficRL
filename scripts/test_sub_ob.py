# scripts/test_sub_ob.py

from world.world_cityflow import World
from common.metrics import Metrics
from common.registry import Registry
import agent
import numpy as np

FEATURE_NAMES = [
    "car_num",        # 0
    "queue_norm",     # 1
    "occupancy",      # 2
    "flow_norm",      # 3
    "stop_car_num",   # 4
    "waiting_norm",   # 5
    "speed_norm",     # 6
    "pressure_norm",  # 7
    "delay_norm",     # 8
]

def print_lane_features(lane_vec, prefix=""):
    """Pretty-print a 9-d lane feature vector with labels."""
    assert lane_vec.shape[-1] == len(FEATURE_NAMES)
    print(prefix, "{")
    for name, val in zip(FEATURE_NAMES, lane_vec):
        print(f"  {name:>13}: {float(val):.4f}")
    print("}")
    
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

    iid0 = world.intersection_ids[0]
    print("inter 0 lane_ids:", hilight_agent.inter_in_lanes[iid0])

    print("\n5. Step the world a few times and inspect again")
  
    actions = np.zeros(len(world.intersection_ids), dtype=int)
    n = 300
    for t in range(n):
        world.step(actions)           # advances CityFlow one step
        obs = hilight_agent.get_ob()  # recompute observations

        if t % 50 == 0:
            all_vehicles = world.eng.get_vehicles(include_waiting=True)
            print(f"t={t}, num vehicles={len(all_vehicles)}")

    sub_obs = obs["sub_obs"]
    gac_input = hilight_agent.build_gac_input(sub_obs)

    print("sub_obs shape after 300 steps:", sub_obs.shape)
    print("gac_input shape after 300 steps:", gac_input.shape)
    print("pressures for inter 0 after 300 steps:", gac_input[0, 0, 48:52])
    print("first lane features at inter 0 after 300 steps:", sub_obs[0, 0, :])
    np.set_printoptions(precision=3, suppress=True)
    print(sub_obs[0:11])

    print("\nLabeled features for a few lanes at intersection 0 after 300 steps:")
    for lane_idx in range( min(12, sub_obs.shape[1]) ):
        lane_vec = sub_obs[0, lane_idx, :]
        print_lane_features(lane_vec, prefix=f"  inter0, lane{lane_idx}")

    print("\nDebug: global state after stepping")
    print("Current sim time:", world.eng.get_current_time())
    all_vehicles = world.eng.get_vehicles(include_waiting=True)
    print("Total vehicles in sim:", len(all_vehicles))

    lane_vehicle_count = world.eng.get_lane_vehicle_count()
    print("Total cars on all lanes:", sum(lane_vehicle_count.values()))


    # 1) Sum over only the lanes you actually include in sub_obs
    incoming_lane_ids = []
    for inter_id in world.intersection_ids:
        incoming_lane_ids.extend(hilight_agent.inter_in_lanes[inter_id])

    total_from_world_incoming = sum(lane_vehicle_count[lid] for lid in incoming_lane_ids)

    # 2) Sum over car_num feature in sub_obs
    total_from_obs = sub_obs[..., 0].sum()

    print("total_from_world_incoming:", total_from_world_incoming)
    print("total_from_obs (sub_obs[...,0].sum()):", total_from_obs)

    assert abs(total_from_obs - total_from_world_incoming) < 1e-6

if __name__ == "__main__":
    main()