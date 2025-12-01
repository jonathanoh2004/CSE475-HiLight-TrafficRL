# scripts/test_get_ob.py

from world.world_cityflow import World
from common.metrics import Metrics
from common.registry import Registry
import agent
import numpy as np

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

    print("\n5. Step the world a few times and inspect again")
    # use a dummy action: all zeros, length = num_intersections
    actions = np.zeros(len(world.intersection_ids), dtype=int)

    for t in range(300):
        world.step(actions)           # advances CityFlow one step
        obs = hilight_agent.get_ob()  # recompute observations

        if t in [0, 5, 10, 20, 29, 35,40, 45, 49, 100, 150, 200, 299]:
            all_vehicles = world.eng.get_vehicles(include_waiting=True)
            print(f"t={t+1}, num vehicles={len(all_vehicles)}")

    sub_obs = obs["sub_obs"]
    gac_input = hilight_agent.build_gac_input(sub_obs)

    print("sub_obs shape after 300 steps:", sub_obs.shape)
    print("gac_input shape after 300 steps:", gac_input.shape)
    print("pressures for inter 0 after 300 steps:", gac_input[0, 0, 48:52])
    print("first lane features at inter 0 after 300 steps:", sub_obs[0, 0, :])
    print(sub_obs[0])

    print("\nDebug: global state after stepping")
    print("Current sim time:", world.eng.get_current_time())
    all_vehicles = world.eng.get_vehicles(include_waiting=True)
    print("Total vehicles in sim:", len(all_vehicles))

    lane_vehicle_count = world.eng.get_lane_vehicle_count()
    print("Total cars on all lanes:", sum(lane_vehicle_count.values()))

if __name__ == "__main__":
    main()

