# scripts/test_get_ob.py

from world.world_cityflow import World
from common.metrics import Metrics
from common.registry import Registry
import agent

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
                        # doesn't actually use them for queue()
    metric = Metrics(lane_metrics, world_metrics, world, agents)

    print("3. Create Hilight agent")
    HilightAgent = Registry.get_model('hilight')
    agent = HilightAgent(world, metric, flow_path="data/raw_data/grid4x4/flow.json")

    print("4. Inspect get_ob()")
    obs = agent.get_ob()
    print("----- get_ob() output -----")
    for k, v in obs.items():
        print(k, type(v), getattr(v, "shape", None))
    
    print("Sample features for intersection 0, lane 0:", obs["sub_obs"][0, 0, :])

if __name__ == "__main__":
    main()

