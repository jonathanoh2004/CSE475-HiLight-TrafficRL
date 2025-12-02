import sys, os
import torch

# ============================================================
# 1. Resolve project root (CSE475-HiLight-TrafficRL)
# ============================================================
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(TEST_DIR)   # one level up

# Add root to PYTHONPATH for imports
sys.path.append(ROOT_DIR)

# ============================================================
# 2. Build absolute paths for CityFlow config + files
# ============================================================

CONFIG_DIR = os.path.join(ROOT_DIR, "data", "raw_data", "grid4x4hl")
CONFIG_PATH = os.path.join(CONFIG_DIR, "cityflow_config.json")
ROADNET_PATH = os.path.join(CONFIG_DIR, "roadnet.json")
FLOW_PATH = os.path.join(CONFIG_DIR, "flow.json")

# ============================================================
# 3. Validate all required files exist (debug helper)
# ============================================================
def check_file(path):
    if not os.path.exists(path):
        print(f"❌ MISSING: {path}")
    else:
        print(f"✔ Found: {path}")

print("\n=== Checking CityFlow Files ===")
check_file(CONFIG_PATH)
check_file(ROADNET_PATH)
check_file(FLOW_PATH)
print("================================\n")

# ============================================================
# 4. Imports (safe because ROOT_DIR is now in sys.path)
# ============================================================
from world.world_cityflow import World
from meta_policy.MetaPolicyTransformer import MetaPolicyTransformer
from meta_policy.MetaPolicyLSTM import MetaPolicyLSTM

# ============================================================
# 5. Load CityFlow world
# ============================================================
print(f"Using config: {CONFIG_PATH}")

world = World(CONFIG_PATH, thread_num=1)
print("\n=== World Loaded Successfully ===\n")

# ============================================================
# Build Meta-Policy Components
# ============================================================
transformer = MetaPolicyTransformer(
    d_reg=4,
    n_regions=4,
    n_layers=3,
    n_heads=2
)

lstm = MetaPolicyLSTM(
    M=4,
    d_reg=4,
    hidden_size=256
)

print("Meta-Policy Transformer and LSTM initialized.\n")

# ============================================================
# 20-step feature collection test
# ============================================================
T = 20
region_features_sequence = []

print("Collecting 20 timesteps of region features...\n")

for t in range(T):
    region_vec = world.get_region_features()           # numpy (4,4)
    region_vec = torch.tensor(region_vec, dtype=torch.float32)

    region_features_sequence.append(region_vec)

    actions = [0] * len(world.intersections)
    world.step(actions)

    print(f"Timestep {t+1}/{T} collected.")

region_seq = torch.stack(region_features_sequence).unsqueeze(0)
print("\nRegion sequence final shape:", region_seq.shape)

# ============================================================
# Run Transformer
# ============================================================
F_g, subregion_seq = transformer(region_seq)

print("\n=== Transformer Output ===")
print("Global embedding F_g:", F_g.shape)
print("Subregion sequence:", subregion_seq.shape)

# ============================================================
# Run LSTM
# ============================================================
subgoal = lstm(subregion_seq)

print("\n=== LSTM Output ===")
print("Subgoal shape:", subgoal.shape)
print("Subgoal:", subgoal)
print()
