import torch
from world_cityflow import World
from meta_policy.MetaPolicyTransformer import MetaPolicyTransformer
from meta_policy.MetaPolicyLSTM import MetaPolicyLSTM

# -------------------------
# Load world
# -------------------------
config_path = "cityflow_config.json"
world = World(config_path, thread_num=1)

print("World loaded successfully.")

# -------------------------
# Build meta-policy components
# -------------------------
transformer = MetaPolicyTransformer(
    d_reg=4,   # 4-dim region features
    num_regions=4,
    depth=3,
    heads=2
)

lstm = MetaPolicyLSTM(
    input_dim=16,   # flattened region embeddings
    hidden_dim=256,
    num_layers=2
)

# -------------------------
# Fake test input from world
# -------------------------
region_vec = world.get_region_features()   # (4, 4)
print("Region features:", region_vec.shape)

# Add batch dimension and time dimension
region_seq = region_vec.unsqueeze(0).unsqueeze(1)  # (B=1, T=1, 4, 4)

# -------------------------
# Transformer test
# -------------------------
F_g, subregion_seq = transformer(region_seq)
print("Transformer global embedding:", F_g.shape)
print("Transformer subregion seq:", subregion_seq.shape)

# -------------------------
# LSTM test
# -------------------------
subregion_flat = subregion_seq  # already (B, T, 16)
subgoal = lstm(subregion_flat)  # output should be (B, 4)
print("Subgoal:", subgoal.shape)

# -------------------------
# Step the world 1 timestep
# -------------------------
actions = [0] * len(world.intersections)
world.step(actions)

print("World step completed.")
