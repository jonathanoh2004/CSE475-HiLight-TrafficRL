import torch
from world_cityflow import World
from meta_policy.MetaPolicyTransformer import MetaPolicyTransformer
from meta_policy.MetaPolicyLSTM import MetaPolicyLSTM

# =====================
# Load CityFlow world
# =====================
config_path = "cityflow_config.json"
world = World(config_path, thread_num=1)
print("\n=== World Loaded Successfully ===\n")

# =====================
# Build Meta-Policy Components
# =====================
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

# =====================
# 20 timesteps of region features
# =====================

T = 20
region_features_sequence = []

print("Collecting 20 timesteps of region features...\n")

for t in range(T):

    # Get region features 4x4 numpy
    region_vec = world.get_region_features()  
    region_vec = torch.tensor(region_vec, dtype=torch.float32)  # (4,4)

    region_features_sequence.append(region_vec)

    # Step world with "do nothing" action since this is just a simple test to test workflow
    actions = [0] * len(world.intersections)
    world.step(actions)

    print(f"Timestep {t+1}/{T} collected.")


# =====================
# Stack region features into full sequence (shape = (B=1, T=20, M=4, d_reg=4))
# =====================

region_seq = torch.stack(region_features_sequence).unsqueeze(0)
print("\nRegion sequence final shape:", region_seq.shape)
print("Expected shape: (1, 20, 4, 4)\n")

# =====================
# Run Transformer
# =====================

F_g, subregion_seq = transformer(region_seq)

print("=== Transformer Output ===")
print("Global embedding F_g shape:", F_g.shape)        # (1,20,4)
print("Subregion sequence shape:", subregion_seq.shape)  # (1,20,16)
print()

# =====================
# Run LSTM
# =====================

subgoal = lstm(subregion_seq)

print("=== LSTM Output ===")
print("Subgoal shape:", subgoal.shape)        # (1,1,4)
print("Subgoal vector:", subgoal)
print()
