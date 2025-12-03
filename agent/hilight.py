from common.registry import Registry
from .base import BaseAgent
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from collections import deque

from meta_policy.MetaPolicyTransformer import MetaPolicyTransformer
from meta_policy.MetaPolicyLSTM import MetaPolicyLSTM
from agent.hilight_gac import GraphAttentionConcat

from agent.hilight_gac import GraphAttentionConcat   # GAC module


@Registry.register_model('hilight')
class HilightAgent(BaseAgent):
    def __init__(self, world, metric, flow_path="flow.json"):
        super().__init__(world)
        self.world = world
        self.eng = world.eng
        self.metric = metric
        self.sub_agents = len(world.intersection_ids) # i think we need this so bc TSCEnv needs the nmber of agents to equal num of intersections

        # # ---- Models ----
        # self.meta_transformer = ...
        # self.meta_lstm = ...
        # self.local_mlp = ...
        # self.gat = ...
        # self.actor = ...
        # self.critic = ...

        # # ---- Buffers ----
        # self.regional_window = deque(maxlen=20)
        # self.sub_buffer = ReplayBuffer(...)
        # self.meta_buffer = ReplayBuffer(...)

        # # ---- Bookkeeping ----
        # self.current_goal = None
        # self.current_step = 0
        # self.meta_interval = 5
        
        # --- load vehicle length & minGap from agent config ---
        import json, math
        with open(flow_path) as f:
            cfg = json.load(f)
        
        if isinstance(cfg, list):
            # take the first flow entry as representative, i don't know if i should gather all vehicles lengths in the simulator and average them
            cfg_vehicle = cfg[0]["vehicle"]
        else:
            cfg_vehicle = cfg["vehicle"]

        self.vehicle_length = cfg_vehicle["length"]

        # handle both formats:
        # - traffic sim configs with "carFollow": {"minGap": ...}
        # - flow.json with "minGap" directly under "vehicle"
        car_follow = cfg_vehicle.get("carFollow")
        if car_follow is not None and "minGap" in car_follow:
            self.min_gap = car_follow["minGap"]
        else:
            self.min_gap = cfg_vehicle["minGap"]
    
        # --- lane capacity ---
        self.lane_capacity = {}

        for lane_id, lane_length in self.world.lane_length.items():
            self.lane_capacity[lane_id] = math.floor(
                lane_length / (self.vehicle_length + self.min_gap)
            )

        # # --- intersection -> incoming lanes mapping , these ar ethe list of lane_ids that enter this intersection---
        self.inter_in_lanes = {}
        for inter in self.world.intersections:          # Intersection objects
            inter_in_lanes = []
            for road in inter.in_roads:
                # same direction logic as get_in_out_lanes
                from_zero = (road["startIntersection"] == inter.id) if self.world.RIGHT else (
                            road["endIntersection"] == inter.id)
                lane_indices = range(len(road["lanes"])) if from_zero else range(len(road["lanes"]) - 1, -1, -1)
                for n in lane_indices:
                    lane_id = f'{road["id"]}_{n}'
                    inter_in_lanes.append(lane_id)
            self.inter_in_lanes[inter.id] = inter_in_lanes

        for iid, lanes in self.inter_in_lanes.items():
            if len(lanes) != len(set(lanes)):
                print("WARNING: duplicate lane IDs for intersection", iid)
                print(lanes)

        self._build_inter_approach_lanes() 

        # -------------------------------------------
        #  HiLight: Meta-Policy + Sub-Policy modules
        # -------------------------------------------

        # ---- 1) Local encoder MLP (per-intersection 56-d node features) ----
        # Input:  (B, N, 56)
        # Output: (B, N, 56)
        self.local_mlp = LocalEncoderMLP(in_dim=56, hidden_dim=128, out_dim=56)

        # ---- 2) Graph Attention Concat (GAC) ----
        # Input:  (B, N, 56)
        # Output: (B, N, 280)   # (K+1)*F = (4+1)*56
        self.gac = GraphAttentionConcat(in_dim=56, num_neighbors=4)

        # ---- 3) Meta-Policy: Transformer + LSTM ----
        # Region embedding dim (d_reg) = 4
        # Number of regions (M) = 4 for grid4x4hl setup
        self.d_reg = 4
        self.M = 4

        # ---- Region assignment for each intersection ----
        # 4x4 grid → rows 0–1 = region 0, 2–3 = region 1, etc.
        self.inter_to_region = []
        for inter_id in self.world.intersection_ids:
            x, y = self.world.intersection_id2coords[inter_id]
            region_x = 0 if x < 400 else 1
            region_y = 0 if y < 400 else 1
            region_idx = region_y * 2 + region_x
            self.inter_to_region.append(region_idx)

        self.inter_to_region = np.array(self.inter_to_region, dtype=np.int64)


        self.meta_transformer = MetaPolicyTransformer(
            d_reg=self.d_reg,
            n_regions=self.M,
            n_layers=3,
            n_heads=2,
            ff_dim=165,   # matches your existing transformer config
        )

        self.meta_lstm = MetaPolicyLSTM(
            M=self.M,
            d_reg=self.d_reg,
            hidden_size=256
        )

        # ---- 4) Sub-Policy Actor-Critic (shared over intersections) ----
        # Use number of signal phases for num_actions
        example_inter = self.world.intersections[0]
        self.num_actions = len(example_inter.phases)

        self.sub_policy = SubPolicyActorCritic(
            z_dim=280,
            d_reg=self.d_reg,
            num_actions=self.num_actions,
            hidden_dim=128,
        )

        # ---- 5) Neighbor index for GAC: (N, K) ----
        # Build once, based on spatial coordinates of intersections
        self._build_neighbor_index(K=4)

        # ---- 6) Regional window (20-step history for Meta-Policy) ----
        # Stores last 20 timesteps of regional features: (4,4) per step
        self.regional_window = deque(maxlen=20)

        self.trainable_modules = [
            self.local_mlp,
            self.gac,
            self.meta_transformer,
            self.meta_lstm,
            self.sub_policy
        ]

    def get_ob(self):

        """
        Build observations that follow Table 4:

            car_num       : totaled sum of all vehicles on each lane (per intersection)
            queue_length  : queue length in meters for each incoming lane (per lane)
            occupancy     : (# vehicles on lane) / (lane capacity) (per lane)
            flow          : # vehicles passing through intersection per unit time (per intersection)
            stop_car_num  : # vehicles with speed < 0.1 m/s, normalized by capacity (per lane)
            waiting_time  : waiting time of the FIRST vehicle on each lane (per lane)
            average_speed : average speed of all vehicles on lane (per intersection)
            pressure      : difference in vehicle count between incoming and
                            corresponding outgoing lanes (per approach)
            delay_time    : (actual travel time - ideal travel time) (per intersection)
        """

        world = self.world
        eng = self.eng
        metric = self.metric

        # car_num: sum of lane_vehicle_count per intersection
        lane_vehicle_count = world.info_functions["lane_count"]()

        lane_vehicles = world.info_functions["lane_vehicles"]()

        vehicle_distance = eng.get_vehicle_distance()

        vehicle_speed = eng.get_vehicle_speed()  

        vehicle_wait_time = world.get_vehicle_waiting_time()

        # flow: number of vehicles passing through per unit time
        lane_flow = world.info_functions["throughput"]() 
        
        # pressure: vehicle count difference between incoming and corresponding
        # outgoing lanes. 
        lane_pressure = world.get_lane_pressure()  

        # delay_time: actual − ideal travel time per lane (assumed to be provided)
        lane_delay = world.info_functions["lane_delay"]()  

        # ---------- BUILD PER-LANE FEATURES ----------
        sub_obs_list = []

        for inter_id in world.intersection_ids:

            lane_ids = self.inter_in_lanes[inter_id]
            lane_feat_list = []

            for lane_id in lane_ids:
                # lane capacity (for occupancy & normalized stop_car_num)
                cap = max(self.lane_capacity.get(lane_id, 1), 1)

                # 1) car_num: total # of vehicles on this lane
                car_num = int(lane_vehicle_count.get(lane_id, 0))

                # 2) queue_length: queue length in meters, metrics.queue() doesn't work
                v_list = lane_vehicles.get(lane_id, [])
                queue_vids = [vid for vid in v_list if vehicle_speed.get(vid, 0.0) < 0.1]

                if queue_vids:
                    distances = [vehicle_distance.get(vid, 0.0) for vid in queue_vids]
                    q_len = max(distances)                       # actual queue length in meters
                else:
                    q_len = 0.0

                # 3) occupancy: (# vehicles) / (lane capacity)
                occupancy = float(car_num) / float(cap)

                # 4) flow: # vehicles passing through per unit time (current step)
                fl = float(lane_flow.get(lane_id, 0.0)) if isinstance(lane_flow, dict) else 0.0

                # lane vehicles & speeds
                v_list = lane_vehicles.get(lane_id, [])
                speeds = [float(vehicle_speed.get(vid, 0.0)) for vid in v_list]

                # 5) stop_car_num:
                # of vehicles with speed < 0.1 m/s, normalized by cap
                stop_count = sum(1 for s in speeds if s < 0.1)
                stop_car_num = float(stop_count) / float(cap)

                # 6) waiting_time:
                if v_list and vehicle_wait_time:
                    first_vid = v_list[0]
                    waiting_time = float(vehicle_wait_time.get(first_vid, 0.0))
                else:
                    # Fallback: if world has a lane-level "first-vehicle" wait time
                    lane_wait_time = getattr(world, "get_lane_waiting_time_count", lambda: {})()
                    waiting_time = float(lane_wait_time.get(lane_id, 0.0))

                # 7) average_speed: average speed of all vehicles on this lane
                avg_speed = float(np.mean(speeds)) if speeds else 0.0

                # 8) pressure: difference between incoming and outgoing counts.
                pressure = float(lane_pressure.get(lane_id, 0.0))

                # 9) delay_time: actual travel time − ideal travel time
                delay_time = float(lane_delay.get(lane_id, 0.0))

                lane_feat = [
                    car_num,        # car_num [0]
                    q_len,          # queue_length [1]
                    occupancy,      # occupancy [2]
                    fl,             # flow [3]
                    stop_car_num,   # stop_car_num [4]
                    waiting_time,   # waiting_time [5]
                    avg_speed,      # average_speed [6]
                    pressure,       # pressure [7]
                    delay_time,     # delay_time [8]
                ]
                lane_feat_list.append(lane_feat)

            inter_feat = np.array(lane_feat_list, dtype=np.float32)
            sub_obs_list.append(inter_feat)

        # ---------- PAD & STACK TO FIXED SHAPE (num_inters, max_lanes, 9) ----------
        num_inters = len(world.intersection_ids)
        max_lanes = max(feat.shape[0] for feat in sub_obs_list)
        feat_dim = 9

        # At this point the feature dimension is 9 per-lane features, we compute the per-intersection (car num, delay time, average speed, flow)
        # and per-approach features (pressure) when we generate features (build_gac_input())
        sub_obs = np.zeros((num_inters, max_lanes, feat_dim), dtype=np.float32)
        for i, inter_feat in enumerate(sub_obs_list):
            n = inter_feat.shape[0]
            sub_obs[i, :n, :] = inter_feat

        return {"sub_obs": sub_obs}

    def _build_inter_approach_lanes(self):
        """
        Build mapping:
          self.inter_approach_lanes[inter_idx]["N"|"E"|"S"|"W"] -> list of row indices in sub_obs
        """
        # 1) intersection coordinates from roadnet.json
        inter_coord = {}
        for inter in self.world.roadnet["intersections"]:
            iid = inter["id"]
            x = inter["point"]["x"]
            y = inter["point"]["y"]
            inter_coord[iid] = (x, y)
        self.inter_coord = inter_coord #This line was added for the GAC

        # 2) intersection id -> index in sub_obs (0..num_inters-1)
        inter_id_to_idx = {iid: idx for idx, iid in enumerate(self.world.intersection_ids)}
        self.inter_id_to_idx = inter_id_to_idx

        # 3) lane_id -> (inter_idx, row_idx) 
        lane_to_inter_row = {}
        for iid, lane_ids in self.inter_in_lanes.items():
            inter_idx = inter_id_to_idx[iid]
            for row_idx, lane_id in enumerate(lane_ids):
                lane_to_inter_row[lane_id] = (inter_idx, row_idx)
        self.lane_to_inter_row = lane_to_inter_row

        # 4) init result structure
        num_inters = len(self.world.intersection_ids)
        inter_approach_lanes = {
            inter_idx: {"N": [], "E": [], "S": [], "W": []}
            for inter_idx in range(num_inters)
        }

        # 5) loop over intersections and their incoming roads
        for inter_obj in self.world.intersections:   # Intersection objects
            inter_id = inter_obj.id
            if inter_id not in inter_coord:
                continue

            inter_idx = inter_id_to_idx[inter_id]
            x0, y0 = inter_coord[inter_id]

            for road in inter_obj.in_roads:
                start = road["startIntersection"]
                end = road["endIntersection"]

                # other intersection of this road
                other = start if end == inter_id else end
                if other not in inter_coord:
                    continue

                x1, y1 = inter_coord[other]
                dx = x0 - x1
                dy = y0 - y1

                # determine direction of traffic as it enters inter_id
                # (grid4x4 is axis-aligned so this works nicely)
                if abs(dx) > abs(dy):
                    # mostly horizontal
                    if dx > 0:
                        # other is west of inter_id, traffic flows east into this intersection
                        dir_at_inter = "E"
                    else:
                        dir_at_inter = "W"
                else:
                    # mostly vertical
                    if dy > 0:
                        # other is south of inter_id, traffic flows north into this intersection
                        dir_at_inter = "N"
                    else:
                        dir_at_inter = "S"

                # pick lane indices for this incoming road in same order as inter_in_lanes
                if self.world.RIGHT:
                    from_zero = (road["startIntersection"] == inter_id)
                else:
                    from_zero = (road["endIntersection"] == inter_id)
                lane_indices = range(len(road["lanes"])) if from_zero else range(len(road["lanes"]) - 1, -1, -1)

                for n in lane_indices:
                    lane_id = f'{road["id"]}_{n}'
                    if lane_id in lane_to_inter_row:
                        inter_idx2, row_idx = lane_to_inter_row[lane_id]
                        # should be the same intersection
                        if inter_idx2 == inter_idx:
                            inter_approach_lanes[inter_idx][dir_at_inter].append(row_idx)

        self.inter_approach_lanes = inter_approach_lanes
        print("Approach lanes for inter 0:", self.inter_approach_lanes[0])
    
    def _build_neighbor_index(self, K: int = 4):
        """
        Build neighbor index tensor for GAC.

        Uses intersection coordinates from world.roadnet to create a
        K-nearest-neighbor graph over intersections.

        Result:
            self.neighbor_index : torch.LongTensor of shape (N, K)
                neighbor_index[i, :] = indices of K nearest neighbors of node i
        """
        import math

        world = self.world
        inter_ids = world.intersection_ids            # list of ids in fixed order
        N = len(inter_ids)

        # id -> index
        id2idx = {iid: idx for idx, iid in enumerate(inter_ids)}

        # gather coordinates for each intersection id
        coords = []
        for iid in inter_ids:
            x, y = world.intersection_id2coords[iid]
            coords.append((x, y))
        coords = np.array(coords, dtype=np.float32)   # (N, 2)

        neighbor_index = []

        for i in range(N):
            # compute distances from i to all j
            dists = []
            xi, yi = coords[i]
            for j in range(N):
                if i == j:
                    continue
                xj, yj = coords[j]
                dx = xi - xj
                dy = yi - yj
                dist = math.sqrt(dx * dx + dy * dy)
                dists.append((dist, j))

            # sort by distance
            dists.sort(key=lambda t: t[0])

            # pick up to K nearest neighbors
            nearest = [j for (_, j) in dists[:K]]

            # if fewer than K (tiny graph), pad with self index
            while len(nearest) < K:
                nearest.append(i)

            neighbor_index.append(nearest)

        # shape: (N, K)
        self.neighbor_index = torch.tensor(neighbor_index, dtype=torch.long)
        print("Neighbor index built for GAC with shape:", self.neighbor_index.shape)

    def update_regional_window(self):
        """
        Pull region-level features from the world and push them into the 20-step window.
        Each feature matrix is shape (4, 4).
        The window becomes a (T, 4, 4) list-like buffer.
        """
        region_feat = self.world.get_region_features()    # numpy (4,4)
        region_feat = torch.tensor(region_feat, dtype=torch.float32)

        self.regional_window.append(region_feat)          # maintain length 20

    def compute_meta_policy(self):
        """
        Uses the last 20 timesteps of regional features and computes:
            - F_g_t : global embedding at current timestep (B, 4)
            - G_t   : per-region subgoals (B, 4, 4)
        """
        # Need 20 timesteps before meta-policy is active
        if len(self.regional_window) < 20:
            # return neutral embeddings until enough history collected
            B = 1
            F_g_t = torch.zeros(B, self.d_reg)
            G_t   = torch.zeros(B, self.M, self.d_reg)
            return F_g_t, G_t

        # 1) Stack window into (B=1, T=20, M=4, d_reg=4)
        region_seq = torch.stack(list(self.regional_window), dim=0)  # (20,4,4)
        region_seq = region_seq.unsqueeze(0)                          # (1,20,4,4)

        # 2) Transformer → F_g_seq (1,20,4), subregion_seq (1,20,16)
        F_g_seq, subregion_seq = self.meta_transformer(region_seq)

        # 3) LSTM → G_seq (1,4,4)
        G_seq = self.meta_lstm(subregion_seq)

        # 4) Use the latest timestep’s global embedding
        F_g_t = F_g_seq[:, -1, :]      # (1,4)

        return F_g_t, G_seq            # G_seq = (1,4,4)
    
    def compute_local_features(self):
        """
        Local (per-intersection) feature pipeline:
            get_ob() -> build_gac_input -> LocalEncoderMLP -> GAC
        Returns:
            z : (B, N, 280)
        """
        # 1) raw lane-level obs → dict {"sub_obs": (num_inters, max_lanes, 9)}
        ob = self.get_ob()
        sub_obs = ob["sub_obs"]

        # 2) build 56-d features per intersection → (1, N, 56)
        gac_in = self.build_gac_input(sub_obs)
        x_local = torch.tensor(gac_in, dtype=torch.float32)

        # 3) Local MLP encoder → (1, N, 56)
        h_local = self.local_mlp(x_local)

        # 4) GAC → (1, N, 280)
        z = self.gac(h_local, self.neighbor_index)
        return z
    
    def compute_subpolicy_action(self):
        """
        Runs the complete HiLight pipeline:
            local features  -> z
            meta-policy     -> F_g_t, G_t
            sub-policy A+C  -> logits, values, actions

        Returns:
            actions : (N,) torch.LongTensor of phase indices
            logits  : (1, N, A)
            values  : (1, N)
        """
        # -------------------------
        # 1. Update regional window
        # -------------------------
        self.update_regional_window()

        # 2. Meta-policy (global & regional guidance)
        F_g_t, G_t = self.compute_meta_policy()

        # 3. Local features → z
        z = self.compute_local_features()    # (1, N, 280)
        B = 1
        N = z.shape[1]

        if len(self.regional_window) < 20:
            # No global or subgoal guidance
            F_g_t = torch.zeros((1, self.d_reg), dtype=torch.float32)
            G_expanded = torch.zeros((1, N, self.d_reg), dtype=torch.float32)

            logits, values = self.sub_policy(z, F_g_t, G_expanded)
            dist   = torch.distributions.Categorical(logits=logits)
            actions = dist.sample().squeeze(0)
            return actions, logits, values

        # Expand G_t (1,4,4) → (1,N,4)
        G_expanded = torch.zeros((B, N, self.d_reg), dtype=torch.float32)
        for i in range(N):
            region_idx = self.inter_to_region[i]
            G_expanded[0, i, :] = G_t[0, region_idx, :]

        logits, values = self.sub_policy(z, F_g_t, G_expanded)
        dist = torch.distributions.Categorical(logits=logits)
        actions = dist.sample().squeeze(0)
        return actions, logits, values
    
    def compute_subpolicy_action_with_logprobs(self):
        """
        Same pipeline as compute_subpolicy_action(), but also returns
        per-intersection log-probs and values for training.

        Returns:
            actions    : (N,) LongTensor
            log_probs  : (N,) Tensor
            values     : (N,) Tensor
        """
        # 1. Update regional window & meta-policy
        self.update_regional_window()
        F_g_t, G_t = self.compute_meta_policy()   # F_g_t: (1,4), G_t: (1,4,4)

        # 2. Local features
        z = self.compute_local_features()         # (1, N, 280)
        B = 1
        N = z.shape[1]

        # 3. Expand G_t (1,4,4) → (1,N,4) via inter_to_region
        G_expanded = torch.zeros((B, N, self.d_reg),
                                 dtype=torch.float32,
                                 device=z.device)
        for i in range(N):
            region_idx = self.inter_to_region[i]
            G_expanded[0, i, :] = G_t[0, region_idx, :]

        # 4. Sub-policy forward
        logits, values = self.sub_policy(
            z,
            F_g_t.to(z.device),
            G_expanded
        )  # logits: (1, N, A), values: (1, N)

        dist = torch.distributions.Categorical(logits=logits)

        # Sample actions + log_probs with gradients
        actions   = dist.sample()          # (1, N)
        log_probs = dist.log_prob(actions) # (1, N)

        # Squeeze batch dim → (N,)
        actions   = actions.squeeze(0)
        log_probs = log_probs.squeeze(0)
        values    = values.squeeze(0)

        return actions, log_probs, values

    def step_train(self):
        """
        One training step:
          - runs HiLight to choose actions
          - steps the CityFlow world
          - computes reward
          - returns scalars needed for actor-critic loss.

        Returns:
            actions_list : list[int]      length = #intersections
            log_prob_mean: scalar Tensor  (mean over intersections)
            value_mean   : scalar Tensor  (mean over intersections)
            reward       : float          (mean reward over intersections)
        """
        # 1) Policy forward with log-probs & values
        actions_tensor, log_probs, values = self.compute_subpolicy_action_with_logprobs()
        # actions_tensor, log_probs, values all shape: (N,)

        # 2) Step simulator using same format as rollout test (list of ints)
        actions_list = actions_tensor.detach().cpu().numpy().tolist()
        self.world.step(actions_list)

        # 3) Compute reward vector & scalar
        reward_vec = self.get_reward()                 # np.array shape (N,)
        reward     = float(reward_vec.mean())          # scalar for training

        # 4) Collapse log_probs & values over intersections
        log_prob_mean = log_probs.mean()
        value_mean    = values.mean()

        return actions_list, log_prob_mean, value_mean, reward

    def parameters(self):
        """Return all trainable torch parameters from internal modules."""
        params = []
        for module in self.trainable_modules:
            params += list(module.parameters())
        return params

    def get_action(self, ob=None, phase=None):
        """
        Returns a dict:
            { intersection_id : chosen_phase }
        matching CityFlow's API requirement.
        """
        actions, logits, values = self.compute_subpolicy_action()

        # Convert tensor actions → Python ints
        actions = self._clip_actions_to_valid_phases(actions).cpu().numpy().tolist()
        N = len(self.world.intersection_ids)
        action_dict = {i: actions[i] for i in range(N)}
        return action_dict
    
    def get_raw_action(self):
        """
        Returns raw action vector (list of ints) in intersection index order.
        This is the format expected by world.step().
        """
        actions, logits, values = self.compute_subpolicy_action()
        actions = self._clip_actions_to_valid_phases(actions)
        return actions.cpu().numpy().tolist()



    def get_reward(self):
        """
        Simple global reward:
        reward = - sum of waiting times across all lanes.
        get_lane_waiting_time_count() -> dict {lane_id: waiting_time}
        Returns a vector of size (num_intersections,) because TSCEnv
        expects one reward per sub-agent.
        """
        waiting_dict = self.world.get_lane_waiting_time_count()

        # Sum all waiting times
        total_wait = sum(float(v) for v in waiting_dict.values())

        # Negative reward → less waiting = better
        reward = -total_wait

        # Broadcast to all intersections (sub-agents)
        N = int(self.sub_agents)
        return np.full((N,), reward, dtype=np.float32)
    
    def _clip_actions_to_valid_phases(self, actions_tensor):
        actions = actions_tensor.clone()
        for i, inter in enumerate(self.world.intersections):
            num_phases = len(inter.phases)
            actions[i] = actions[i] % num_phases
        return actions


    def build_gac_input(self, sub_obs):
        # sub_obs: (num_inters, max_lanes, 9)
        num_inters, max_lanes, feat_dim = sub_obs.shape

        node_features = []

        # iterates through all intersections
        for inter_idx in range(num_inters):
            lanes = sub_obs[inter_idx]  # (max_lanes, 9)

            # --- 1) Per-lane part (48 dims) ---
            lane_core = lanes[:, [1, 2, 4, 5]]      # queue, occupancy, stop, waiting
            per_lane_flat = lane_core.flatten()     # (max_lanes*4 = 48,), this produces a 1D vector because GAT expects one feature vector per node

            # --- 2) Per-approach pressure (4 dims) ---
            approach_pressures = []
            for dir in ["N", "E", "S", "W"]:
                row_idxs = self.inter_approach_lanes[inter_idx][dir]  # list of lane row indices
                if row_idxs:
                    p = lanes[row_idxs, 7].sum()   # pressure is column 7
                else:
                    p = 0.0
                approach_pressures.append(p)
            approach_pressures = np.array(approach_pressures, dtype=np.float32)  # (4,)

            # --- 3) Per-intersection scalars (4 dims) ---
            car_num_inter   = lanes[:, 0].sum()
            flow_inter      = lanes[:, 3].sum()
            valid_mask = lanes[:, 0] > 0
            if valid_mask.any():
                avg_speed_inter = lanes[valid_mask, 6].mean()
            else:
                avg_speed_inter = 0.0
            delay_inter     = lanes[:, 8].sum()
            inter_scalars = np.array(
                [car_num_inter, flow_inter, avg_speed_inter, delay_inter],
                dtype=np.float32
            )  # (4,)

            # --- 4) concatenate to 56 dims ---
            node_feat_56 = np.concatenate(
                [per_lane_flat, approach_pressures, inter_scalars],
                axis=-1
            )  # (56,)
            node_features.append(node_feat_56)

        gac_input = np.stack(node_features, axis=0)  # (num_inters, 56)
        gac_input = gac_input[None, ...]            # (1, num_inters, 56)
        return gac_input


class LocalEncoderMLP(nn.Module):
    def __init__(self, in_dim=56, hidden_dim=128, out_dim=56):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        """
        x: (B, N, F)
           B = batch size
           N = number of intersections (nodes)
           F = feature dimension (56)
        """
        B, N, F = x.shape

        # Treat each node as a separate sample for the MLP:
        x = x.view(B * N, F)   

        # Apply the shared MLP to every node
        x = self.mlp(x)        

        # Reshape back to (B, N, out_dim)
        x = x.view(B, N, -1)   

        return x
    
class SubPolicyActorCritic(nn.Module):
    """
    HiLight Sub-Policy (Section 4.2):
      - Shared-parameter Actor-Critic over all intersections.
      - Input per node i:
            z_i  : GAC output  (280-d)
            F_g  : global feature from Meta-Policy  (4-d)
            g_i  : local sub-goal for region/node i (4-d)
        concat -> 288-d feature per intersection.

    Shapes:
        z      : (B, N, z_dim)       default z_dim = 280
        F_g    : (B, d_reg) or (B, 1, d_reg) or (B, N, d_reg)
        G      : (B, d_reg) or (B, 1, d_reg) or (B, N, d_reg)
        logits : (B, N, num_actions)
        values : (B, N)
    """

    def __init__(
        self,
        z_dim: int = 280,
        d_reg: int = 4,
        num_actions: int = 4,
        hidden_dim: int = 128,
    ):
        super().__init__()

        self.z_dim = z_dim
        self.d_reg = d_reg
        self.num_actions = num_actions

        # total input: z_i (280) + F_g (4) + g_i (4) = 288
        in_dim = z_dim + d_reg + d_reg

        # ----- Actor head -----
        self.actor = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

        # ----- Critic head -----
        self.critic = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def _tile_or_broadcast(self, x: torch.Tensor, N: int) -> torch.Tensor:
        """
        Utility: take x with shape:
            (B, d)    or
            (B, 1, d) or
            (B, N, d)
        and return a tensor of shape (B, N, d).
        """
        if x is None:
            return None

        if x.dim() == 2:
            # (B, d) → (B, 1, d) → (B, N, d)
            x = x.unsqueeze(1).expand(-1, N, -1)
        elif x.dim() == 3:
            B, T_or_N, d = x.shape
            if T_or_N == 1:
                # (B, 1, d) → (B, N, d)
                x = x.expand(-1, N, -1)
            elif T_or_N == N:
                # already (B, N, d)
                pass
            else:
                raise ValueError(
                    f"Unexpected shape for guidance tensor: {x.shape}, "
                    f"expected second dim 1 or N={N}"
                )
        else:
            raise ValueError(f"Guidance tensor must be 2D or 3D, got {x.dim()}D")

        return x

    def forward(self, z: torch.Tensor, F_g: torch.Tensor, G: torch.Tensor):
        """
        z   : (B, N, z_dim)   from GAC
        F_g : global feature at current timestep
        G   : sub-goal feature per region / node

        Returns:
            logits : (B, N, num_actions)
            values : (B, N)
        """
        B, N, z_dim = z.shape
        assert z_dim == self.z_dim, f"Expected z_dim={self.z_dim}, got {z_dim}"

        # Broadcast F_g and G over intersections if needed
        Fg_tiled = self._tile_or_broadcast(F_g, N)  # (B, N, d_reg)
        G_tiled  = self._tile_or_broadcast(G, N)    # (B, N, d_reg)

        # Concatenate [z_i, F_g, g_i]
        feat = torch.cat([z, Fg_tiled, G_tiled], dim=-1)  # (B, N, 288)

        logits = self.actor(feat)          # (B, N, num_actions)
        values = self.critic(feat).squeeze(-1)  # (B, N)

        return logits, values

    @torch.no_grad()
    def act(self, z: torch.Tensor, F_g: torch.Tensor, G: torch.Tensor):
        """
        Convenience method to sample actions + return value + log_prob for A2C/PPO style updates.
        """
        logits, values = self.forward(z, F_g, G)   # (B, N, A), (B, N)
        dist = torch.distributions.Categorical(logits=logits)

        actions   = dist.sample()                 # (B, N)
        log_probs = dist.log_prob(actions)        # (B, N)

        return actions, log_probs, values
