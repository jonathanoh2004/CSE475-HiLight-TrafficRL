from common.registry import Registry
from .base import BaseAgent
import numpy as np
import torch
import torch.nn as nn
#from collections import OrderedDict

@Registry.register_model('hilight')
class HilightAgent(BaseAgent):
    def __init__(self, world, metric, flow_path="flow.json"):
        super().__init__(world)
        self.world = world
        self.eng = world.eng
        self.metric = metric

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

        self._build_inter_approach_lanes() #Added for GAC
        self._build_knn_neighbors(k=4)  #added for GAC
        # # --- regions & regional window for meta-policy ---
        # self.region_of_inter = self._build_region_mapping() # to be implemented
        # self.region_coords = self._compute_region_coords() # to be implemented
        # from collections import deque
        # self.regional_window = deque(maxlen=20)

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

    #Added for GAC
    def _build_knn_neighbors(self, k=4):
        #Build K-nearest-neighbor index for intersections.
        #Result is self.neighbor_index: LongTensor of shape (num_inters, k)

        num_inters = len(self.world.intersection_ids)
        coords = np.zeros((num_inters, 2), dtype=np.float32)

        # fill in center coords in the order of self.world.intersection_ids
        for idx, iid in enumerate(self.world.intersection_ids):
            x, y = self.inter_coord[iid]
            coords[idx] = [x, y]

        neighbor_index = np.zeros((num_inters, k), dtype=np.int64)
        for i in range(num_inters):
            # compute distance from node i to all others
            dx = coords[:, 0] - coords[i, 0]
            dy = coords[:, 1] - coords[i, 1]
            dist_sq = dx * dx + dy * dy

            # exclude self by setting a huge distance
            dist_sq[i] = np.inf

            # indices of k nearest neighbors
            knn = np.argsort(dist_sq)[:k]
            neighbor_index[i] = knn

        # store as torch LongTensor
        self.neighbor_index = torch.from_numpy(neighbor_index).long()


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