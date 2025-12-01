from common.registry import Registry
from .base import BaseAgent
import numpy as np


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
        vehicle_speed = eng.get_vehicle_speed()  
        vehicle_wait_time = world.get_vehicle_waiting_time()

        # flow: number of vehicles passing through per unit time
        lane_flow = world.info_function["throughput"]() 
        
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
                    veh_info = eng.get_vehicle_info()            # dict: vid -> info
                    distances = [veh_info[vid]["distance"] for vid in queue_vids]
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
                #    # of vehicles with speed < 0.1 m/s, normalized by cap
                stop_count = sum(1 for s in speeds if s < 0.1)
                stop_car_num = float(stop_count) / float(cap)

                # 6) waiting_time:
                #    waiting time of the FIRST vehicle on each incoming lane.
                #    If we know per-vehicle waiting time, use the first vehicle
                #    in v_list; otherwise fall back to env's lane-level metric.
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
                #    We assume world.get_lane_pressure already implements this
                #    according to your simulator definition.
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
