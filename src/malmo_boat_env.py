"""
malmo_boat_env_ppo.py
"""

import json
import math
import time
from pathlib import Path

import gym
import MalmoPython
import numpy as np
from gym import spaces

from ice_track_testing import generate_tracks, RESET_BLOCK_TYPE

CLIENT_PORT = 10000
TICK_LENGTH = 0.05
TIME_WAIT   = 0.05


class MalmoBoatEnv(gym.Env):

    # Discrete(4) phased action space.
    # Each action plays out over 2 ticks — steer tick then throttle tick.
    # This prevents simultaneous steer+throttle spinouts on frictionless ice.
    #
    #   0 = steer left  → then forward
    #   1 = steer right → then forward
    #   2 = forward only (already aligned, just go)
    #   3 = coast (release all, let boat decelerate)
    ACTION_PHASES = {
        0: [('left',    1), ('forward', 1)],
        1: [('right',   1), ('forward', 1)],
        2: [('forward', 1), ('forward', 1)],
        3: [(None,      0), (None,      0)],
    }
    ACTION_LABELS = {0: 'left+fwd', 1: 'right+fwd', 2: 'forward', 3: 'coast'}

    def __init__(self,
                 mission_xml_path="boat_mission.xml",
                 millisec_per_tick=20,
                 num_tracks=3):

        super(MalmoBoatEnv, self).__init__()

        self.millisec_per_tick = millisec_per_tick
        self.num_tracks_cfg    = num_tracks
        self.xml_template      = Path(mission_xml_path).read_text()

        self.action_space      = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32
        )

        self.agent_host     = MalmoPython.AgentHost()
        self.mission_record = MalmoPython.MissionRecordSpec()
        self.mission_record.recordRewards()
        self.mission_record.recordObservations()
        self.pool = MalmoPython.ClientPool()
        self.pool.add(MalmoPython.ClientInfo('127.0.0.1', CLIENT_PORT))

        self._generate_new_tracks()

        self.current_track_idx         = 0
        self.episodes_on_current_track = 0
        self.episodes_per_track        = 2
        self._mission_running          = False
        self._mission_needs_restart    = True

        self.checkpoints                   = []
        self.spawn_point                   = None
        self.num_check_points              = 0
        self.current_target_checkpoint_idx = 0
        self.prev_dist                     = None
        self.prev_abs_rel                  = None
        self.last_raw_obs                  = None
        self.last_obs                      = np.zeros(15, dtype=np.float32)
        self.checkpoint_threshold          = 7.0
        self.steps = 0

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def reset(self):
        if self.episodes_on_current_track >= self.episodes_per_track:
            self.episodes_on_current_track = 0
            self.current_track_idx = (self.current_track_idx + 1) % self.num_tracks
            print(f"Switching to track {self.current_track_idx}")

        if self._mission_needs_restart:
            return self._full_reset()
        return self._quick_respawn()

    def step(self, action):
        self.steps += 1
        self._send_action(action)

        # Extra wait for coast so deceleration has time to happen on ice
        if int(action) == 3:
            time.sleep(TICK_LENGTH * 12)

        for key in ["forward", "back", "left", "right"]:
            self.agent_host.sendCommand(f"{key} 0")

        world_state = self.agent_host.getWorldState()
        obs    = self._get_observation(world_state)
        reward = self._compute_reward(action)
        done   = self._check_done(world_state)

        return obs, reward, done, {
            'checkpoint':        self.current_target_checkpoint_idx,
            'total_checkpoints': self.num_check_points,
            'track_idx':         self.current_track_idx,
        }

    def close(self):
        if self._mission_running:
            try:
                self.agent_host.sendCommand("quit")
            except Exception:
                pass
        self._mission_running = False
        print("Env closed.")

    # -------------------------------------------------------------------------
    # Track generation
    # -------------------------------------------------------------------------

    def _generate_new_tracks(self):
        data             = generate_tracks(num_tracks=self.num_tracks_cfg)
        self.track_xml   = data['draw_xml']
        self.tracks_data = data['tracks']
        self.num_tracks  = data['num_tracks']
        print(f"Generated {self.num_tracks} tracks.")

    # -------------------------------------------------------------------------
    # Reset helpers
    # -------------------------------------------------------------------------

    def _full_reset(self):
        print("Starting mission (full reset)...")
        self._init_episode_state()
        spawn_x, spawn_z = self.spawn_point

        xml = (self.xml_template
               .replace('{PLACEHOLDER_MSPERTICK}', str(self.millisec_per_tick))
               .replace('{PLACEHOLDER_TRACK_XML}',  self.track_xml)
               .replace('{PLACEHOLDER_SPAWN_X}',    str(spawn_x))
               .replace('{PLACEHOLDER_SPAWN_Z}',    str(spawn_z)))
        mission = MalmoPython.MissionSpec(xml, False)

        for retry in range(5):
            try:
                self.agent_host.startMission(
                    mission, self.pool, self.mission_record, 0, 'boat_racer'
                )
                break
            except RuntimeError as e:
                print(f"Retry {retry+1}/5: {e}")
                time.sleep(5 * (retry + 1))
        else:
            raise RuntimeError("Could not start mission after 5 attempts")

        world_state = self.agent_host.getWorldState()
        while not world_state.has_mission_begun:
            time.sleep(TIME_WAIT * self.millisec_per_tick / 20)
            world_state = self.agent_host.getWorldState()

        self._mission_running       = True
        self._mission_needs_restart = False
        self.episodes_on_current_track += 1

        self.agent_host.sendCommand("chat /gamerule doFireTick false")
        self.agent_host.sendCommand("chat /gamerule fireDamage false")
        time.sleep(0.5)

        print("Waiting for world to build...")
        time.sleep(15)
        self._tp_spawn_and_boat()
        return self._get_observation()

    def _quick_respawn(self):
        print(f"Quick respawn on track {self.current_track_idx}")
        self._init_episode_state()
        self._tp_spawn_and_boat()
        self.episodes_on_current_track += 1
        return self._get_observation()

    def _init_episode_state(self):
        track            = self.tracks_data[self.current_track_idx]
        self.spawn_point = tuple(track['spawn_point'])
        self.checkpoints = [tuple(cp) for cp in track['checkpoints']]
        self.checkpoints.append(self.checkpoints[0])
        self.num_check_points              = len(self.checkpoints)
        self.current_target_checkpoint_idx = 1
        self.prev_dist        = None
        self.prev_abs_rel     = None
        self.aligned_steps    = 0
        self.steps_since_checkpoint = 0
        self.prev_yaw = 0
        self.prev_x = None
        self.prev_z = None

    def _tp_spawn_and_boat(self):
        spawn_x, spawn_z = self.spawn_point
        

        for key in ["forward", "back", "left", "right"]:
            self.agent_host.sendCommand(f"{key} 0")
        time.sleep(TICK_LENGTH * 10)

        # Kill all leftover boats.
        self.agent_host.sendCommand("chat /kill @e[type=minecraft:boat]")
        time.sleep(TICK_LENGTH * 20)

        self.agent_host.sendCommand(f"tp {spawn_x} 230 {spawn_z}")
        time.sleep(TICK_LENGTH * 20)

        # Look Down.
        self.agent_host.sendCommand("moveMouse 0 -1000")
        time.sleep(TICK_LENGTH * 5)
        # Summon boat at spawn level
        self.agent_host.sendCommand(
            f"chat /summon minecraft:boat {spawn_x} 227 {spawn_z}"
        )
        time.sleep(TICK_LENGTH * 20)
        
        # TP directly onto the boat — same coords, triggers mount hitbox
        self.agent_host.sendCommand(f"tp {spawn_x} 227 {spawn_z}")

        self.agent_host.sendCommand("use 1")
        time.sleep(TICK_LENGTH * 5)
        self.agent_host.sendCommand("use 0")
        time.sleep(TICK_LENGTH * 5)

        self.agent_host.sendCommand("moveMouse 0 600")
        time.sleep(TICK_LENGTH * 5)

    # -------------------------------------------------------------------------
    # Step helpers
    # -------------------------------------------------------------------------

    def _send_action(self, action):
        action_idx = int(action)
        phases = self.ACTION_PHASES[action_idx]

        for (key, val) in phases:
            # Clear all inputs before each tick
            for k in ["forward", "back", "left", "right"]:
                self.agent_host.sendCommand(f"{k} 0")
            if key is not None:
                self.agent_host.sendCommand(f"{key} {val}")
            time.sleep(TICK_LENGTH * 6)

        # Release everything after both ticks
        for k in ["forward", "back", "left", "right"]:
            self.agent_host.sendCommand(f"{k} 0")

    def _check_done(self, world_state):
        if self.current_target_checkpoint_idx >= self.num_check_points:
            print("All checkpoints reached!")
            return True
        if self.last_raw_obs is not None:
            if self.last_raw_obs.get('YPos', 227) < 226.5:
                print("Lava detected -- quick respawn")
                return True
        if not world_state.is_mission_running:
            print("Mission stopped -- full restart")
            self._mission_needs_restart = True
            self._mission_running       = False
            return True
        self.steps_since_checkpoint += 1
        if self.steps_since_checkpoint >= 200:
            print("Timeout -- no checkpoint reached in 200 steps")
            return True
        return False
    
    def _compute_reward(self, action):
        if self.last_raw_obs is None:
            return 0.0

        obs   = self.last_raw_obs
        x     = obs.get('XPos', 0)
        y     = obs.get('YPos', 227)
        z     = obs.get('ZPos', 0)
        vx    = obs.get('XVel', 0.0)
        vz    = obs.get('ZVel', 0.0)
        yaw   = obs.get('Yaw',  0.0)
        speed = math.sqrt(vx**2 + vz**2)

        if y < 226.5:
            return -10.0

        if self.current_target_checkpoint_idx >= len(self.checkpoints):
            return 0.0

        # Angle to NEXT CHECKPOINT ONLY — not all checkpoints
        tx, tz   = self.checkpoints[self.current_target_checkpoint_idx]
        dx_t     = tx - x
        dz_t     = tz - z
        dist     = math.sqrt(dx_t**2 + dz_t**2)

        # Verified-correct Minecraft yaw → relative angle
        # yaw=0=south(+Z), yaw=90=west(-X), yaw=-90=east(+X)
        # positive rel = target is to the RIGHT, negative = to the LEFT
        yaw_rad  = math.radians(yaw)
        target_a = math.atan2(dx_t, dz_t)
        facing_a = math.atan2(-math.sin(yaw_rad), math.cos(yaw_rad))
        rel      = math.atan2(math.sin(target_a - facing_a),
                              math.cos(target_a - facing_a))
        abs_rel  = abs(rel)

        reward = 0.0
        idle_pressure = (self.steps_since_checkpoint / 200) ** 2
        reward -= idle_pressure * 1.0

        # --- Stage 1: alignment reward with dwell time bonus ---
        alignment = (math.pi - abs_rel) / math.pi
        if speed > 0.1:
            reward += alignment * 1.5
        else:
            reward += alignment * 0.2

        ALIGNED_THRESHOLD = math.radians(30)
        if abs_rel < ALIGNED_THRESHOLD and speed > 0.1:
            self.aligned_steps += 1
            dwell_bonus = min(self.aligned_steps, 10) * 0.1
            reward += dwell_bonus
        else:
            self.aligned_steps = 0

        # --- Steering correction: outcome-based, not action-based ---
        if self.prev_abs_rel is not None and abs_rel > math.radians(15):
            improvement = self.prev_abs_rel - abs_rel
            if improvement > 0:
                reward += improvement * 1.5
            else:
                misalign_factor = abs_rel / math.pi
                reward += improvement * 1.5 * (1.0 + misalign_factor)

        self.prev_abs_rel = abs_rel

        # --- Angular velocity penalty ---
        if self.prev_yaw is not None:
            dyaw = yaw - self.prev_yaw
            dyaw = (dyaw + 180) % 360 - 180
            ang_vel_deg = abs(dyaw)
            if ang_vel_deg > 15 and abs_rel > math.radians(20):
                reward -= (ang_vel_deg - 15) * 0.05

        # --- Edge danger ---
        edge_sensors = self._get_edge_sensors(
            self.last_raw_obs,
            self.last_raw_obs.get('Yaw', 0.0)
        )
        front_edge = edge_sensors[0]
        look_ahead_severity = self._get_look_ahead_danger(
            self.last_raw_obs,
            self.last_raw_obs.get('Yaw', 0.0),
            distance=6
        )
        speed_scale = max(0.0, 1.0 - look_ahead_severity)
        reward += speed * speed_scale

        if front_edge and speed > 0.3:
            reward -= speed * look_ahead_severity * 3.0

        if self.prev_dist is not None:
            delta = self.prev_dist - dist
            reward += delta * 10.0

        self.prev_dist = dist

        # --- Checkpoint bonus ---
        # Check using both current AND previous position to catch high-speed passes
        def dist_to_checkpoint(cx, cz):
            return math.sqrt((cx - x)**2 + (cz - z)**2)

        def passed_through(cx, cz):
            """True if the line from prev_pos to cur_pos passed within threshold of checkpoint."""
            if self.prev_x is None:
                return False
            # Vector from prev to current position
            pdx = x - self.prev_x
            pdz = z - self.prev_z
            # Vector from prev to checkpoint
            cdx = cx - self.prev_x
            cdz = cz - self.prev_z
            seg_len_sq = pdx**2 + pdz**2
            if seg_len_sq < 1e-6:
                return False
            t = max(0.0, min(1.0, (cdx*pdx + cdz*pdz) / seg_len_sq))
            closest_x = self.prev_x + t * pdx
            closest_z = self.prev_z + t * pdz
            return math.sqrt((cx - closest_x)**2 + (cz - closest_z)**2) < self.checkpoint_threshold

        def check_checkpoint(idx, bonus):
            if idx >= len(self.checkpoints):
                return False
            cx, cz = self.checkpoints[idx]
            if dist_to_checkpoint(cx, cz) < self.checkpoint_threshold or passed_through(cx, cz):
                return True
            return False

        if check_checkpoint(self.current_target_checkpoint_idx, 100.0):
            reward += 100.0
            self.steps_since_checkpoint = 0
            self.prev_dist = None
            self.current_target_checkpoint_idx += 1
            self.aligned_steps = 0
            print(f"Checkpoint {self.current_target_checkpoint_idx} reached!")
            if self.current_target_checkpoint_idx >= self.num_check_points:
                reward += 200.0
                print("Lap complete!")
        elif check_checkpoint(self.current_target_checkpoint_idx + 1, 60.0):
            reward += 60.0
            self.steps_since_checkpoint = 0
            self.prev_dist = None
            self.current_target_checkpoint_idx += 2
            self.aligned_steps = 0
            print(f"Skipped to checkpoint {self.current_target_checkpoint_idx}!")
            if self.current_target_checkpoint_idx >= self.num_check_points:
                reward += 200.0
                print("Lap complete!")

        # --- PRINT MONITOR ---
        if self.steps % 10 == 0:
            current_yaw = (obs.get('Yaw', 0.0) + 180) % 360 - 180
            target_rad  = math.atan2(dx_t, dz_t)
            target_yaw  = -math.degrees(target_rad)
            error_deg   = (target_yaw - current_yaw + 180) % 360 - 180
            a_msg = self.ACTION_LABELS.get(int(action), '?')
            print(f"STEP: {self.steps} | ACTION: {a_msg}")
            print(f"DEBUG | Boat Yaw: {current_yaw:6.1f}° | Target: {target_yaw:6.1f}° | ERROR: {error_deg:6.1f}° | Reward: {reward:.4f}")

        self.prev_x = x
        self.prev_z = z
        return reward

    def _get_look_ahead_danger(self, raw, yaw, distance=5):
        """
        Checks if there is lava/edge anywhere from 1 to 'distance' blocks ahead.
        Returns a float: 0.0 (safe) to 1.0 (danger detected).
        """
        grid = raw.get('nearby_blocks', [])
        if len(grid) < 147: return 0.0

        ICE = {'packed_ice', 'ice', 'minecraft:packed_ice'}

        yaw_rad = math.radians(yaw)
        dir_x = -math.sin(yaw_rad)
        dir_z = math.cos(yaw_rad)

        for d in range(1, distance + 1):
            check_x = int(round(dir_x * d))
            check_z = int(round(dir_z * d))
            ix = check_x + 3
            iz = check_z + 3
            if 0 <= ix < 7 and 0 <= iz < 7:
                idx = 0 * 49 + iz * 7 + ix
                if grid[idx] not in ICE:
                    return (distance - d + 1) / distance

        return 0.0

    def _get_edge_sensors(self, raw, yaw):
        """
        Returns 4 binary edge signals relative to the agent's facing direction:
          [front_edge, right_edge, left_edge, behind_edge]
        1.0 = lava within 2 blocks in that direction, 0.0 = safe.
        """
        grid = raw.get('nearby_blocks', [])
        if len(grid) < 147:
            return [0.0, 0.0, 0.0, 0.0]

        ICE = {'packed_ice', 'ice', 'minecraft:packed_ice'}

        def is_edge(dx, dz):
            ix  = dx + 3
            iz  = dz + 3
            iy  = 0
            idx = iy * 49 + iz * 7 + ix
            if idx >= len(grid):
                return False
            return grid[idx] not in ICE

        def cone_edge(world_dx_range, world_dz_range):
            return any(
                is_edge(dx, dz)
                for dx in world_dx_range
                for dz in world_dz_range
            )

        yaw_mod = yaw % 360

        if 315 <= yaw_mod or yaw_mod < 45:
            front  = cone_edge(range(-1,2), range(1,3))
            behind = cone_edge(range(-1,2), range(-3,-1))
            right  = cone_edge(range(1,3),  range(-1,2))
            left   = cone_edge(range(-3,-1),range(-1,2))
        elif 45 <= yaw_mod < 135:
            front  = cone_edge(range(-3,-1),range(-1,2))
            behind = cone_edge(range(1,3),  range(-1,2))
            right  = cone_edge(range(-1,2), range(1,3))
            left   = cone_edge(range(-1,2), range(-3,-1))
        elif 135 <= yaw_mod < 225:
            front  = cone_edge(range(-1,2), range(-3,-1))
            behind = cone_edge(range(-1,2), range(1,3))
            right  = cone_edge(range(-3,-1),range(-1,2))
            left   = cone_edge(range(1,3),  range(-1,2))
        else:
            front  = cone_edge(range(1,3),  range(-1,2))
            behind = cone_edge(range(-3,-1),range(-1,2))
            right  = cone_edge(range(-1,2), range(-3,-1))
            left   = cone_edge(range(-1,2), range(1,3))

        return [float(front), float(right), float(left), float(behind)]

    def _get_observation(self, world_state=None):
        if world_state is None:
            world_state = self.agent_host.getWorldState()

        if world_state.number_of_observations_since_last_state > 0:
            raw = json.loads(world_state.observations[-1].text)
            self.last_raw_obs = raw

            x   = raw.get('XPos', 0.0)
            z   = raw.get('ZPos', 0.0)
            vx  = raw.get('XVel', 0.0)
            vz  = raw.get('ZVel', 0.0)
            yaw = raw.get('Yaw',  0.0)

            for entity in raw.get('entities', []):
                if entity.get('name') == 'Boat':
                    yaw = entity.get('yaw', yaw)
                    break

            obs_vals = []

            for i in range(3):
                idx = self.current_target_checkpoint_idx + i
                if idx < len(self.checkpoints):
                    tx, tz = self.checkpoints[idx]
                    obs_vals.extend([tx - x, tz - z])
                else:
                    obs_vals.extend([0.0, 0.0])

            obs_vals.extend([vx, vz])

            if self.prev_yaw is not None:
                dyaw = yaw - self.prev_yaw
                dyaw = (dyaw + 180) % 360 - 180
                ang_vel = dyaw / 180.0
            else:
                ang_vel = 0.0
            self.prev_yaw = yaw
            obs_vals.append(ang_vel)

            if self.current_target_checkpoint_idx < len(self.checkpoints):
                tx, tz   = self.checkpoints[self.current_target_checkpoint_idx]
                dx_t     = tx - x
                dz_t     = tz - z
                yaw_rad  = math.radians(yaw)
                target_a = math.atan2(dx_t, dz_t)
                facing_a = math.atan2(-math.sin(yaw_rad), math.cos(yaw_rad))
                rel      = math.atan2(math.sin(target_a - facing_a),
                                      math.cos(target_a - facing_a))
                obs_vals.extend([math.cos(rel), math.sin(rel)])
            else:
                obs_vals.extend([1.0, 0.0])

            edge = self._get_edge_sensors(raw, yaw)
            obs_vals.extend(edge)

            self.last_obs = np.array(obs_vals, dtype=np.float32)
            return self.last_obs

        if self.last_raw_obs is not None:
            x = self.last_raw_obs.get('XPos', 0)
            z = self.last_raw_obs.get('ZPos', 0)
            obs_vals = []
            for i in range(3):
                idx = self.current_target_checkpoint_idx + i
                if idx < len(self.checkpoints):
                    tx, tz = self.checkpoints[idx]
                    obs_vals.extend([tx - x, tz - z])
                else:
                    obs_vals.extend([0.0, 0.0])
            obs_vals.extend([0.0, 0.0])
            obs_vals.append(0.0)
            obs_vals.extend([1.0, 0.0])
            obs_vals.extend([0.0, 0.0, 0.0, 0.0])
            return np.array(obs_vals, dtype=np.float32)

        return np.zeros(15, dtype=np.float32)