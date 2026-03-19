"""
malmo_boat_env_ppo.py

Key fixes vs previous version:
  - Removed pure-spin actions (steer only without throttle).
    On frictionless ice these spin the boat forever at zero cost.
    Steering now ONLY applies when moving forward, matching the old
    working PPO env which explicitly blocked steering without throttle.
  - Reward redesigned around a clear two-stage behavior:
      Stage 1: align toward checkpoint (yaw penalty until facing it)
      Stage 2: move forward once aligned (speed reward when abs_rel < 45deg)
    The -1000 lava penalty was replaced with -10 to avoid dominating
    the value function during early training.
  - Yaw math verified correct: atan2(dx_t, dz_t) for target bearing,
    atan2(-sin(yaw_rad), cos(yaw_rad)) for facing, both in MC space.
    rel is computed against ONLY the next checkpoint, not all of them.
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

    # Steering is ONLY applied when throttle != 0 (moving forward or back).
    # This prevents the agent from spinning in place on frictionless ice.
    # MultiDiscrete([2, 3]): throttle=[stop, forward], steering=[none, left, right]
    # "back" removed — boats on ice almost never need to reverse and it
    # creates confusing gradients when the agent goes backward toward a checkpoint.
    THROTTLE_MAP = {0: None,      1: "forward"}
    STEERING_MAP = {0: None,      1: "left",   2: "right"}

    def __init__(self,
                 mission_xml_path="boat_mission.xml",
                 millisec_per_tick=20,
                 num_tracks=3):

        super(MalmoBoatEnv, self).__init__()

        self.millisec_per_tick = millisec_per_tick
        self.num_tracks_cfg    = num_tracks
        self.xml_template      = Path(mission_xml_path).read_text()

        # [throttle(0-1), steering(0-2)]
        # Steering is silently ignored if throttle==0, so the effective
        # actions are: coast, forward, forward+left, forward+right
        self.action_space      = spaces.MultiDiscrete([2, 3])
        # 10 base + 4 directional edge sensors = 14
        # Edge sensors: lava within 2 blocks in front/right/left/behind
        # relative to agent facing direction (not world axes)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32
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
        self.last_obs                      = np.zeros(14, dtype=np.float32)
        self.checkpoint_threshold          = 5.0
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
        throttle_idx = int(action[0])

        # Coasting step waits longer so the boat actually decelerates.
        # Ice has near-zero friction so a short wait does nothing —
        # the boat keeps full speed into the next action.
        # Doubling the wait when not pressing forward gives physics
        # time to shed momentum, making "do nothing" a real brake.
        if throttle_idx == 0:
            time.sleep(TICK_LENGTH * 12)   # coast = longer wait = more decel
        else:
            time.sleep(TICK_LENGTH * 6)    # moving = normal cadence

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
        self.aligned_steps    = 0    # consecutive steps spent facing checkpoint
        self.steps_since_checkpoint = 0

    def _tp_spawn_and_boat(self):
        spawn_x, spawn_z = self.spawn_point
        

        for key in ["forward", "back", "left", "right"]:
            self.agent_host.sendCommand(f"{key} 0")
        time.sleep(TICK_LENGTH * 10)

        self.agent_host.sendCommand(f"tp {spawn_x} 230 {spawn_z}")
        time.sleep(TICK_LENGTH * 20)

        # Look Down
        self.agent_host.sendCommand("moveMouse 0 -1000")
        time.sleep(TICK_LENGTH * 5)
        # 3. Summon boat at spawn level
        self.agent_host.sendCommand(
            f"chat /summon minecraft:boat {spawn_x} 227 {spawn_z}"
        )
        time.sleep(TICK_LENGTH * 20)
        
        # 4. TP directly onto the boat — same coords, triggers mount hitbox
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
        throttle_idx, steering_idx = int(action[0]), int(action[1])
        for key in ["forward", "back", "left", "right"]:
            self.agent_host.sendCommand(f"{key} 0")

        throttle = self.THROTTLE_MAP[throttle_idx]
        if throttle:
            self.agent_host.sendCommand(f"{throttle} 1")
            # Only steer if actually moving — prevents frictionless ice spinning
            steering = self.STEERING_MAP[steering_idx]
            if steering:
                self.agent_host.sendCommand(f"{steering} 1")

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
        # Get steering index from your MultiDiscrete action [throttle, steering]
        # 0: None, 1: Left, 2: Right

        steering_idx = int(action[1])
        throttle_idx = int(action[0])

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
        # Base alignment: smooth 0..1 signal, always positive. Based on speed constraints as well.
        alignment = (math.pi - abs_rel) / math.pi # 0 ... 1
        if speed > 0.1:
            reward += alignment * 1.5
        else:
            reward += alignment * 0.2 # Enough to teach orientation but not farm.

        # Dwell time: track consecutive steps spent well-aligned (<30 deg).
        # Each step holding alignment earns a growing bonus — this directly
        # rewards the agent for holding its nose pointed at the checkpoint
        # rather than constantly overcorrecting.
        # When near an edge and the track curves, the agent learns to stop
        # and hold alignment rather than charging blindly forward.
        ALIGNED_THRESHOLD = math.radians(30)   # within 30 deg = "aligned" and must be moving at least a little.
        if abs_rel < ALIGNED_THRESHOLD and speed > 0.1:
            self.aligned_steps += 1
            # Bonus grows with dwell time, capped at 10 steps worth
            # so it doesn't completely dominate the reward at long holds
            dwell_bonus = min(self.aligned_steps, 10) * 0.1
            reward += dwell_bonus
        else:
            # Reset counter — broke alignment or too slow.
            self.aligned_steps = 0
        

        # --- Steering correction: outcome-based, not action-based ---
        # Reward/penalize based on whether abs_rel actually improved this step.
        # This teaches counter-steering on ice without hardcoding which button
        # to press — the agent learns "whatever I just did made it better/worse".
        #
        # Only active when misaligned (> 15 deg) so it doesn't penalize tiny
        # corrections when already well-pointed.
        if self.prev_abs_rel is not None and abs_rel > math.radians(15):
            improvement = self.prev_abs_rel - abs_rel  # positive = got better
            if improvement > 0:
                # Getting more aligned — reward scales with how fast
                reward += improvement * 3.0
            else:
                # Getting less aligned — spinning out or overcorrecting
                # Scale penalty harder the more misaligned we already are
                # so spinouts at 180 deg are punished more than small wobbles
                misalign_factor = abs_rel / math.pi  # 0..1
                reward += improvement * 3.0 * (1.0 + misalign_factor)

        self.prev_abs_rel = abs_rel

        
        
        # --- Stage 2: forward progress only when aligned ---
        # Distance shaping and speed reward only fire when facing within 45 deg.
        # Agent must earn the right to go fast by aligning first.
        #
        # Edge penalty: if lava is within 2 blocks ahead, penalize speed.
        # This teaches the agent to slow down before the edge rather than
        # charging into lava. The sensor is already in the observation so
        # the network can also learn to brake preemptively.
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
        # Reduce speed reward when front edge is detected
        speed_scale = max(0.0, 1.0 - look_ahead_severity)
        reward += speed * speed_scale

        # Also penalize for moving fast toward an edge
        if front_edge and speed > 0.3:
            reward -= speed  * look_ahead_severity * 5.0   # stronger than the speed reward above

        if self.prev_dist is not None:
            delta = self.prev_dist - dist
            reward += delta * 10.0

        self.prev_dist = dist

        # --- Checkpoint bonus ---
        if dist < self.checkpoint_threshold:
            reward += 100.0
            self.steps_since_checkpoint = 0  # reset on progress
            self.current_target_checkpoint_idx += 1
            self.aligned_steps = 0   # reset dwell on new target
            print(f"Checkpoint {self.current_target_checkpoint_idx} reached!")
            if self.current_target_checkpoint_idx >= self.num_check_points:
                reward += 200.0
                print("Lap complete!")
        # 1. Get the current yaw (normalized to -180 to 180)
        current_yaw = (obs.get('Yaw', 0.0) + 180) % 360 - 180

        # 2. Calculate the target yaw from spawn/current pos to checkpoint
        # dz is longitudinal (North/South), dx is lateral (East/West)
        target_rad = math.atan2(dx_t, dz_t)
        target_yaw = -math.degrees(target_rad)

        # 3. Calculate the degrees off-target
        # If this is 0.0, you are staring perfectly at the checkpoint.
        # If this is positive, the checkpoint is to your LEFT.
        # If this is negative, the checkpoint is to your RIGHT.
        error_deg = (target_yaw - current_yaw + 180) % 360 - 180

        # --- PRINT MONITOR ---
        if self.steps % 10 == 0:
            t_msg = self.THROTTLE_MAP.get(throttle_idx, "stop")
            s_msg = self.STEERING_MAP.get(steering_idx, "none")
            print(f"STEP: {self.steps} | ACTION: [{t_msg}, {s_msg}]")
            print(f"DEBUG | Boat Yaw: {current_yaw:6.1f}° | Target: {target_yaw:6.1f}° | ERROR: {error_deg:6.1f}° | Print: Reward: {reward:.4f}")

        return reward

    def _get_look_ahead_danger(self, raw, yaw, distance=5):
        """
        Checks if there is lava/edge anywhere from 1 to 'distance' blocks ahead.
        Returns a float: 0.0 (safe) to 1.0 (danger detected).
        """
        grid = raw.get('nearby_blocks', [])
        if len(grid) < 147: return 0.0

        ICE = {'packed_ice', 'ice', 'minecraft:packed_ice'}
        
        # Map yaw to world direction unit vectors
        # MC: 0=S(+Z), 90=W(-X), 180=N(-Z), 270=E(+X)
        yaw_rad = math.radians(yaw)
        # We calculate the vector the boat is facing
        dir_x = -math.sin(yaw_rad)
        dir_z = math.cos(yaw_rad)

        # Check multiple points along the line ahead
        for d in range(1, distance + 1):
            # Calculate grid coordinates for distance 'd'
            check_x = int(round(dir_x * d))
            check_z = int(round(dir_z * d))
            
            # Grid indexing (from your _get_edge_sensors logic)
            ix = check_x + 3
            iz = check_z + 3
            
            # Ensure we are within the 7x7 grid bounds (-3 to +3)
            if 0 <= ix < 7 and 0 <= iz < 7:
                idx = 0 * 49 + iz * 7 + ix # iy=0 is floor
                if grid[idx] not in ICE:
                    # Return a value based on how close the danger is
                    # (Closer danger = higher value)
                    return (distance - d + 1) / distance
                    
        return 0.0

    def _get_edge_sensors(self, raw, yaw):
        """
        Returns 4 binary edge signals relative to the agent's facing direction:
          [front_edge, right_edge, left_edge, behind_edge]
        1.0 = lava within 2 blocks in that direction, 0.0 = safe.

        Uses the nearby_blocks grid (7x3x7, y-1 to y+1, centered on agent).
        We check y=-1 (floor level) because lava floor is at y=225
        and ice track is at y=226. A missing floor block = lava = edge.

        Grid indexing: index = iy*49 + iz*7 + ix
          where ix=dx+3, iy=dy+1, iz=dz+3
        """
        grid = raw.get('nearby_blocks', [])
        if len(grid) < 147:
            return [0.0, 0.0, 0.0, 0.0]

        LAVA = {'lava', 'flowing_lava', 'minecraft:lava'}
        ICE  = {'packed_ice', 'ice', 'minecraft:packed_ice'}

        def is_edge(dx, dz):
            """True if floor at (dx, dz) relative to agent is lava (not ice)."""
            ix  = dx + 3
            iz  = dz + 3
            idx = 1 * 49 + iz * 7 + ix   # iy=1 means dy=-1+1=0... wait
            # iy: dy=-1 → iy=0, dy=0 → iy=1, dy=1 → iy=2
            iy  = 0   # dy = -1 = one below agent = floor
            idx = iy * 49 + iz * 7 + ix
            if idx >= len(grid):
                return False
            block = grid[idx]
            # Edge = not ice (lava or air gap)
            return block not in ICE

        # Check a 2-block cone in each world direction
        # Then rotate to agent-relative using yaw
        # World directions: +Z=south, -Z=north, +X=east, -X=west
        def cone_edge(world_dx_range, world_dz_range):
            return any(
                is_edge(dx, dz)
                for dx in world_dx_range
                for dz in world_dz_range
            )

        # Agent facing direction from yaw (MC: 0=south, 90=west)
        # Compute which world direction is "in front"
        yaw_mod = yaw % 360

        # Quantize to 4 cardinal directions for sensor mapping
        # front/right/left/behind relative to agent heading
        if 315 <= yaw_mod or yaw_mod < 45:      # facing south (+Z)
            front  = cone_edge(range(-1,2), range(1,3))
            behind = cone_edge(range(-1,2), range(-3,-1))
            right  = cone_edge(range(1,3),  range(-1,2))
            left   = cone_edge(range(-3,-1),range(-1,2))
        elif 45 <= yaw_mod < 135:                # facing west (-X)
            front  = cone_edge(range(-3,-1),range(-1,2))
            behind = cone_edge(range(1,3),  range(-1,2))
            right  = cone_edge(range(-1,2), range(1,3))
            left   = cone_edge(range(-1,2), range(-3,-1))
        elif 135 <= yaw_mod < 225:               # facing north (-Z)
            front  = cone_edge(range(-1,2), range(-3,-1))
            behind = cone_edge(range(-1,2), range(1,3))
            right  = cone_edge(range(-3,-1),range(-1,2))
            left   = cone_edge(range(1,3),  range(-1,2))
        else:                                    # facing east (+X) 225-315
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

            # Prefer boat entity yaw over agent yaw when riding
            for entity in raw.get('entities', []):
                if entity.get('name') == 'Boat':
                    yaw = entity.get('yaw', yaw)
                    break

            obs_vals = []

            # Next 3 checkpoints relative position
            for i in range(3):
                idx = self.current_target_checkpoint_idx + i
                if idx < len(self.checkpoints):
                    tx, tz = self.checkpoints[idx]
                    obs_vals.extend([tx - x, tz - z])
                else:
                    obs_vals.extend([0.0, 0.0])

            obs_vals.extend([vx, vz])

            # Relative angle to NEXT checkpoint only as cos/sin
            # cos=1 → perfectly aligned, cos=-1 → facing away
            # sin>0 → target to right,   sin<0 → target to left
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

            # Edge sensors: lava within 2 blocks in front/right/left/behind
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
            obs_vals.extend([0.0, 0.0, 1.0, 0.0])
            obs_vals.extend([0.0, 0.0, 0.0, 0.0])  # edge sensors default safe
            return np.array(obs_vals, dtype=np.float32)

        return np.zeros(14, dtype=np.float32)