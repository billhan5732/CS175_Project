import MalmoPython
import json
import math
import os
import time

import gym
import numpy as np
from gym import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback

from ice_track_testing import RESET_BLOCK_TYPE

TICK_LENGTH = 0.05
TESTING     = False

# -----------------------------------------------------------------------------
# Set this to wherever you saved the world folder after running build_world.py
# Example: "C:/Users/coolb/CS175/CS175_Project/saved_worlds/ice_track"
# -----------------------------------------------------------------------------
SAVED_WORLD_PATH = "C:/Users/coolb/CS175/CS175_Project/saved_worlds/ice_track"
TRACK_DATA_PATH  = "track_data.json"

print("Imported successfully!")


def _load_track_data():
    """Load checkpoint/spawn data saved by build_world.py."""
    if not os.path.exists(TRACK_DATA_PATH):
        raise FileNotFoundError(
            f"'{TRACK_DATA_PATH}' not found — run build_world.py first."
        )
    with open(TRACK_DATA_PATH) as f:
        return json.load(f)


def _build_mission_xml(track_data, saved_world_path):
    """
    Mission XML that loads the pre-built saved world.
    No DrawBlocks at all — the track is already there.
    Mission start goes from ~50s down to ~3s.
    """
    spawn_x, spawn_z = track_data['tracks'][0]['spawn_point']
    world_path = saved_world_path.replace("\\", "/")

    return f'''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
    <Mission xmlns="http://ProjectMalmo.microsoft.com"
             xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
        <About>
            <Summary>Ice Boat Racing Training</Summary>
        </About>

        <ServerSection>
            <ServerInitialConditions>
                <Time>
                    <StartTime>6000</StartTime>
                    <AllowPassageOfTime>false</AllowPassageOfTime>
                </Time>
                <Weather>clear</Weather>
                <AllowSpawning>false</AllowSpawning>
            </ServerInitialConditions>
            <ServerHandlers>
                <!-- Pre-built world — zero DrawBlocks, instant load -->
                <FileWorldGenerator src="{world_path}" forceReset="false"/>
            </ServerHandlers>
        </ServerSection>

        <AgentSection mode="Creative">
            <n>IceBoatRacer</n>
            <AgentStart>
                <Placement x="{spawn_x}" y="227" z="{spawn_z}" pitch="0" yaw="0"/>
            </AgentStart>
            <AgentHandlers>
                <ObservationFromFullStats/>
                <ObservationFromNearbyEntities>
                    <Range name="entities" xrange="10" yrange="2" zrange="10"/>
                </ObservationFromNearbyEntities>
                <ObservationFromGrid>
                    <Grid name="nearby_blocks">
                        <min x="-3" y="-1" z="-3"/>
                        <max x="3" y="1" z="3"/>
                    </Grid>
                </ObservationFromGrid>
                <HumanLevelCommands/>
                <ChatCommands/>
                <AbsoluteMovementCommands>
                    <ModifierList type="allow-list">
                        <command>tp</command>
                        <command>setYaw</command>
                        <command>setPitch</command>
                    </ModifierList>
                </AbsoluteMovementCommands>
                <AgentQuitFromReachingCommandQuota total="0"/>
            </AgentHandlers>
        </AgentSection>
    </Mission>'''


class MalmoBoatEnv(gym.Env):
    """
    Malmo Ice Boat Racing — DQN.

    Prerequisites:
      - Run build_world.py once to place the track and save the Minecraft world
      - Set SAVED_WORLD_PATH above to where you saved it
      - track_data.json must exist in the same directory as this file
    """

    ACTION_MAP = {
        0: ("forward", None),
        1: ("forward", "left"),
        2: ("forward", "right"),
        3: (None,      "left"),
        4: (None,      "right"),
        5: (None,      None),
    }

    def __init__(self):
        super(MalmoBoatEnv, self).__init__()

        self.action_space = spaces.Discrete(len(self.ACTION_MAP))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32
        )

        self.agent_host = MalmoPython.AgentHost()

        raw              = _load_track_data()
        self.tracks_data = raw['tracks']
        self.num_tracks  = raw['num_tracks']
        self.mission_xml = _build_mission_xml(raw, SAVED_WORLD_PATH)

        print(f"Loaded {self.num_tracks} tracks from {TRACK_DATA_PATH}")
        print("Using FileWorldGenerator — no DrawBlocks, fast mission start.")

        self.current_track_idx         = 0
        self.episodes_on_current_track = 0
        self.episodes_per_track        = 2

        self._mission_running       = False
        self._mission_needs_restart = True

        self.current_target_checkpoint_idx = 0
        self.checkpoints      = []
        self.spawn_point      = None
        self.num_check_points = 0
        self.prev_dist        = None

        self.reset_block_type = RESET_BLOCK_TYPE

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
        self._send_action(action)
        time.sleep(TICK_LENGTH * 6)

        world_state = self.agent_host.getWorldState()
        obs    = self._parse_observation(world_state)
        reward = self._compute_reward(world_state)
        done   = self._check_done(world_state)

        info = {
            'checkpoint':        self.current_target_checkpoint_idx,
            'total_checkpoints': self.num_check_points,
            'track_idx':         self.current_track_idx,
        }

        return obs, reward, done, info

    def close(self):
        if self._mission_running:
            try:
                self.agent_host.sendCommand("quit")
                print("Quit sent, waiting for mission to end...")
                for _ in range(50):
                    time.sleep(0.1)
                    if not self.agent_host.getWorldState().is_mission_running:
                        break
            except Exception:
                pass
            self._mission_running = False
        print("Env closed.")

    # -------------------------------------------------------------------------
    # Reset helpers
    # -------------------------------------------------------------------------

    def _full_reset(self):
        print("Starting mission (loading saved world)...")
        mission        = MalmoPython.MissionSpec(self.mission_xml, True)
        mission_record = MalmoPython.MissionRecordSpec()

        for retry in range(5):
            try:
                self.agent_host.startMission(mission, mission_record)
                break
            except RuntimeError as e:
                print(f"Retry {retry + 1}/5: {e}")
                time.sleep(5 * (retry + 1))
        else:
            raise RuntimeError("Could not start mission after 5 attempts")

        print("Waiting for mission to begin...")
        start       = time.time()
        world_state = self.agent_host.getWorldState()
        while not world_state.has_mission_begun:
            if time.time() - start > 30:
                raise RuntimeError(
                    "Mission timed out — check SAVED_WORLD_PATH is correct"
                )
            time.sleep(0.1)
            world_state = self.agent_host.getWorldState()

        self._mission_running       = True
        self._mission_needs_restart = False

        self._init_episode_state()
        time.sleep(3)  # short wait — no block placement needed
        self._tp_spawn_and_boat()

        self.episodes_on_current_track += 1
        return self._parse_observation(self.agent_host.getWorldState())

    def _quick_respawn(self):
        print(f"Quick respawn on track {self.current_track_idx}")
        self._init_episode_state()
        self._tp_spawn_and_boat()
        self.episodes_on_current_track += 1
        return self._parse_observation(self.agent_host.getWorldState())

    def _init_episode_state(self):
        track            = self.tracks_data[self.current_track_idx]
        self.spawn_point = tuple(track['spawn_point'])
        self.checkpoints = [tuple(cp) for cp in track['checkpoints']]
        self.checkpoints.append(self.checkpoints[0])
        self.num_check_points              = len(self.checkpoints)
        self.current_target_checkpoint_idx = 1
        self.prev_dist                     = None

    def _tp_spawn_and_boat(self):
        spawn_x, spawn_z = self.spawn_point

        # Clear keys
        for key in ["forward", "back", "left", "right"]:
            self.agent_host.sendCommand(f"{key} 0")
        self.agent_host.sendCommand("use 0")
        time.sleep(TICK_LENGTH * 10)

        # TP high above spawn
        self.agent_host.sendCommand(f"tp {spawn_x} 235 {spawn_z}")
        time.sleep(TICK_LENGTH * 10)

        # Summon fresh boat
        self.agent_host.sendCommand(
            f"chat /summon minecraft:boat {spawn_x} 227 {spawn_z}"
        )
        time.sleep(TICK_LENGTH * 20)

        # TP to just above boat
        self.agent_host.sendCommand(f"tp {spawn_x} 228 {spawn_z}")
        time.sleep(TICK_LENGTH * 10)

        # Pitch down before dropping to boat level
        self.agent_host.sendCommand("setPitch 80")
        time.sleep(TICK_LENGTH * 10)

        # Drop to boat level
        self.agent_host.sendCommand(f"tp {spawn_x} 227.5 {spawn_z}")
        time.sleep(TICK_LENGTH * 10)

        # Mount with retry
        mounted = False
        for attempt in range(3):
            self.agent_host.sendCommand("use 1")
            time.sleep(TICK_LENGTH * 5)
            self.agent_host.sendCommand("use 0")
            time.sleep(TICK_LENGTH * 5)

            ws = self.agent_host.getWorldState()
            if ws.number_of_observations_since_last_state > 0:
                raw = json.loads(ws.observations[-1].text)
                if raw.get('IsRiding', False) or raw.get('YPos', 999) < 227.5:
                    print(f"Mounted on attempt {attempt + 1}")
                    mounted = True
                    break

            print(f"Mount attempt {attempt + 1} failed, retrying...")
            time.sleep(TICK_LENGTH * 5)

        if not mounted:
            print("Warning: could not confirm mount — continuing anyway")

    # -------------------------------------------------------------------------
    # Step helpers
    # -------------------------------------------------------------------------

    def _send_action(self, action):
        for key in ["forward", "back", "left", "right"]:
            self.agent_host.sendCommand(f"{key} 0")

        throttle, steering = self.ACTION_MAP[int(action)]
        if throttle:
            self.agent_host.sendCommand(f"{throttle} 1")
        if steering:
            self.agent_host.sendCommand(f"{steering} 1")

    def _parse_observation(self, world_state):
        if world_state.number_of_observations_since_last_state > 0:
            raw = json.loads(world_state.observations[-1].text)
            x   = raw.get('XPos', 0.0)
            z   = raw.get('ZPos', 0.0)
            vx  = raw.get('XVel', 0.0)
            vz  = raw.get('ZVel', 0.0)
            yaw = raw.get('Yaw',  0.0)

            obs_vals = []
            for i in range(3):
                idx = self.current_target_checkpoint_idx + i
                if idx < len(self.checkpoints):
                    tx, tz = self.checkpoints[idx]
                    obs_vals.extend([tx - x, tz - z])
                else:
                    obs_vals.extend([0.0, 0.0])
            obs_vals.extend([vx, vz, yaw])
            return np.array(obs_vals, dtype=np.float32)

        return np.zeros(9, dtype=np.float32)

    def _compute_reward(self, world_state):
        if world_state.number_of_observations_since_last_state == 0:
            return 0.0

        raw = json.loads(world_state.observations[-1].text)
        x   = raw.get('XPos', 0.0)
        y   = raw.get('YPos', 0.0)
        z   = raw.get('ZPos', 0.0)

        if not raw.get('IsAlive', True) or self._is_in_lava(raw):
            return -1000.0

        reward = 0.0

        traveled = self._check_checkpoint_blocks(raw, x, y, z)
        if traveled:
            reward += 500.0 * traveled
            self.current_target_checkpoint_idx += traveled
            print(f"Checkpoint {self.current_target_checkpoint_idx} reached!")
            if self.current_target_checkpoint_idx >= self.num_check_points:
                reward += 500.0

        if self.current_target_checkpoint_idx < len(self.checkpoints):
            tx, tz = self.checkpoints[self.current_target_checkpoint_idx]
            dist = math.sqrt((tx - x) ** 2 + (tz - z) ** 2)
            if self.prev_dist is not None:
                reward += (self.prev_dist - dist) * 5.0
            self.prev_dist = dist

        reward -= 0.1
        return reward

    def _check_done(self, world_state):
        if not world_state.is_mission_running:
            self._mission_needs_restart = True
            self._mission_running = False
            return True

        if world_state.number_of_observations_since_last_state > 0:
            raw = json.loads(world_state.observations[-1].text)
            if not raw.get('IsAlive', True) or self._is_in_lava(raw):
                self._mission_needs_restart = False
                return True

        if self.current_target_checkpoint_idx >= self.num_check_points:
            self._mission_needs_restart = False
            return True

        return False

    # -------------------------------------------------------------------------
    # Block-scanning helpers
    # -------------------------------------------------------------------------

    def _iter_grid(self, obs):
        grid = obs.get('nearby_blocks', [])
        idx  = 0
        for y_off in range(-1, 2):
            for z_off in range(-3, 4):
                for x_off in range(-3, 4):
                    if idx >= len(grid):
                        return
                    yield grid[idx], x_off, y_off, z_off
                    idx += 1

    def _check_checkpoint_blocks(self, obs, agent_x, agent_y, agent_z):
        for block, x_off, y_off, z_off in self._iter_grid(obs):
            if block in ('gold_block', 'emerald_block'):
                horiz = math.sqrt(x_off ** 2 + z_off ** 2)
                if 1.5 < y_off < 2.5 and horiz < 1.5:
                    if self.current_target_checkpoint_idx < len(self.checkpoints):
                        ex, ez = self.checkpoints[self.current_target_checkpoint_idx]
                        bx, bz = agent_x + x_off, agent_z + z_off
                        if math.sqrt((ex - bx) ** 2 + (ez - bz) ** 2) < 1.0:
                            return 1
        return 0

    def _is_in_lava(self, obs):
        lava_types = {'lava', 'flowing_lava', self.reset_block_type}
        for block, x_off, y_off, z_off in self._iter_grid(obs):
            if block in lava_types:
                if abs(x_off) < 1.0 and abs(y_off) < 1.0 and abs(z_off) < 1.0:
                    return True
        return False