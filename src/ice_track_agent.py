import MalmoPython
import json
import random
import math
import time

from ice_track_testing import create_varied_environments, create_combined_tracks_mission, RESET_BLOCK_TYPE

import gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from gym import spaces
import numpy as np

TICK_LENGTH = 0.05
CHECK_POINT_SKIP_ALLOWED = False
TESTING = False

print("imported successfully!")


class MalmoBoatEnv(gym.Env):
    def __init__(self):
        super(MalmoBoatEnv, self).__init__()

        # --- Define action space (DISCRETE) ---
        # throttle: 0=nothing, 1=forward, 2=back
        # steering: 0=nothing, 1=left, 2=right
        self.action_space = spaces.MultiDiscrete([3, 3])

        # --- Define observation space ---
        # [dx_to_checkpoint1, dz_to_checkpoint1,
        #  dx_to_checkpoint2, dz_to_checkpoint2,
        #  dx_to_checkpoint3, dz_to_checkpoint3,
        #  velocity_x, velocity_z, yaw]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(9,),
            dtype=np.float32
        )

        # Malmo agent host
        self.agent_host = MalmoPython.AgentHost()

        # Generate combined mission with all tracks
        combined_data = create_combined_tracks_mission(num_tracks=5, track_x_spacing=200)
        self.mission_xml = combined_data['mission_xml']
        self.tracks_data = combined_data['tracks']
        self.num_tracks = combined_data['num_tracks']
        self.track_spacing = combined_data['track_spacing']

        self.current_track_idx = 0

        # Track switching
        self.episodes_on_current_track = 0
        self.episodes_per_track = 2  # Switch track every 10 episodes

        # Mission state
        self._mission_running = False
        self._mission_needs_restart = True

        # Track current checkpoint
        self.current_target_checkpoint_idx = 0
        self.checkpoints = []
        self.spawn_point = None
        self.num_check_points = 0
        self.prev_dist = None

        self.reset_block_type = RESET_BLOCK_TYPE

    def _get_current_track_info(self):
        """Get spawn point and checkpoints for current track"""
        track_data = self.tracks_data[self.current_track_idx]
        return track_data['spawn_point'], track_data['checkpoints']

    def _check_done(self, world_state):
        """Check if episode should terminate"""
        # All checkpoints reached
        if self.current_target_checkpoint_idx >= len(self.checkpoints):
            return True

        # Check if agent is in lava (manual detection since Creative mode doesn't die)
        if world_state.number_of_observations_since_last_state > 0:
            msg = world_state.observations[-1].text
            observation = json.loads(msg)

            # Check if touching lava
            if self._is_in_lava(observation):
                return True

        # Mission ended unexpectedly - need full restart
        if not world_state.is_mission_running:
            self._mission_needs_restart = True
            return True

        return False

    def reset(self):
        """Reset the environment"""
        # Switch tracks every N episodes
        if self.episodes_on_current_track >= self.episodes_per_track:
            self.episodes_on_current_track = 0
            self.current_track_idx = (self.current_track_idx + 1) % self.num_tracks
            print(f"Switching to track {self.current_track_idx}")

        # Start mission on first reset
        if self._mission_needs_restart:
            return self._full_reset()
        else:
            return self._quick_respawn()

    def _quick_respawn(self):
        """Quick respawn - teleport to current track's spawn"""
        print(f"Quick Respawn on track {self.current_track_idx}")

        # Update spawn and checkpoints for current track FIRST
        self.spawn_point, self.checkpoints = self._get_current_track_info()
        self.checkpoints = self.checkpoints.copy()
        self.checkpoints.append(self.checkpoints[0])  # Add loop back
        self.num_check_points = len(self.checkpoints)

        self.tpToTrackSpawnAndSpawnBoat()

        # Reset tracking
        self.current_target_checkpoint_idx = 1
        self.prev_dist = None
        self.episodes_on_current_track += 1

        return self._get_observation()

    def tpToTrackSpawnAndSpawnBoat(self):

        spawn_x, spawn_z = self.spawn_point

        for key in ["forward", "back", "left", "right"]:
            self.agent_host.sendCommand(f"{key} 0")
        time.sleep(TICK_LENGTH * 10)



        # Teleport to spawn
        self.agent_host.sendCommand(f"tp {spawn_x} 230 {spawn_z}")
        #self.agent_host.sendCommand("chat /kill @e[type=boat]")
        for key in ["forward", "back", "left", "right"]:
            self.agent_host.sendCommand(f"{key} 0")
        time.sleep(TICK_LENGTH * 10)

        # Look down and enter boat
        self.agent_host.sendCommand("moveMouse 0 -1000")
        self.agent_host.sendCommand("setYaw 0")
        time.sleep(TICK_LENGTH * 5)

        # Summon new boat at spawn
        self.agent_host.sendCommand(f"chat /summon minecraft:boat {spawn_x} 227 {spawn_z}")
        time.sleep(TICK_LENGTH * 10)
        self.agent_host.sendCommand(f"tp {spawn_x} 227 {spawn_z}")

        self.agent_host.sendCommand("use 1")
        time.sleep(TICK_LENGTH * 5)
        self.agent_host.sendCommand("use 0")
        time.sleep(TICK_LENGTH * 5)

        self.agent_host.sendCommand("moveMouse 0 600")
        time.sleep(TICK_LENGTH * 5)

    def _full_reset(self):
        """Start the mission - only called once"""
        print("Starting combined mission with all tracks...")

        mission = MalmoPython.MissionSpec(self.mission_xml, True)
        mission_record = MalmoPython.MissionRecordSpec()

        max_retries = 3
        for retry in range(max_retries):
            try:
                self.agent_host.startMission(mission, mission_record)
                break
            except RuntimeError as e:
                if retry < max_retries - 1:
                    print(f"Error starting mission (attempt {retry + 1}/{max_retries}): {e}")
                    time.sleep(2.0 * (retry + 1))
                else:
                    print(f"Failed to start mission after {max_retries} attempts: {e}")
                    raise

        # Wait for mission to begin
        world_state = self.agent_host.getWorldState()
        while not world_state.has_mission_begun:
            time.sleep(0.1)
            world_state = self.agent_host.getWorldState()

        self._mission_running = True
        self._mission_needs_restart = False

        # Set up first track
        self.spawn_point, self.checkpoints = self._get_current_track_info()
        self.checkpoints = self.checkpoints.copy()
        self.checkpoints.append(self.checkpoints[0])
        self.num_check_points = len(self.checkpoints)
        self.current_target_checkpoint_idx = 1
        self.prev_dist = None

        time.sleep(10)#load in wait time

        self.tpToTrackSpawnAndSpawnBoat()

        self.episodes_on_current_track += 1

        return self._get_observation()

    def step(self, action):
        """Execute one step in the environment"""
        throttle_action, steering_action = action

        # Release all keys first
        for key in ["forward", "back", "left", "right"]:
            self.agent_host.sendCommand(f"{key} 0")

        # Throttle control (discrete)
        if throttle_action == 1:
            self.agent_host.sendCommand("forward 1")
        elif throttle_action == 2:
            self.agent_host.sendCommand("back 1")

        # Steering control (discrete)
        if steering_action == 1:
            self.agent_host.sendCommand("left 1")
        elif steering_action == 2:
            self.agent_host.sendCommand("right 1")

        # Wait for physics to update
        time.sleep(TICK_LENGTH * 6)

        # Get current world state
        world_state = self.agent_host.getWorldState()

        # Get observations
        obs = self._get_observation()

        # Compute reward
        reward = self._compute_reward(obs, world_state)

        # Check if done
        done = self._check_done(world_state)

        # Info dict
        info = {
            'checkpoint': self.current_target_checkpoint_idx,
            'total_checkpoints': len(self.checkpoints),
            'track_idx': self.current_track_idx,
            'reason': None
        }

        return obs, reward, done, info

    def _compute_reward(self, obs, world_state):
        """Compute reward based on current state"""
        reward = 0.0

        # Get current position from latest observation
        if world_state.number_of_observations_since_last_state > 0:
            msg = world_state.observations[-1].text
            observation = json.loads(msg)
            x = observation.get('XPos', 0)
            y = observation.get('YPos', 0)
            z = observation.get('ZPos', 0)

            # Check if agent died - heavy penalty (shouldn't happen in Creative)
            is_alive = observation.get('IsAlive', True)
            if not is_alive:
                reward -= 1000.0
                return reward

            # Check if touching lava - heavy penalty
            if self._is_in_lava(observation):
                reward -= 1000.0
                return reward

            checkpoints_traveled = self._check_checkpoint_blocks(observation, x, y, z)

            # Large reward for checkpoints
            reward += 500.0 * checkpoints_traveled
            self.current_target_checkpoint_idx += checkpoints_traveled

            # Extra bonus for completing all checkpoints
            if checkpoints_traveled > 0:
                print(f"Made it to checkpoint {self.current_target_checkpoint_idx}!")
                if self.current_target_checkpoint_idx >= len(self.checkpoints):
                    reward += 500.0

            # Distance-based shaping
            if self.current_target_checkpoint_idx < len(self.checkpoints):
                target = self.checkpoints[self.current_target_checkpoint_idx]
                dist = np.sqrt((target[0] - x) ** 2 + (target[1] - z) ** 2)

                if self.prev_dist is not None:
                    reward += (self.prev_dist - dist) * 5.0
                    #print(f"Progressed Distance To Next Checkpoint: {self.prev_dist - dist}")
                self.prev_dist = dist

            # Small time penalty to encourage faster completion
            reward -= 0.1

        return reward

    def approximate_checkpoint_idx(self, block_x, block_z):
        for checkpoint_idx in range(len(self.checkpoints)):
            x_dist = (self.checkpoints[checkpoint_idx])[0] - block_x
            z_dist = (self.checkpoints[checkpoint_idx])[1] - block_z

            if np.sqrt(x_dist ** 2 + z_dist ** 2) < 3:
                return checkpoint_idx
        return -1

    def _check_checkpoint_blocks(self, observation, agent_x, agent_y, agent_z):
        """Check if agent is passing under a checkpoint block (gold or emerald)"""

        # Get the grid data
        if 'nearby_blocks' not in observation:
            return 0

        grid = observation.get('nearby_blocks', [])

        # Grid dimensions from mission XML: min=(-3,-1,-3), max=(3,1,3)
        # This creates a 7x3x7 grid
        idx = 0
        for y_offset in range(-1, 2):  # -1, 0, 1
            for z_offset in range(-3, 4):  # -3 to 3
                for x_offset in range(-3, 4):  # -3 to 3
                    if idx >= len(grid):
                        break

                    block_type = grid[idx]
                    idx += 1

                    # Check for checkpoint blocks
                    if block_type in ['gold_block', 'emerald_block']:
                        # Calculate actual block position
                        bx = agent_x + x_offset
                        by = agent_y + y_offset
                        bz = agent_z + z_offset

                        # Check if block is ~2 blocks above agent and horizontally aligned
                        horizontal_dist = np.sqrt(x_offset ** 2 + z_offset ** 2)

                        # Block should be 2 blocks above and agent should be roughly under it
                        if 1.5 < (by - agent_y) < 2.5 and horizontal_dist < 1.5:
                            if CHECK_POINT_SKIP_ALLOWED:
                                checkpoint_id = self.approximate_checkpoint_idx(int(bx), int(bz))
                                if checkpoint_id < 0:
                                    continue

                                forward_dist = (
                                                       checkpoint_id - self.current_target_checkpoint_idx + self.num_check_points
                                               ) % self.num_check_points

                                # Allow skipping 1 checkpoint, but penalize larger skips or backward movement
                                if forward_dist == 0:
                                    return 0  # Same checkpoint, no reward
                                elif forward_dist <= 2:  # Next checkpoint or skip one
                                    return forward_dist
                                else:  # Going backward
                                    return -forward_dist
                            else:
                                # Verify this is the NEXT checkpoint we're expecting
                                if self.current_target_checkpoint_idx < len(self.checkpoints):
                                    expected = self.checkpoints[self.current_target_checkpoint_idx]
                                    checkpoint_dist = np.sqrt((expected[0] - bx) ** 2 + (expected[1] - bz) ** 2)

                                    # Make sure this block is at the expected checkpoint location
                                    if checkpoint_dist < 1.0:
                                        return 1

        return 0

    def _is_in_lava(self, observation):
        """Check if the boat is in/touching lava"""

        if 'nearby_blocks' not in observation:
            return False

        grid = observation.get('nearby_blocks', [])

        # Grid dimensions from mission XML
        idx = 0
        for y_offset in range(-1, 2):  # -1, 0, 1
            for z_offset in range(-3, 4):  # -3 to 3
                for x_offset in range(-3, 4):  # -3 to 3
                    if idx >= len(grid):
                        break

                    block_type = grid[idx]
                    idx += 1

                    if block_type in ['lava', 'flowing_lava', self.reset_block_type]:
                        # Check if lava is very close (boat is touching/in it)
                        if abs(x_offset) < 1.0 and abs(y_offset) < 1.0 and abs(z_offset) < 1.0:
                            return True

        return False

    def _get_observation(self):
        """Get current observation from Malmo"""
        world_state = self.agent_host.getWorldState()

        if world_state.number_of_observations_since_last_state > 0:
            msg = world_state.observations[-1].text
            observation = json.loads(msg)

            x = observation.get('XPos', 0)
            z = observation.get('ZPos', 0)
            vx = observation.get('XVel', 0)
            vz = observation.get('ZVel', 0)
            yaw = observation.get('Yaw', 0)

            # Get next 3 checkpoints (or as many as remain)
            obs_values = []

            for i in range(3):  # Next 3 checkpoints
                checkpoint_idx = self.current_target_checkpoint_idx + i

                if checkpoint_idx < len(self.checkpoints):
                    target = self.checkpoints[checkpoint_idx]
                    dx = target[0] - x
                    dz = target[1] - z
                else:
                    # If no more checkpoints, use zeros
                    dx, dz = 0, 0

                obs_values.extend([dx, dz])

            # Add velocity and yaw
            obs_values.extend([vx, vz, yaw])

            return np.array(obs_values, dtype=np.float32)

        # Fallback if no observations yet
        return np.zeros(9, dtype=np.float32)

    def close(self):
        """Clean up"""
        if self._mission_running:
            try:
                self.agent_host.sendCommand("quit")
            except:
                pass


if __name__ == "__main__" and not TESTING:
    # Create the environment
    env = MalmoBoatEnv()

    # Create the PPO model with higher entropy for exploration
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.01,  # Entropy coefficient for exploration
    )

    # Save checkpoints every 10000 steps
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./models/",
        name_prefix="boat_racing_ppo"
    )

    print("Starting training...")

    try:
        model.learn(
            total_timesteps=100000,
            callback=checkpoint_callback
        )
        print("Training complete! Saving final model...")
        model.save("boat_racing_ppo_final")
        print("Model saved as 'boat_racing_ppo_final'")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user (Ctrl+C)!")
        print("Saving model before exit...")
        model.save("boat_racing_ppo_interrupted")
        print("Model saved as 'boat_racing_ppo_interrupted'")
        print(f"Latest checkpoint also available in ./models/ directory")

    finally:
        # Clean up environment
        env.close()
        print("Environment closed.")

if __name__ == "__main__":
    # Generate 5 training environments
    envs = create_varied_environments(5)

    # Print info about the first environment
    print(f"Generated star track with {len(envs[0]['checkpoints'])} checkpoints")
    print(f"Starting vertex: {envs[0]['start_vertex_idx']} (marked with EMERALD)")
    print(f"Number of bridge shortcuts: {len(envs[0]['bridges'])}")
    print(f"Bridge connections: {envs[0]['bridges']}")

    # Start a mission with the first environment
    agent_host = MalmoPython.AgentHost()

    my_mission = MalmoPython.MissionSpec(envs[0]['mission_xml'], True)
    my_mission_record = MalmoPython.MissionRecordSpec()

    # Launch the mission
    try:
        agent_host.startMission(my_mission, my_mission_record)
    except RuntimeError as e:
        print(f"Error starting mission: {e}")
        exit(1)

    # Wait for the mission to start
    print("Waiting for mission to start...")
    world_state = agent_host.getWorldState()
    while not world_state.has_mission_begun:
        time.sleep(0.1)
        world_state = agent_host.getWorldState()

    agent_host.sendCommand("use 1")
    time.sleep(TICK_LENGTH)
    agent_host.sendCommand("use 0")
    time.sleep(TICK_LENGTH)
    agent_host.sendCommand("moveMouse 0 500")

    print("Mission started! Star track with bridge shortcuts - CONTINUOUS PATH!")
    print("Camera positioned directly above, looking down.")
    print("EMERALD block = starting checkpoint")
    print("GOLD blocks = other checkpoints")
    print("Press CTRL+C to exit.")