import MalmoPython
import json
import random
import math
import time
import sys

from ice_track_testing import create_combined_tracks_mission, RESET_BLOCK_TYPE

import gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from gym import spaces
import numpy as np

TICK_LENGTH = 0.05
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
        #  velocity_x, velocity_z, cos_angle_to_target, sin_angle_to_target]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(10,),
            dtype=np.float32
        )

        # Malmo agent host
        self.agent_host = MalmoPython.AgentHost()
        try:
            self.agent_host.parse(sys.argv)
        except RuntimeError as e:
            print(f'ERROR parsing agent arguments: {e}')
            print(self.agent_host.getUsage())

        # Generate combined mission with all tracks
        combined_data = create_combined_tracks_mission(num_tracks=5, track_x_spacing=200)
        self.mission_xml = combined_data['mission_xml']
        self.tracks_data = combined_data['tracks']
        self.num_tracks = combined_data['num_tracks']
        self.track_spacing = combined_data['track_spacing']

        self.current_track_idx = 0

        # Track switching
        self.episodes_on_current_track = 0
        self.episodes_per_track = 2  # Switch track every 2 episodes

        # Mission state
        self._mission_running = False
        self._mission_needs_restart = True

        # Track current checkpoint
        self.current_target_checkpoint_idx = 0
        self.checkpoints = []
        self.spawn_point = None
        self.num_check_points = 0
        self.prev_dist = None

        # Store last raw observation for reward computation
        self.last_raw_obs = None

        self.reset_block_type = RESET_BLOCK_TYPE

        # Checkpoint detection threshold
        self.checkpoint_threshold = 5.0  # 5 blocks radius

    def _get_current_track_info(self):
        """Get spawn point and checkpoints for current track"""
        track_data = self.tracks_data[self.current_track_idx]
        return track_data['spawn_point'], track_data['checkpoints']

    def _check_done(self, world_state):
        """Check if episode should terminate"""
        # All checkpoints reached
        if self.current_target_checkpoint_idx >= len(self.checkpoints):
            print("All checkpoints reached!")
            return True

        # Check if agent is in lava using coordinates
        if self.last_raw_obs is not None:
            if self._is_in_lava_coords(self.last_raw_obs):
                print("Fell into lava! Resetting...")
                return True

        # Mission ended unexpectedly - need full restart
        if not world_state.is_mission_running:
            print("Mission not running!")
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

        time.sleep(10)  # load in wait time

        self.tpToTrackSpawnAndSpawnBoat()

        self.episodes_on_current_track += 1

        return self._get_observation()

    def step(self, action):
        """Execute one step in the environment"""
        throttle_action, steering_action = action

        # CRITICAL: Release all keys MULTIPLE times to ensure commands clear
        for _ in range(3):
            for key in ["forward", "back", "left", "right"]:
                self.agent_host.sendCommand(f"{key} 0")
            time.sleep(0.01)

        # Throttle control (discrete)
        if throttle_action == 1:
            self.agent_host.sendCommand("forward 1")
        elif throttle_action == 2:
            self.agent_host.sendCommand("back 1")

        # Steering control (discrete) - only apply if throttle is active
        # This prevents spinning in place
        if throttle_action != 0:  # Only steer if moving
            if steering_action == 1:
                self.agent_host.sendCommand("left 1")
            elif steering_action == 2:
                self.agent_host.sendCommand("right 1")

        # Wait for physics to update
        time.sleep(TICK_LENGTH * 6)

        # Clear commands again after action
        for key in ["forward", "back", "left", "right"]:
            self.agent_host.sendCommand(f"{key} 0")

        # Get current world state
        world_state = self.agent_host.getWorldState()

        # Get observations
        obs = self._get_observation(world_state)

        # Compute reward using stored raw observation
        reward = self._compute_reward()

        # Check if done
        done = self._check_done(world_state)

        # Info dict
        info = {
            'checkpoint': self.current_target_checkpoint_idx,
            'total_checkpoints': len(self.checkpoints),
            'track_idx': self.current_track_idx,
        }

        return obs, reward, done, info

    def _compute_reward(self):
        """Compute reward based on current state"""
        reward = 0.0

        # Use stored raw observation
        if self.last_raw_obs is not None:
            observation = self.last_raw_obs
            x = observation.get('XPos', 0)
            y = observation.get('YPos', 0)
            z = observation.get('ZPos', 0)

            # Get yaw to detect spinning
            yaw = observation.get('Yaw', 0)

            # Track yaw changes to penalize spinning
            if hasattr(self, 'last_yaw'):
                yaw_change = abs(yaw - self.last_yaw)
                # Normalize for wrap-around (359 -> 0)
                if yaw_change > 180:
                    yaw_change = 360 - yaw_change

                # Penalize excessive turning (spinning in place)
                if yaw_change > 30:  # More than 30 degrees per step
                    reward -= yaw_change * 0.5  # Penalty proportional to spin

            self.last_yaw = yaw

            # Check if touching lava - heavy penalty
            if self._is_in_lava_coords(observation):
                reward -= 1000.0
                return reward

            # Check checkpoint using coordinates
            checkpoints_traveled = self._check_checkpoint_coords(x, y, z)

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
                self.prev_dist = dist

            # Small time penalty to encourage faster completion
            reward -= 0.1

        return reward

    def _check_checkpoint_coords(self, agent_x, agent_y, agent_z):
        """Check if agent is close to the next checkpoint using coordinates"""

        # One-time debug on first call
        if not hasattr(self, '_checkpoint_debug_done'):
            self._checkpoint_debug_done = True
            print(f"\nCheckpoint system initialized - Using coordinate-based detection")
            print(f"Agent at ({agent_x:.1f}, {agent_y:.1f}, {agent_z:.1f})")
            print(f"Next checkpoint at: {self.checkpoints[self.current_target_checkpoint_idx]}")
            print(f"Detection threshold: {self.checkpoint_threshold} blocks\n")

        # Check if we're close to the next expected checkpoint
        if self.current_target_checkpoint_idx < len(self.checkpoints):
            expected = self.checkpoints[self.current_target_checkpoint_idx]

            # Calculate horizontal distance to checkpoint
            checkpoint_dist = np.sqrt((expected[0] - agent_x) ** 2 + (expected[1] - agent_z) ** 2)

            # If within threshold distance of checkpoint, count it as reached
            if checkpoint_dist < self.checkpoint_threshold:
                print(f"Reached checkpoint {self.current_target_checkpoint_idx}! Distance: {checkpoint_dist:.2f}")
                return 1

        return 0

    def _is_in_lava_coords(self, observation):
        """Check if the boat is in lava using Y coordinate"""
        y = observation.get('YPos', 227)

        # If boat has fallen below ice level (y=226), it's in lava
        # Normal boat position is y=227 (on ice)
        if y < 226.5:
            return True

        return False

    def _get_observation(self, world_state=None):
        """Get current observation from Malmo"""
        if world_state is None:
            world_state = self.agent_host.getWorldState()

        if world_state.number_of_observations_since_last_state > 0:
            msg = world_state.observations[-1].text
            observation = json.loads(msg)

            # Store raw observation for reward computation
            self.last_raw_obs = observation

            x = observation.get('XPos', 0)
            z = observation.get('ZPos', 0)
            vx = observation.get('XVel', 0)
            vz = observation.get('ZVel', 0)

            # Try to get boat yaw from nearby entities, fall back to agent yaw
            yaw = observation.get('Yaw', 0)
            if 'entities' in observation:
                for entity in observation['entities']:
                    if entity.get('name') == 'Boat':
                        yaw = entity.get('yaw', yaw)
                        break

            # Get next 3 checkpoints (or as many as remain)
            obs_values = []

            for i in range(3):  # Next 3 checkpoints
                checkpoint_idx = self.current_target_checkpoint_idx + i

                if checkpoint_idx < len(self.checkpoints):
                    target = self.checkpoints[checkpoint_idx]
                    dx = target[0] - x
                    dz = target[1] - z
                else:
                    dx, dz = 0, 0

                obs_values.extend([dx, dz])

            # Add velocity
            obs_values.extend([vx, vz])

            # Compute relative angle to next checkpoint
            if self.current_target_checkpoint_idx < len(self.checkpoints):
                target = self.checkpoints[self.current_target_checkpoint_idx]
                dx_to_target = target[0] - x
                dz_to_target = target[1] - z

                # Angle to target (in radians)
                angle_to_target = math.atan2(dz_to_target, dx_to_target)

                # Convert yaw to radians
                yaw_radians = math.radians(-yaw + 90)

                # Relative angle
                relative_angle = angle_to_target - yaw_radians
                relative_angle = math.atan2(math.sin(relative_angle), math.cos(relative_angle))

                # Convert to cos/sin
                cos_angle = math.cos(relative_angle)
                sin_angle = math.sin(relative_angle)
            else:
                cos_angle = 1.0
                sin_angle = 0.0

            obs_values.extend([cos_angle, sin_angle])

            return np.array(obs_values, dtype=np.float32)

        # Fallback: use last known observation or zeros
        # This is OK - we'll get fresh data on next step
        if self.last_raw_obs is not None:
            # Reuse last observation to construct state
            x = self.last_raw_obs.get('XPos', 0)
            z = self.last_raw_obs.get('ZPos', 0)
            obs_values = []
            for i in range(3):
                checkpoint_idx = self.current_target_checkpoint_idx + i
                if checkpoint_idx < len(self.checkpoints):
                    target = self.checkpoints[checkpoint_idx]
                    obs_values.extend([target[0] - x, target[1] - z])
                else:
                    obs_values.extend([0, 0])
            obs_values.extend([0, 0, 1.0, 0.0])  # zero velocity, neutral angle
            return np.array(obs_values, dtype=np.float32)

        return np.zeros(10, dtype=np.float32)

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

    # Create the PPO model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.01,
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
        env.close()
        print("Environment closed.")