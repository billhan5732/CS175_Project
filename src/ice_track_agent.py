import MalmoPython
import json
import random
import math
import time

from ice_track_test import create_varied_environments

import gym
from stable_baselines3 import PPO
from gym import spaces
import numpy as np

TICK_LENGTH = 0.05
CHECK_POINT_SKIP_ALLOWED = True
TESTING = False

print("imported successfully!")


class MalmoBoatEnv(gym.Env):
    def __init__(self):
        super(MalmoBoatEnv, self).__init__()

        # --- Define action space ---
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )

        # --- Define observation space ---
        # [dx_to_checkpoint, dz_to_checkpoint, velocity_x, velocity_z]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(4,),
            dtype=np.float32
        )

        # Malmo agent host
        self.agent_host = MalmoPython.AgentHost()

        # Current environment index
        self.envs = create_varied_environments(5)  # Assumes this function exists
        self.current_env_idx = 0

        # Track current checkpoint
        self.current_target_checkpoint_idx = 0
        self.checkpoints = []
        self.spawn_point = None

        # Previous position for velocity estimation (backup)
        self.prev_pos = None

        self.num_check_points = 0

    def reset(self):
        """Reset the environment and start a new Malmo mission"""

        # End any currently running mission
        if hasattr(self, '_mission_running') and self._mission_running:
            try:
                world_state = self.agent_host.getWorldState()
                if world_state.is_mission_running:
                    self.agent_host.sendCommand("quit")
                    time.sleep(0.5)  # Give it time to clean up
            except:
                pass

        # Pick the next environment (round-robin)
        env_data = self.envs[self.current_env_idx]
        self.current_env_idx = (self.current_env_idx + 1) % len(self.envs)

        mission_xml = env_data['mission_xml']
        self.spawn_point = env_data['spawn_point']

        self.checkpoints = env_data['checkpoints'].copy()
        self.checkpoints.append(self.checkpoints[0])
        self.num_check_points = len(self.checkpoints)
        self.current_target_checkpoint_idx = 1

        # Initialize distance tracking
        self.prev_dist = None

        # Create Malmo mission spec
        mission = MalmoPython.MissionSpec(mission_xml, True)
        mission_record = MalmoPython.MissionRecordSpec()

        # Start mission with retry logic
        max_retries = 3
        for retry in range(max_retries):
            try:
                self.agent_host.startMission(mission, mission_record)
                break
            except RuntimeError as e:
                if retry < max_retries - 1:
                    print(f"Error starting mission (attempt {retry + 1}/{max_retries}): {e}")
                    time.sleep(2)
                else:
                    print(f"Failed to start mission after {max_retries} attempts: {e}")
                    raise

        # Wait for mission to begin
        world_state = self.agent_host.getWorldState()
        while not world_state.has_mission_begun:
            time.sleep(0.1)
            world_state = self.agent_host.getWorldState()

        self._mission_running = True

        # Enter boat
        self.agent_host.sendCommand("use 1")
        time.sleep(TICK_LENGTH)
        self.agent_host.sendCommand("use 0")
        time.sleep(TICK_LENGTH)

        # look back up
        self.agent_host.sendCommand("moveMouse 0 500")

        # Get initial observation
        obs = self._get_observation()

        return obs

    def step(self, action):
        """Execute one step in the environment"""
        throttle, steering = action
        print(f"throttle: {throttle}")
        print(f"steering: {steering}")

        for key in ["forward", "back", "left", "right"]:
            self.agent_host.sendCommand(f"{key} 0")

        # Send commands to Malmo
        if throttle > 0.1:
            self.agent_host.sendCommand(f"forward 1")
            self.agent_host.sendCommand(f"back -1")
        elif throttle < -0.1:
            self.agent_host.sendCommand(f"forward -1")
            self.agent_host.sendCommand(f"back 1")

        if steering > 0.1:
            self.agent_host.sendCommand(f"left -1")
            self.agent_host.sendCommand(f"right 1")
        elif steering < -0.1:
            self.agent_host.sendCommand(f"left 1")
            self.agent_host.sendCommand(f"right -1")

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
            'reason': None  # Will be set if episode ends
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

            checkpoints_traveled = self._check_checkpoint_blocks(observation, x, y, z)
            print(f"checkpoints_traveled: {checkpoints_traveled}")

            reward += 50.0 * checkpoints_traveled
            self.current_target_checkpoint_idx += checkpoints_traveled

            # Extra bonus for completing all checkpoints
            if checkpoints_traveled > 0:
                if self.current_target_checkpoint_idx >= len(self.checkpoints):
                    reward += 500.0

            # Distance-based shaping
            if self.current_target_checkpoint_idx < len(self.checkpoints):
                target = self.checkpoints[self.current_target_checkpoint_idx]
                # Changed from target[2] to target[1]
                dist = np.sqrt((target[0] - x) ** 2 + (target[1] - z) ** 2)

                if self.prev_dist is not None:
                    reward += (self.prev_dist - dist) * 1.0
                self.prev_dist = dist

            # Check for lava (instant death penalty)
            if self._is_in_lava(observation):
                reward -= 100.0

            # Small time penalty to encourage faster completion
            reward -= 0.1
        print(f"Reward this Step: {reward}\n")
        return reward

    def approximate_checkpoint_idx(self, block_x, block_z):
        for checkpoint_idx in range(len(self.checkpoints)):
            # Changed from [2] to [1] for z coordinate
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
        x_size = 7  # -3 to 3
        y_size = 3  # -1 to 1
        z_size = 7  # -3 to 3

        # Iterate through the grid
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
                        vertical_diff = y_offset

                        # Block should be 2 blocks above and agent should be roughly under it
                        if 1.5 < (by - agent_y) < 2.5 and horizontal_dist < 1.5:
                            print("Detected Checkpoint Block")
                            if CHECK_POINT_SKIP_ALLOWED:
                                checkpoint_id = self.approximate_checkpoint_idx(int(bx), int(bz))
                                if checkpoint_id < 0:
                                    continue

                                forward_dist = (checkpoint_id - self.current_target_checkpoint_idx + self.num_check_points) % self.num_check_points

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
        agent_x = observation.get('XPos', 0)
        agent_y = observation.get('YPos', 0)
        agent_z = observation.get('ZPos', 0)

        # Grid dimensions from mission XML
        idx = 0
        for y_offset in range(-1, 2):  # -1, 0, 1
            for z_offset in range(-3, 4):  # -3 to 3
                for x_offset in range(-3, 4):  # -3 to 3
                    if idx >= len(grid):
                        break

                    block_type = grid[idx]
                    idx += 1

                    if block_type in ['lava', 'flowing_lava']:
                        # Check if lava is very close (boat is touching/in it)
                        if abs(x_offset) < 1.0 and abs(y_offset) < 1.0 and abs(z_offset) < 1.0:
                            print("Agent in LAVA detected!")
                            return True

        return False

    def _check_done(self, world_state):
        """Check if episode should terminate"""
        # Mission ended
        if not world_state.is_mission_running:
            return True

        # All checkpoints reached
        if self.current_target_checkpoint_idx >= len(self.checkpoints):
            return True

        # Check if agent died (hit lava)
        if world_state.number_of_observations_since_last_state > 0:
            msg = world_state.observations[-1].text
            observation = json.loads(msg)

            if self._is_in_lava(observation):
                return True  # Episode ends immediately on lava contact

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

            # Get target checkpoint
            if self.current_target_checkpoint_idx < len(self.checkpoints):
                target = self.checkpoints[self.current_target_checkpoint_idx]
                # Checkpoints are (x, z) tuples, not (x, y, z)
                dx = target[0] - x
                dz = target[1] - z  # Changed from target[2] to target[1]
            else:
                dx, dz = 0, 0

            return np.array([dx, dz, vx, vz], dtype=np.float32)

        # Fallback if no observations yet
        return np.zeros(4, dtype=np.float32)

    def close(self):
        """Clean up"""
        # Malmo cleanup if needed
        pass


if __name__ == "__main__" and not TESTING:
    # Comment out or remove the test mission code
    # Just go straight to training

    env = MalmoBoatEnv()
    model = PPO("MlpPolicy", env, verbose=1)

    print("Starting training...")

    try:
        model.learn(total_timesteps=100000)
        print("Training complete! Saving model...")
        model.save("boat_racing_ppo")
        print("Model saved as 'boat_racing_ppo'")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user (Ctrl+C)!")
        print("Saving model before exit...")
        model.save("boat_racing_ppo_interrupted")
        print("Model saved as 'boat_racing_ppo_interrupted'")
        print("You can resume training by loading this model.")

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


    agent_host.sendCommand("use 1")  # Press 'use' to enter boat
    time.sleep(TICK_LENGTH)
    agent_host.sendCommand("use 0")
    time.sleep(TICK_LENGTH)
    #agent_host.sendCommand("setPitch 0")
    #agent_host.sendCommand("use 0")

    agent_host.sendCommand("moveMouse 0 500")
    #time.sleep(TICK_LENGTH * 10)
    #agent_host.sendCommand("moveMouse 0 0")


    #agent_host.sendCommand("moveMouse -90 0")
    #time.sleep(TICK_LENGTH * 10)
    #agent_host.sendCommand("moveMouse 0 0")

    #agent_host.sendCommand("mouseMove -90 0")
    #time.sleep(TICK_LENGTH * 10)
    #agent_host.sendCommand("mouseMove 0 0")

    #agent_host.sendCommand("mouseMove 90 0")
    #time.sleep(TICK_LENGTH * 10)
    #agent_host.sendCommand("mouseMove 0 0")

    #time.sleep(TICK_LENGTH*100)
    #agent_host.sendCommand("moveMouse 0 1")
    #time.sleep(TICK_LENGTH*6)

    #agent_host.sendCommand("pitch -1")
    #time.sleep(TICK_LENGTH*9)
    #agent_host.sendCommand("pitch 0")



    print("Mission started! Star track with bridge shortcuts - CONTINUOUS PATH!")
    print("Camera positioned directly above, looking down.")
    print("EMERALD block = starting checkpoint")
    print("GOLD blocks = other checkpoints")
    print("Press CTRL+C to exit.")

    movePow = -1
    elapsed = 0
    while True:
        #agent_host.sendCommand("crouch 1")
        #agent_host.sendCommand(f"pitch {np.sin(elapsed)}")
        #agent_host.sendCommand("moveMouse 0 -1")
        agent_host.sendCommand(f"forward -1")
        time.sleep(TICK_LENGTH*5)
        agent_host.sendCommand(f"forward 0")
        agent_host.sendCommand(f"left -1")
        time.sleep(TICK_LENGTH * 5)
        agent_host.sendCommand(f"left 0")
        agent_host.sendCommand(f"right 1")
        time.sleep(TICK_LENGTH * 5)
        agent_host.sendCommand(f"right 0")
        agent_host.sendCommand(f"back 1")
        time.sleep(TICK_LENGTH * 5)
        agent_host.sendCommand(f"back 0")
        #agent_host.sendCommand(f"move 1")

        #agent_host.sendCommand(f"left {np.sin(elapsed)}")

        elapsed += TICK_LENGTH*2
        time.sleep(TICK_LENGTH)
    # Keep the mission running
    # try:
    #    while world_state.is_mission_running:
    #        time.sleep(0.1)
