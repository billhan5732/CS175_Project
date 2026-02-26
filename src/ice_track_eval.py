import MalmoPython
import json
import math
import time
import sys
import argparse

import gym
from stable_baselines3 import SAC
from gym import spaces
import numpy as np

from ice_track_testing import create_combined_tracks_mission, RESET_BLOCK_TYPE

TICK_LENGTH = 0.05
SEED = 67
INPUT_THRESHOLD = 0.1
CHECKPOINT_TIME_LIMIT = 300


class MalmoBoatEvalEnv(gym.Env):
    """
    Eval environment — observation/action spaces match training env exactly.
    Single checkpoint objective, no curriculum, no track switching.
    """

    def __init__(self):
        super(MalmoBoatEvalEnv, self).__init__()

        # Must exactly match training env
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(6,),
            dtype=np.float32
        )

        self.agent_host = MalmoPython.AgentHost()
        try:
            self.agent_host.parse(sys.argv)
        except RuntimeError as e:
            print(f"ERROR parsing agent arguments: {e}")
            print(self.agent_host.getUsage())

        combined_data = create_combined_tracks_mission(num_tracks=2, track_x_spacing=200, seed=SEED)
        self.mission_xml   = combined_data['mission_xml']
        self.tracks_data   = combined_data['tracks']
        self.num_tracks    = combined_data['num_tracks']

        self._mission_running = False
        self._mission_needs_restart = True

        self.current_target_checkpoint_idx = 1
        self.checkpoints = []
        self.spawn_point = None
        self.prev_dist = None
        self.prev_pos = None
        self.prev_yaw = None
        self.last_raw_obs = None
        self.checkpoint_threshold = 5.0
        self.reset_block_type = RESET_BLOCK_TYPE

        self.position_history = []
        self.position_history_len = 20
        self.stuck_threshold = 1.0

        self.checkpoint_timer = 0
        self.last_steering = 0.0
        self.last_throttle = 0.0

        self.current_track_idx = 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_current_track_info(self):
        track_data = self.tracks_data[self.current_track_idx]
        return track_data['spawn_point'], track_data['checkpoints']

    def _kill_boats_and_respawn(self, spawn_x, spawn_z):
        self.agent_host.sendCommand("chat /kill @e[type=minecraft:boat]")
        time.sleep(TICK_LENGTH * 5)

        for key in ["forward", "back", "left", "right"]:
            self.agent_host.sendCommand(f"{key} 0")
        time.sleep(TICK_LENGTH * 10)

        self.agent_host.sendCommand(f"tp {spawn_x} 230 {spawn_z}")
        for key in ["forward", "back", "left", "right"]:
            self.agent_host.sendCommand(f"{key} 0")
        time.sleep(TICK_LENGTH * 10)

        self.agent_host.sendCommand("moveMouse 0 -1000")
        self.agent_host.sendCommand("setYaw 0")
        time.sleep(TICK_LENGTH * 5)

        self.agent_host.sendCommand(f"chat /summon minecraft:boat {spawn_x} 227 {spawn_z}")
        time.sleep(TICK_LENGTH * 10)
        self.agent_host.sendCommand(f"tp {spawn_x} 227 {spawn_z}")

        self.agent_host.sendCommand("use 1")
        time.sleep(TICK_LENGTH * 5)
        self.agent_host.sendCommand("use 0")
        time.sleep(TICK_LENGTH * 5)

        self.agent_host.sendCommand("moveMouse 0 600")
        time.sleep(TICK_LENGTH * 5)

        self.prev_yaw = None
        self.last_raw_obs = None

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self):
        if self._mission_needs_restart:
            return self._full_reset()
        else:
            return self._quick_respawn()

    def _quick_respawn(self):
        self.spawn_point, self.checkpoints = self._get_current_track_info()
        self.checkpoints = self.checkpoints.copy()
        self.checkpoints.append(self.checkpoints[0])

        # Always start from checkpoint 0 in eval for consistency
        self.current_target_checkpoint_idx = 1
        self.prev_dist = None
        self.prev_pos = None
        self.prev_yaw = None
        self.position_history = []
        self.checkpoint_timer = 0
        self.last_steering = 0.0
        self.last_throttle = 0.0

        self._kill_boats_and_respawn(self.checkpoints[0][0], self.checkpoints[0][1])
        return self._get_observation()

    def _full_reset(self):
        print("Starting eval mission...")

        mission = MalmoPython.MissionSpec(self.mission_xml, True)
        mission_record = MalmoPython.MissionRecordSpec()

        for attempt in range(3):
            try:
                self.agent_host.startMission(mission, mission_record)
                break
            except RuntimeError as e:
                if attempt < 2:
                    print(f"Mission start failed (attempt {attempt + 1}): {e}")
                    time.sleep(2.0 * (attempt + 1))
                else:
                    raise

        world_state = self.agent_host.getWorldState()
        while not world_state.has_mission_begun:
            time.sleep(0.1)
            world_state = self.agent_host.getWorldState()

        self._mission_running = True
        self._mission_needs_restart = False

        self.spawn_point, self.checkpoints = self._get_current_track_info()
        self.checkpoints = self.checkpoints.copy()
        self.checkpoints.append(self.checkpoints[0])

        self.current_target_checkpoint_idx = 1
        self.prev_dist = None
        self.prev_pos = None
        self.prev_yaw = None
        self.position_history = []
        self.checkpoint_timer = 0
        self.last_steering = 0.0
        self.last_throttle = 0.0

        time.sleep(10)
        self._kill_boats_and_respawn(self.checkpoints[0][0], self.checkpoints[0][1])
        return self._get_observation()

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, action):
        throttle = float(action[0])
        steering = float(action[1])

        self.last_throttle = throttle
        self.last_steering = steering

        for _ in range(3):
            for key in ["forward", "back", "left", "right"]:
                self.agent_host.sendCommand(f"{key} 0")
            time.sleep(0.01)

        if throttle > INPUT_THRESHOLD:
            self.agent_host.sendCommand("forward 1")
        elif throttle < -INPUT_THRESHOLD:
            self.agent_host.sendCommand("back 1")

        if steering > INPUT_THRESHOLD:
            self.agent_host.sendCommand("right 1")
        elif steering < -INPUT_THRESHOLD:
            self.agent_host.sendCommand("left 1")

        time.sleep(TICK_LENGTH * 3)

        for key in ["forward", "back", "left", "right"]:
            self.agent_host.sendCommand(f"{key} 0")

        world_state = self.agent_host.getWorldState()
        obs = self._get_observation(world_state)

        self.checkpoint_timer += 1

        reward = self._compute_reward()

        if throttle > INPUT_THRESHOLD:
            reward += 0.1  # FORWARD_BONUS

        done = self._check_done(world_state)

        info = {
            'checkpoint': self.current_target_checkpoint_idx,
            'total_checkpoints': len(self.checkpoints),
            'track_idx': self.current_track_idx,
            'checkpoint_timer': self.checkpoint_timer,
        }

        return obs, reward, done, info

    # ------------------------------------------------------------------
    # Reward — exact copy of training env
    # ------------------------------------------------------------------

    def _compute_reward(self):
        reward = 0.0

        if self.last_raw_obs is not None:
            observation = self.last_raw_obs
            x = observation.get('XPos', 0)
            y = observation.get('YPos', 0)
            z = observation.get('ZPos', 0)

            if self.prev_pos is not None:
                dx_moved = x - self.prev_pos[0]
                dz_moved = z - self.prev_pos[1]
                speed = math.sqrt(dx_moved ** 2 + dz_moved ** 2)
            else:
                speed = 0.0
            self.prev_pos = (x, z)

            yaw = observation.get('Yaw', 0)
            facing_x = -math.sin(math.radians(yaw))
            facing_z =  math.cos(math.radians(yaw))

            # Spinning in place
            if self.checkpoint_timer > 10 and speed < 0.3 and abs(self.last_steering) > INPUT_THRESHOLD:
                reward += -10.0
                return reward

            # Yaw penalty when stationary
            if self.prev_yaw is not None and speed < 0.3:
                yaw_delta = abs(yaw - self.prev_yaw)
                if yaw_delta > 180:
                    yaw_delta = 360 - yaw_delta
                reward -= yaw_delta * 0.02
            self.prev_yaw = yaw

            # Lava
            if self._is_in_lava_coords(observation):
                reward += -10.0
                return reward

            # Checkpoint reached
            if self.current_target_checkpoint_idx < len(self.checkpoints):
                if self._check_checkpoint_coords(x, y, z):
                    print(f"Reached checkpoint {self.current_target_checkpoint_idx}!")
                    reward += 10.0
                    self.current_target_checkpoint_idx += 1
                    self.checkpoint_timer = 0
                    self.prev_dist = None
                    return reward

            # Direction reward — only when moving
            if self.current_target_checkpoint_idx < len(self.checkpoints):
                target = self.checkpoints[self.current_target_checkpoint_idx]
                dx = target[0] - x
                dz = target[1] - z
                target_len = math.sqrt(dx ** 2 + dz ** 2)
                cos_angle = (facing_x * (dx / target_len) + facing_z * (dz / target_len)) if target_len > 0 else 1.0

                if speed > 0.3:
                    reward += max(0, cos_angle) ** 3 * 1.0

                dist = math.sqrt(dx ** 2 + dz ** 2)
                if speed > 0.5 and self.prev_dist is not None:
                    reward += (self.prev_dist - dist) * 1.0
                self.prev_dist = dist

            # Stuck penalty
            self.position_history.append((x, z))
            if len(self.position_history) > self.position_history_len:
                self.position_history.pop(0)
            if len(self.position_history) == self.position_history_len:
                oldest_x, oldest_z = self.position_history[0]
                displacement = math.sqrt((x - oldest_x) ** 2 + (z - oldest_z) ** 2)
                if displacement < self.stuck_threshold:
                    reward += -1.0

            reward += -0.01

        return reward

    # ------------------------------------------------------------------
    # Done / helpers
    # ------------------------------------------------------------------

    def _check_done(self, world_state):
        if self.current_target_checkpoint_idx >= len(self.checkpoints):
            print("All checkpoints reached!")
            return True
        if self.last_raw_obs is not None and self._is_in_lava_coords(self.last_raw_obs):
            print("Fell into lava!")
            return True
        if self.checkpoint_timer > CHECKPOINT_TIME_LIMIT:
            print("Checkpoint timer expired!")
            return True
        if not world_state.is_mission_running:
            print("Mission not running!")
            self._mission_needs_restart = True
            return True
        return False

    def _check_checkpoint_coords(self, agent_x, agent_y, agent_z):
        expected = self.checkpoints[self.current_target_checkpoint_idx]
        dist = np.sqrt((expected[0] - agent_x) ** 2 + (expected[1] - agent_z) ** 2)
        return dist < self.checkpoint_threshold

    def _is_in_lava_coords(self, observation):
        return observation.get('YPos', 227) < 226.5

    # ------------------------------------------------------------------
    # Observation — exact copy of training env
    # ------------------------------------------------------------------

    def _get_observation(self, world_state=None):
        if world_state is None:
            world_state = self.agent_host.getWorldState()

        if world_state.number_of_observations_since_last_state > 0:
            msg = world_state.observations[-1].text
            observation = json.loads(msg)

            x = observation.get('XPos', 0)
            z = observation.get('ZPos', 0)

            if self.last_raw_obs is not None:
                vx = x - self.last_raw_obs.get('XPos', x)
                vz = z - self.last_raw_obs.get('ZPos', z)
            else:
                vx, vz = 0.0, 0.0

            self.last_raw_obs = observation

            yaw = observation.get('Yaw', 0)
            if 'entities' in observation:
                for entity in observation['entities']:
                    if entity.get('name') == 'Boat':
                        yaw = entity.get('yaw', yaw)
                        break

            if self.current_target_checkpoint_idx < len(self.checkpoints):
                target = self.checkpoints[self.current_target_checkpoint_idx]
                dx_to_target = target[0] - x
                dz_to_target = target[1] - z

                facing_x = -math.sin(math.radians(yaw))
                facing_z =  math.cos(math.radians(yaw))

                target_len = math.sqrt(dx_to_target ** 2 + dz_to_target ** 2)
                if target_len > 0:
                    cos_angle = facing_x * (dx_to_target / target_len) + facing_z * (dz_to_target / target_len)
                    sin_angle = facing_x * (dz_to_target / target_len) - facing_z * (dx_to_target / target_len)
                    if cos_angle > 0.95:
                        print(f"Facing checkpoint {self.current_target_checkpoint_idx} | cos_angle: {cos_angle:.3f}")
                else:
                    cos_angle = 1.0
                    sin_angle = 0.0

                obs_values = [dx_to_target, dz_to_target, vx, vz, cos_angle, sin_angle]
            else:
                obs_values = [0.0, 0.0, vx, vz, 1.0, 0.0]

            return np.array(obs_values, dtype=np.float32)

        if self.last_raw_obs is not None:
            x = self.last_raw_obs.get('XPos', 0)
            z = self.last_raw_obs.get('ZPos', 0)
            if self.current_target_checkpoint_idx < len(self.checkpoints):
                target = self.checkpoints[self.current_target_checkpoint_idx]
                obs_values = [target[0] - x, target[1] - z, 0.0, 0.0, 1.0, 0.0]
            else:
                obs_values = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
            return np.array(obs_values, dtype=np.float32)

        return np.zeros(6, dtype=np.float32)

    def close(self):
        if self._mission_running:
            try:
                self.agent_host.sendCommand("quit")
            except Exception:
                pass
        self._mission_running = False


# ------------------------------------------------------------------
# Eval runner
# ------------------------------------------------------------------

def run_evaluation(model_path="boat_racing_sac_interrupted", num_episodes=5):
    env = MalmoBoatEvalEnv()

    print(f"Loading model from: {model_path}")
    model = SAC.load(model_path, env=env)
    print("Model loaded successfully!")

    episode_rewards = []
    episode_checkpoints = []

    for episode in range(num_episodes):
        print(f"\n{'='*50}")
        print(f"Episode {episode + 1} / {num_episodes}")
        print(f"{'='*50}")

        obs = env.reset()
        done = False
        total_reward = 0.0
        step_count = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            step_count += 1

            if step_count % 50 == 0:
                print(f"  Step {step_count} | Checkpoint {info['checkpoint']}/{info['total_checkpoints']} "
                      f"| Running reward: {total_reward:.1f}")

        checkpoints_reached = info['checkpoint'] - 1  # subtract 1 since idx is next target
        print(f"\nEpisode {episode + 1} complete!")
        print(f"  Steps:        {step_count}")
        print(f"  Total reward: {total_reward:.2f}")
        print(f"  Checkpoints:  {checkpoints_reached} / {info['total_checkpoints'] - 1}")
        episode_rewards.append(total_reward)
        episode_checkpoints.append(checkpoints_reached)

    print(f"\n{'='*50}")
    print(f"Evaluation Summary ({num_episodes} episodes)")
    print(f"  Mean reward:      {np.mean(episode_rewards):.2f}")
    print(f"  Max reward:       {np.max(episode_rewards):.2f}")
    print(f"  Min reward:       {np.min(episode_rewards):.2f}")
    print(f"  Mean checkpoints: {np.mean(episode_checkpoints):.1f}")
    print(f"{'='*50}")

    env.close()
    return episode_rewards


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained SAC boat racing agent.")
    parser.add_argument("--model", type=str, default="boat_racing_sac_interrupted",
                        help="Path to saved SAC model")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of episodes to run")
    args = parser.parse_args()

    run_evaluation(model_path=args.model, num_episodes=args.episodes)