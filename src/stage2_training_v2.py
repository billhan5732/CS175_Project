import MalmoPython
import json
import random
import math
import time
import sys
import os

from ice_track_testing import create_combined_tracks_mission, RESET_BLOCK_TYPE

import gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, CallbackList
from gym import spaces
import numpy as np

TICK_LENGTH = 0.05
TESTING = False
SEED = 67

# Stage 2 configuration - spawn further ahead but not at the end
SPAWN_MIN_CHECKPOINT = 0  # Can spawn at start
SPAWN_MAX_CHECKPOINT = -3  # Up to 3 checkpoints from the end

INPUT_THRESHOLD = 0.1
CHECKPOINT_TIME_LIMIT = 300  # Back to Stage 1 value

# STAGE 1 REWARD VALUES (unchanged from original)
SPIN_PENALTY = -10.0
LAVA_PENALTY = -10.0
CHECKPOINT_REWARD = 10.0  # Back to original 10.0
COMPLETION_BONUS = 50.0  # Keep completion bonus
DIRECTION_REWARD = 1.0
DISTANCE_SCALE = 1.0
STUCK_PENALTY = -1.0
TIME_PENALTY = -0.01
FORWARD_BONUS = 0.1
YAW_PENALTY_SCALE = 0.02
ALIGNMENT_REWARD = 0.05

# RESUME TRAINING SETTINGS
RESUME_FROM_CHECKPOINT = True  # Set to True to resume from interrupted/checkpoint
RESUME_MODEL_PATH = "models_stage2/boat_racing_sac_stage2_interrupted"  # Path to resume from

print("Stage 2 Training - Multi-checkpoint sequences (Stage 1 rewards)!")


class MalmoBoatEnv(gym.Env):
    def __init__(self):
        super(MalmoBoatEnv, self).__init__()

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

        # Malmo agent host
        self.agent_host = MalmoPython.AgentHost()
        try:
            self.agent_host.parse(sys.argv)
        except RuntimeError as e:
            print(f'ERROR parsing agent arguments: {e}')
            print(self.agent_host.getUsage())

        combined_data = create_combined_tracks_mission(num_tracks=2, track_x_spacing=200, seed=SEED)
        self.mission_xml = combined_data['mission_xml']
        self.tracks_data = combined_data['tracks']
        self.num_tracks = combined_data['num_tracks']
        self.track_spacing = combined_data['track_spacing']

        self.current_track_idx = 0
        self.episodes_on_current_track = 0
        self.episodes_per_track = 15

        self._mission_running = False
        self._mission_needs_restart = True

        self.current_target_checkpoint_idx = 1
        self.checkpoints = []
        self.spawn_point = None
        self.num_check_points = 0
        self.prev_dist = None
        self.prev_pos = None
        self.prev_yaw = None

        self.last_raw_obs = None
        self.checkpoint_threshold = 5.0

        self.position_history = []
        self.position_history_len = 20
        self.stuck_threshold = 1.0

        self.checkpoint_timer = 0
        self.last_steering = 0.0
        self.last_throttle = 0.0

        self.total_episodes = 0
        self.checkpoints_reached_this_episode = 0
        self.starting_checkpoint = 0

        self.last_alignment_print = 0

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

    def _check_done(self, world_state):
        # Full lap complete - SUCCESS!
        if self.current_target_checkpoint_idx >= len(self.checkpoints):
            print(f"🏁 Full lap complete! Total checkpoints: {self.checkpoints_reached_this_episode}")
            return True

        if self.last_raw_obs is not None:
            if self._is_in_lava_coords(self.last_raw_obs):
                print(f"💀 Fell into lava after {self.checkpoints_reached_this_episode} checkpoints")
                return True

        if self.checkpoint_timer > CHECKPOINT_TIME_LIMIT:
            print(f"⏱️ Timeout after {self.checkpoints_reached_this_episode} checkpoints")
            return True

        if not world_state.is_mission_running:
            self._mission_needs_restart = True
            return True

        return False

    def reset(self):
        self.total_episodes += 1

        if self.episodes_on_current_track >= self.episodes_per_track:
            self.episodes_on_current_track = 0
            self.current_track_idx = (self.current_track_idx + 1) % self.num_tracks
            print(f"\n🔄 Switching to track {self.current_track_idx}\n")

        if self._mission_needs_restart:
            return self._full_reset()
        else:
            return self._quick_respawn()

    def _quick_respawn(self):
        self.spawn_point, self.checkpoints = self._get_current_track_info()
        self.checkpoints = self.checkpoints.copy()
        self.checkpoints.append(self.checkpoints[0])
        self.num_check_points = len(self.checkpoints)

        # Stage 2: Spawn at various points, but not too close to the end
        max_spawn_idx = len(self.checkpoints) - 1 + SPAWN_MAX_CHECKPOINT
        start_idx = random.randint(SPAWN_MIN_CHECKPOINT, max_spawn_idx)
        self.starting_checkpoint = start_idx
        self.current_target_checkpoint_idx = start_idx + 1

        print(f"Episode {self.total_episodes} | Track {self.current_track_idx} | "
              f"Starting at checkpoint {start_idx}/{len(self.checkpoints) - 1} | "
              f"Goal: reach checkpoint {start_idx + 1}")

        self.prev_dist = None
        self.prev_pos = None
        self.prev_yaw = None
        self.position_history = []
        self.checkpoint_timer = 0
        self.last_steering = 0.0
        self.last_throttle = 0.0
        self.checkpoints_reached_this_episode = 0
        self.episodes_on_current_track += 1
        self.last_alignment_print = 0

        self._kill_boats_and_respawn(
            self.checkpoints[start_idx][0],
            self.checkpoints[start_idx][1]
        )
        return self._get_observation()

    def _full_reset(self):
        print("🚀 Starting combined mission with all tracks...")

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
        self.num_check_points = len(self.checkpoints)

        max_spawn_idx = len(self.checkpoints) - 1 + SPAWN_MAX_CHECKPOINT
        start_idx = random.randint(SPAWN_MIN_CHECKPOINT, max_spawn_idx)
        self.starting_checkpoint = start_idx
        self.current_target_checkpoint_idx = start_idx + 1

        print(f"Episode {self.total_episodes} | Track {self.current_track_idx} | "
              f"Starting at checkpoint {start_idx}/{len(self.checkpoints) - 1} | "
              f"Goal: reach checkpoint {start_idx + 1}")

        self.prev_dist = None
        self.prev_pos = None
        self.prev_yaw = None
        self.position_history = []
        self.checkpoint_timer = 0
        self.last_steering = 0.0
        self.last_throttle = 0.0
        self.checkpoints_reached_this_episode = 0
        self.last_alignment_print = 0

        time.sleep(10)

        self._kill_boats_and_respawn(
            self.checkpoints[start_idx][0],
            self.checkpoints[start_idx][1]
        )

        self.episodes_on_current_track += 1
        return self._get_observation()

    def step(self, action):
        throttle = action[0]
        steering = action[1]

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

        # Forward bonus applied in step (not in _compute_reward)
        if throttle > INPUT_THRESHOLD:
            reward += FORWARD_BONUS

        done = self._check_done(world_state)

        info = {
            'checkpoint': self.current_target_checkpoint_idx,
            'total_checkpoints': len(self.checkpoints),
            'checkpoints_reached': self.checkpoints_reached_this_episode,
            'starting_checkpoint': self.starting_checkpoint,
            'track_idx': self.current_track_idx,
            'checkpoint_timer': self.checkpoint_timer,
        }

        return obs, reward, done, info

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
            facing_z = math.cos(math.radians(yaw))

            # Spinning in place
            if self.checkpoint_timer > 10 and speed < 0.3 and abs(self.last_steering) > INPUT_THRESHOLD:
                print("⚠️ Spinning in place!")
                reward += SPIN_PENALTY
                return reward

            # Yaw penalty when stationary
            if self.prev_yaw is not None and speed < 0.3:
                yaw_delta = abs(yaw - self.prev_yaw)
                if yaw_delta > 180:
                    yaw_delta = 360 - yaw_delta
                reward -= yaw_delta * YAW_PENALTY_SCALE
            self.prev_yaw = yaw

            # Lava
            if self._is_in_lava_coords(observation):
                reward += LAVA_PENALTY
                return reward

            # Checkpoint reached
            if (self.current_target_checkpoint_idx < len(self.checkpoints)
                    and self._check_checkpoint_coords(x, y, z)):
                self.checkpoints_reached_this_episode += 1
                remaining = len(self.checkpoints) - 1 - self.current_target_checkpoint_idx
                print(f"✓ Checkpoint {self.current_target_checkpoint_idx} reached! "
                      f"(Episode total: {self.checkpoints_reached_this_episode}, "
                      f"Remaining: {remaining})")
                reward += CHECKPOINT_REWARD

                # Check if completed full lap
                if self.current_target_checkpoint_idx == len(self.checkpoints) - 1:
                    reward += COMPLETION_BONUS
                    print(f"🎉 LAP COMPLETE! Bonus: +{COMPLETION_BONUS}")

                self.current_target_checkpoint_idx += 1
                self.checkpoint_timer = 0
                self.prev_dist = None
                self.last_alignment_print = 0

            # Alignment and direction rewards
            if self.current_target_checkpoint_idx < len(self.checkpoints):
                target = self.checkpoints[self.current_target_checkpoint_idx]
                dx = target[0] - x
                dz = target[1] - z
                target_len = math.sqrt(dx ** 2 + dz ** 2)
                if target_len > 0:
                    cos_angle = facing_x * (dx / target_len) + facing_z * (dz / target_len)
                else:
                    cos_angle = 1.0

                # Alignment reward - even when stationary
                if cos_angle > 0.95:
                    reward += ALIGNMENT_REWARD
                    if self.checkpoint_timer - self.last_alignment_print > 20:
                        print(f"👁️ Facing checkpoint {self.current_target_checkpoint_idx} | "
                              f"Alignment: {cos_angle:.3f}")
                        self.last_alignment_print = self.checkpoint_timer

                # Direction reward - only when moving
                if speed > 0.3:
                    reward += max(0, cos_angle) ** 3 * DIRECTION_REWARD

                # Distance shaping
                dist = math.sqrt(dx ** 2 + dz ** 2)
                if speed > 0.5 and self.prev_dist is not None:
                    reward += (self.prev_dist - dist) * DISTANCE_SCALE
                self.prev_dist = dist

            # Stuck penalty
            self.position_history.append((x, z))
            if len(self.position_history) > self.position_history_len:
                self.position_history.pop(0)
            if len(self.position_history) == self.position_history_len:
                oldest_x, oldest_z = self.position_history[0]
                displacement = math.sqrt((x - oldest_x) ** 2 + (z - oldest_z) ** 2)
                if displacement < self.stuck_threshold:
                    print("⚠️ Agent stuck in place!")
                    reward += STUCK_PENALTY

            # Time penalty
            reward += TIME_PENALTY

        return reward

    def _check_checkpoint_coords(self, agent_x, agent_y, agent_z):
        expected = self.checkpoints[self.current_target_checkpoint_idx]
        checkpoint_dist = np.sqrt((expected[0] - agent_x) ** 2 + (expected[1] - agent_z) ** 2)
        return checkpoint_dist < self.checkpoint_threshold

    def _is_in_lava_coords(self, observation):
        y = observation.get('YPos', 227)
        return y < 226.5

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
                facing_z = math.cos(math.radians(yaw))

                target_len = math.sqrt(dx_to_target ** 2 + dz_to_target ** 2)
                if target_len > 0:
                    target_x = dx_to_target / target_len
                    target_z = dz_to_target / target_len
                    cos_angle = facing_x * target_x + facing_z * target_z
                    sin_angle = facing_x * target_z - facing_z * target_x
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
            except:
                pass


class ProgressLoggingCallback(BaseCallback):
    """Enhanced callback with more detailed statistics"""

    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.episode_checkpoints = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
        self.episode_count = 0
        self.completed_laps = 0

    def _on_step(self):
        self.current_episode_reward += self.locals['rewards'][0]
        self.current_episode_length += 1

        if self.locals['dones'][0]:
            self.episode_count += 1
            info = self.locals['infos'][0]

            checkpoints_reached = info.get('checkpoints_reached', 0)
            total_checkpoints = info.get('total_checkpoints', 1) - 1
            starting_checkpoint = info.get('starting_checkpoint', 0)

            completed = checkpoints_reached == (total_checkpoints - starting_checkpoint)
            if completed:
                self.completed_laps += 1

            self.episode_rewards.append(self.current_episode_reward)
            self.episode_checkpoints.append(checkpoints_reached)
            self.episode_lengths.append(self.current_episode_length)

            print(f"\n{'=' * 70}")
            print(f"Episode {self.episode_count} Complete:")
            print(f"  Reward: {self.current_episode_reward:.1f}")
            print(f"  Checkpoints: {checkpoints_reached}/{total_checkpoints - starting_checkpoint}")
            print(f"  Length: {self.current_episode_length} steps")
            print(f"  Status: {'✅ COMPLETE' if completed else '❌ INCOMPLETE'}")
            print(f"{'=' * 70}\n")

            if self.episode_count % 10 == 0:
                recent_rewards = self.episode_rewards[-10:]
                recent_checkpoints = self.episode_checkpoints[-10:]
                print(f"\n{'=' * 70}")
                print(f"📊 Last 10 Episodes Summary:")
                print(f"  Avg reward: {np.mean(recent_rewards):.1f}")
                print(f"  Avg checkpoints: {np.mean(recent_checkpoints):.1f}")
                print(f"  Completion rate: {self.completed_laps}/{self.episode_count} "
                      f"({100 * self.completed_laps / self.episode_count:.1f}%)")
                print(f"{'=' * 70}\n")

            self.current_episode_reward = 0
            self.current_episode_length = 0

        return True


if __name__ == "__main__" and not TESTING:
    stage2_dir = "./models_stage2_v2/"
    os.makedirs(stage2_dir, exist_ok=True)

    env = MalmoBoatEnv()

    print("\n" + "=" * 70)

    if RESUME_FROM_CHECKPOINT:
        if os.path.exists(f"{RESUME_MODEL_PATH}.zip"):
            print(f"🔄 RESUMING training from: {RESUME_MODEL_PATH}")
            model = SAC.load(RESUME_MODEL_PATH, env=env)
            print("✓ Checkpoint loaded successfully!")
            print("   Training will continue from where it left off")
        else:
            print(f"⚠️ Checkpoint not found: {RESUME_MODEL_PATH}.zip")
            print("   Falling back to stage 1 model")
            RESUME_FROM_CHECKPOINT = False

    if not RESUME_FROM_CHECKPOINT:
        print("Loading stage 1 model: boat_racing_sac_single_checkpoint")
        model = SAC.load("boat_racing_sac_single_checkpoint", env=env)
        print("✓ Model loaded successfully!")

    print("=" * 70 + "\n")

    progress_callback = ProgressLoggingCallback()
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=stage2_dir,
        name_prefix="boat_racing_sac_stage2_v2"
    )

    print("=" * 70)
    print("STAGE 2 V2 TRAINING - Multi-checkpoint (Stage 1 Rewards)")
    print("=" * 70)
    if RESUME_FROM_CHECKPOINT:
        print(f"Resuming from: {RESUME_MODEL_PATH}")
    else:
        print(f"Starting from: boat_racing_sac_single_checkpoint")
    print(f"Spawn range: checkpoint {SPAWN_MIN_CHECKPOINT} to {SPAWN_MAX_CHECKPOINT} from end")
    print(f"Checkpoint timeout: {CHECKPOINT_TIME_LIMIT} steps")
    print(f"Checkpoint reward: +{CHECKPOINT_REWARD} (Stage 1 value)")
    print(f"Alignment reward: +{ALIGNMENT_REWARD}")
    print(f"Completion bonus: +{COMPLETION_BONUS}")
    print(f"Models saved to: {stage2_dir}")
    print(f"Track seed: {SEED}")
    print("=" * 70 + "\n")

    try:
        model.learn(
            total_timesteps=1000000,
            callback=CallbackList([progress_callback, checkpoint_callback]),
            reset_num_timesteps=False
        )
        print("\n" + "=" * 70)
        print("🎉 Training complete! Saving final model...")
        model.save(f"{stage2_dir}boat_racing_sac_stage2_v2_final")
        print(f"✓ Model saved as '{stage2_dir}boat_racing_sac_stage2_v2_final'")
        print(f"📈 Completed laps: {progress_callback.completed_laps}/{progress_callback.episode_count} "
              f"({100 * progress_callback.completed_laps / progress_callback.episode_count:.1f}%)")
        print("=" * 70)

    except KeyboardInterrupt:
        print("\n\n" + "=" * 70)
        print("⚠️ Training interrupted by user (Ctrl+C)!")
        print("Saving model before exit...")
        model.save(f"{stage2_dir}boat_racing_sac_stage2_v2_interrupted")
        print(f"✓ Model saved as '{stage2_dir}boat_racing_sac_stage2_v2_interrupted'")
        print(f"📈 Completed laps so far: {progress_callback.completed_laps}/{progress_callback.episode_count} "
              f"({100 * progress_callback.completed_laps / progress_callback.episode_count:.1f}%)")
        print(f"\n💡 To resume training, set:")
        print(f"   RESUME_FROM_CHECKPOINT = True")
        print(f"   RESUME_MODEL_PATH = '{stage2_dir}boat_racing_sac_stage2_v2_interrupted'")
        print("=" * 70)

    finally:
        env.close()
        print("Environment closed.")