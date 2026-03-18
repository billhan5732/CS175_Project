from malmo import MalmoPython
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

        self.action_space = spaces.MultiDiscrete([3, 3])

        # Observation:
        # [dx1, dz1,
        #  dx2, dz2,
        #  dx3, dz3,
        #  vx, vz,
        #  cos_angle_to_target, sin_angle_to_target,
        #  normalized_progress_to_segment_end, normalized_lateral_offset]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(12,),
            dtype=np.float32
        )

        self.agent_host = MalmoPython.AgentHost()
        try:
            self.agent_host.parse(sys.argv)
        except RuntimeError as e:
            print(f'ERROR parsing agent arguments: {e}')
            print(self.agent_host.getUsage())

        combined_data = create_combined_tracks_mission(num_tracks=2, track_x_spacing=200)
        self.mission_xml = combined_data['mission_xml']
        self.tracks_data = combined_data['tracks']
        self.num_tracks = combined_data['num_tracks']
        self.track_spacing = combined_data['track_spacing']

        self.current_track_idx = 0

        self.episodes_on_current_track = 0
        self.episodes_per_track = 2

        self._mission_running = False
        self._mission_needs_restart = True

        self.current_target_checkpoint_idx = 1
        self.checkpoints = []
        self.spawn_point = None
        self.num_check_points = 0
        self.prev_dist = None
        self.last_raw_obs = None

        self.reset_block_type = RESET_BLOCK_TYPE
        self.checkpoint_threshold = 8

        # New reward / episode state
        self.max_episode_steps = 550
        self.max_no_progress_steps = 90
        self.min_progress_delta = 0.08
        self.max_same_segment_steps = 200
        self.same_segment_steps = 0
        self.same_segment_timeout_penalty = 120.0
        self.max_episode_timeout_penalty = 80.0

        self.episode_step_count = 0
        self.no_progress_steps = 0
        self.prev_segment_progress = None
        self.prev_x = None
        self.prev_z = None

        self.debug_rewards = False
        self._checkpoint_debug_done = False
        self.original_num_checkpoints = 0
        self.wrong_way_finish_hit = False

        self.max_consecutive_turn_steps = 1
        self.turn_cooldown_steps = 1
        self.turn_streak = 0
        self.last_turn_direction = 0   # 0 none, 1 left, 2 right

        self.force_forward_steps = 0
        self.force_forward_duration = 2

        self.wall_positions = set()
        self.wall_touch_streak = 0
        self.wall_touch_radius = 1.25
        self.wall_touch_penalty = 1.5

    def _mc_print(self, msg):
        print(msg)
        msg = msg.replace('"', "'")
        try:
            self.agent_host.sendCommand(
                f'chat /tellraw @p {{"text":"{msg}","color":"yellow"}}'
            )
        except Exception:
            pass

    def _update_debug_hud(self, progress=None, speed=None):
        try:
            seg_from = max(0, self.current_target_checkpoint_idx - 1)
            seg_to = self.current_target_checkpoint_idx

            if progress is None:
                progress_text = "N/A"
            else:
                progress_text = f"{progress:.2f}"

            if speed is None:
                speed_text = "N/A"
            else:
                speed_text = f"{speed:.2f}"

            msg = (
                f"Segment {seg_from}->{seg_to} | "
                f"Progress {progress_text} | "
                f"Speed {speed_text} | "
                f"NoProg {self.no_progress_steps} | "
                f"SameSeg {self.same_segment_steps}"
            )

            msg = msg.replace('"', "'")

            self.agent_host.sendCommand(
                f'chat /title @p actionbar {{"text":"{msg}","color":"aqua"}}'
            )
        except Exception:
            pass

    def _hit_last_checkpoint_too_early(self, x, z):
        if self.episode_step_count < 10:
            return False

        if self.current_target_checkpoint_idx > 2:
            return False

        if not self.checkpoints or self.original_num_checkpoints <= 0:
            return False

        # The real final checkpoint before the wraparound copy of checkpoint 0
        real_last_idx = self.original_num_checkpoints - 1
        last_cp = self.checkpoints[real_last_idx]

        dist_to_last = np.sqrt((last_cp[0] - x) ** 2 + (last_cp[1] - z) ** 2)

        if dist_to_last >= self.checkpoint_threshold:
            return False

        # Allow it only when the agent is actually supposed to be on the final segment
        # or already targeting the real last checkpoint.
        if self.current_target_checkpoint_idx >= real_last_idx:
            return False

        return True

    def _get_current_track_info(self):
        track_data = self.tracks_data[self.current_track_idx]
        return track_data['spawn_point'], track_data['checkpoints'], track_data.get('wall_positions', [])


    def _is_touching_wall(self, x, z):
        if not self.wall_positions:
            return False

        cx = int(round(x))
        cz = int(round(z))
        search_r = 2
        radius_sq = self.wall_touch_radius * self.wall_touch_radius

        for wx in range(cx - search_r, cx + search_r + 1):
            for wz in range(cz - search_r, cz + search_r + 1):
                if (wx, wz) not in self.wall_positions:
                    continue
                dx = x - wx
                dz = z - wz
                if dx * dx + dz * dz <= radius_sq:
                    return True

        return False

    def _get_current_segment_points(self):
        if not self.checkpoints or self.current_target_checkpoint_idx >= len(self.checkpoints):
            return None, None

        prev_idx = self.current_target_checkpoint_idx - 1
        start_cp = self.checkpoints[prev_idx]
        end_cp = self.checkpoints[self.current_target_checkpoint_idx]
        return start_cp, end_cp

    def _segment_geometry(self, x, z):
        start_cp, end_cp = self._get_current_segment_points()
        if start_cp is None or end_cp is None:
            return None

        sx, sz = start_cp
        ex, ez = end_cp

        seg_dx = ex - sx
        seg_dz = ez - sz
        seg_len = math.sqrt(seg_dx * seg_dx + seg_dz * seg_dz) + 1e-6

        ux = seg_dx / seg_len
        uz = seg_dz / seg_len

        rel_x = x - sx
        rel_z = z - sz

        progress = rel_x * ux + rel_z * uz
        lateral = rel_x * (-uz) + rel_z * ux

        return {
            "start": start_cp,
            "end": end_cp,
            "seg_len": seg_len,
            "ux": ux,
            "uz": uz,
            "progress": progress,
            "lateral": lateral,
        }

    def _check_done(self, world_state):
        if self.wrong_way_finish_hit:
            print("Touched final checkpoint too early.")
            return True

        if self.current_target_checkpoint_idx >= len(self.checkpoints):
            print("All checkpoints reached! Lap complete.")
            return True

        if self.no_progress_steps >= self.max_no_progress_steps:
            print("No progress for too long. Resetting.")
            return True

        if self.same_segment_steps >= self.max_same_segment_steps:
            self._mc_print("Stayed in the same segment too long!")
            return True

        if self.episode_step_count >= self.max_episode_steps:
            self._mc_print("Episode step limit reached!")
            return True

        if not world_state.is_mission_running:
            print("Mission not running!")
            self._mission_needs_restart = True
            return True

        return False

    def reset(self):
        if self.episodes_on_current_track >= self.episodes_per_track:
            self.episodes_on_current_track = 0
            self.current_track_idx = (self.current_track_idx + 1) % self.num_tracks

        if self._mission_needs_restart:
            return self._full_reset()
        return self._quick_respawn()

    def _quick_respawn(self):
        print(f"----------------------------------------------------------")

        self.spawn_point, self.checkpoints, wall_positions = self._get_current_track_info()
        self.checkpoints = self.checkpoints.copy()
        self.wall_positions = set((int(wx), int(wz)) for wx, wz in wall_positions)

        self.original_num_checkpoints = len(self.checkpoints)

        self.checkpoints.append(self.checkpoints[0])
        self.num_check_points = len(self.checkpoints)

        self.current_target_checkpoint_idx = 1
        self.prev_dist = None
        self.episode_step_count = 0
        self.no_progress_steps = 0
        self.prev_segment_progress = None
        self.last_raw_obs = None
        self.same_segment_steps = 0
        self.prev_x = None
        self.prev_z = None
        self._checkpoint_debug_done = False
        self.wrong_way_finish_hit = False
        self.turn_streak = 0
        self.last_turn_direction = 0
        self.force_forward_steps = 0
        self.wall_touch_streak = 0

        self.tpToTrackSpawnAndSpawnBoat()

        self.episodes_on_current_track += 1

        return self._get_observation()

    def _cleanup_boats_for_current_track(self):
        if not self.checkpoints:
            return

        cps = self.checkpoints[:-1] if len(self.checkpoints) > 1 and self.checkpoints[0] == self.checkpoints[-1] else self.checkpoints

        xs = [cp[0] for cp in cps]
        zs = [cp[1] for cp in cps]

        padding = 25
        min_x = math.floor(min(xs) - padding)
        max_x = math.ceil(max(xs) + padding)
        min_z = math.floor(min(zs) - padding)
        max_z = math.ceil(max(zs) + padding)

        dx = max_x - min_x
        dz = max_z - min_z

        # Stop movement first
        for key in ["forward", "back", "left", "right"]:
            self.agent_host.sendCommand(f"{key} 0")
        time.sleep(TICK_LENGTH * 2)

        # Kill all boats in this track region
        self.agent_host.sendCommand('chat /kill @e[type=minecraft:boat]')
        time.sleep(TICK_LENGTH * 8)

    def tpToTrackSpawnAndSpawnBoat(self):
        self._cleanup_boats_for_current_track()
        y_boat = 227
        y_steve = 228

        cp0_x, cp0_z = self.checkpoints[0]
        cp1_x, cp1_z = self.checkpoints[1]

        dx = cp1_x - cp0_x
        dz = cp1_z - cp0_z
        norm = math.sqrt(dx * dx + dz * dz) + 1e-6
        ux = dx / norm
        uz = dz / norm

        px = -uz
        pz = ux

        forward = 3.0
        side = 2.0

        boat_x = cp0_x + ux * forward + px * side
        boat_z = cp0_z + uz * forward + pz * side

        for _ in range(3):
            for key in ["forward", "back", "left", "right"]:
                self.agent_host.sendCommand(f"{key} 0")
            time.sleep(TICK_LENGTH * 2)

        self.agent_host.sendCommand("setPitch 0")
        self.agent_host.sendCommand("setYaw 0")
        time.sleep(TICK_LENGTH * 6)

        self.agent_host.sendCommand(f"chat /summon minecraft:boat {boat_x} {y_boat} {boat_z}")
        time.sleep(TICK_LENGTH * 10)

        self.agent_host.sendCommand(f"tp {boat_x} {y_steve} {boat_z}")
        time.sleep(TICK_LENGTH * 10)

        self.agent_host.sendCommand("moveMouse 0 -1000")
        time.sleep(TICK_LENGTH * 6)

        self.agent_host.sendCommand("use 1")
        time.sleep(TICK_LENGTH * 8)
        self.agent_host.sendCommand("use 0")
        time.sleep(TICK_LENGTH * 6)

        self.agent_host.sendCommand("moveMouse 0 600")
        time.sleep(TICK_LENGTH * 6)

        for key in ["forward", "back", "left", "right"]:
            self.agent_host.sendCommand(f"{key} 0")
        time.sleep(TICK_LENGTH * 2)

    def _full_reset(self):
        print("Starting combined mission with all tracks...")

        with open("debug_mission.xml", "w", encoding="utf-8") as f:
            f.write(self.mission_xml)

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

        world_state = self.agent_host.getWorldState()
        while not world_state.has_mission_begun:
            time.sleep(0.1)
            world_state = self.agent_host.getWorldState()

        self._mission_running = True
        self._mission_needs_restart = False
        self.agent_host.sendCommand("chat /gamerule sendCommandFeedback false")
        self.agent_host.sendCommand("chat /gamerule commandBlockOutput false")
        self.agent_host.sendCommand("chat /gamerule logAdminCommands false")

        self.spawn_point, self.checkpoints, wall_positions = self._get_current_track_info()
        self.checkpoints = self.checkpoints.copy()
        self.wall_positions = set((int(wx), int(wz)) for wx, wz in wall_positions)

        self.original_num_checkpoints = len(self.checkpoints)

        self.checkpoints.append(self.checkpoints[0])
        self.num_check_points = len(self.checkpoints)

        self.current_target_checkpoint_idx = 1
        self.prev_dist = None
        self.episode_step_count = 0
        self.no_progress_steps = 0
        self.prev_segment_progress = None
        self.last_raw_obs = None
        self.same_segment_steps = 0
        self.prev_x = None
        self.prev_z = None
        self._checkpoint_debug_done = False
        self.wrong_way_finish_hit = False
        self.turn_streak = 0
        self.last_turn_direction = 0
        self.force_forward_steps = 0
        self.wall_touch_streak = 0

        time.sleep(10)

        self.tpToTrackSpawnAndSpawnBoat()

        self.episodes_on_current_track += 1

        return self._get_observation()

    def step(self, action):
        self.episode_step_count += 1

        throttle_action, steering_action = action

        if self.force_forward_steps > 0:
            throttle_action = 1
            steering_action = 0
            self.force_forward_steps -= 1
            self.turn_streak = 0
            self.last_turn_direction = 0
        else:
            if steering_action == 0:
                self.turn_streak = 0
                self.last_turn_direction = 0
            else:
                if steering_action == self.last_turn_direction:
                    self.turn_streak += 1
                else:
                    self.turn_streak = 1
                    self.last_turn_direction = steering_action

                if self.turn_streak > self.max_consecutive_turn_steps:
                    throttle_action = 1
                    steering_action = 0
                    self.force_forward_steps = self.force_forward_duration - 1
                    self.turn_streak = 0
                    self.last_turn_direction = 0

        for _ in range(3):
            for key in ["forward", "back", "left", "right"]:
                self.agent_host.sendCommand(f"{key} 0")
            time.sleep(0.01)

        if throttle_action == 1:
            self.agent_host.sendCommand("forward 1")
        elif throttle_action == 2:
            self.agent_host.sendCommand("back 1")

        if steering_action == 1:
            self.agent_host.sendCommand("left 1")
        elif steering_action == 2:
            self.agent_host.sendCommand("right 1")

        time.sleep(TICK_LENGTH * 6)

        for key in ["forward", "back", "left", "right"]:
            self.agent_host.sendCommand(f"{key} 0")

        world_state = self.agent_host.getWorldState()

        obs = self._get_observation(world_state)
        reward = self._compute_reward()
        done = self._check_done(world_state)

        info = {
            'checkpoint': self.current_target_checkpoint_idx,
            'total_checkpoints': len(self.checkpoints),
            'track_idx': self.current_track_idx,
            'episode_step_count': self.episode_step_count,
            'no_progress_steps': self.no_progress_steps,
        }

        return obs, reward, done, info

    def _compute_reward(self):
        if self.last_raw_obs is None:
            return 0.0

        obs = self.last_raw_obs
        x = obs.get('XPos', 0.0)
        z = obs.get('ZPos', 0.0)
        vx = obs.get('XVel', 0.0)
        vz = obs.get('ZVel', 0.0)

        reward = 0.0

        if self._hit_last_checkpoint_too_early(x, z):
            self.wrong_way_finish_hit = True
            return -1500.0

        if self.current_target_checkpoint_idx >= len(self.checkpoints):
            return 0.0

        hit = self._check_checkpoint_coords(x, 0, z)
        if hit > 0:
            reward += 1200.0 * hit
            self.current_target_checkpoint_idx += hit

            self.prev_dist = None
            self.prev_segment_progress = None
            self.no_progress_steps = 0
            self.same_segment_steps = 0

            if self.current_target_checkpoint_idx >= len(self.checkpoints):
                reward += 6000.0
                return reward

        self.same_segment_steps += 1

        if self.same_segment_steps >= self.max_same_segment_steps:
            reward -= self.same_segment_timeout_penalty
            return reward

        geom = self._segment_geometry(x, z)
        if geom is None:
            return reward

        progress = geom["progress"]
        seg_len = geom["seg_len"]
        ux = geom["ux"]
        uz = geom["uz"]
        lateral = geom["lateral"]

        clamped_progress = max(-10.0, min(seg_len + 10.0, progress))

        target = self.checkpoints[self.current_target_checkpoint_idx]
        dx = target[0] - x
        dz = target[1] - z
        dist = math.sqrt(dx * dx + dz * dz) + 1e-6

        dist_improvement = 0.0
        if self.prev_dist is not None:
            dist_improvement = self.prev_dist - dist

        if self.prev_segment_progress is not None:
            delta_progress = clamped_progress - self.prev_segment_progress

            reward += 12.0 * delta_progress

            if delta_progress < 0:
                reward += 5.0 * delta_progress
            elif delta_progress > self.min_progress_delta:
                reward += 0.4

            if delta_progress > self.min_progress_delta or dist_improvement > 0.05:
                self.no_progress_steps = 0
            else:
                self.no_progress_steps += 1
        else:
            delta_progress = 0.0
            self.no_progress_steps += 1

        self.prev_segment_progress = clamped_progress

        v_forward = vx * ux + vz * uz
        reward += 1.2 * v_forward
        if v_forward < 0:
            reward += 0.9 * v_forward

        reward -= 0.06 * abs(lateral)

        reward += 3.0 * dist_improvement
        self.prev_dist = dist

        touching_wall = self._is_touching_wall(x, z)
        if touching_wall:
            self.wall_touch_streak += 1
            reward -= self.wall_touch_penalty
            reward -= 0.15 * min(self.wall_touch_streak, 5)
        else:
            self.wall_touch_streak = 0

        pos_speed = 0.0
        if self.prev_x is not None and self.prev_z is not None:
            step_dx = x - self.prev_x
            step_dz = z - self.prev_z
            pos_speed = math.sqrt(step_dx * step_dx + step_dz * step_dz) / (TICK_LENGTH * 6)

        self.prev_x = x
        self.prev_z = z

        if pos_speed < 0.04:
            reward -= 0.3

        if self.episode_step_count % 8 == 0:
            self._update_debug_hud(progress=clamped_progress, speed=pos_speed)

        reward -= 0.03

        if self.debug_rewards:
            print(
                f"reward={reward:.2f}, progress={clamped_progress:.2f}, "
                f"delta={delta_progress:.2f}, lateral={lateral:.2f}, "
                f"v_forward={v_forward:.2f}, dist_improve={dist_improvement:.2f}, "
                f"wall={touching_wall}, no_prog={self.no_progress_steps}"
            )
        
        if self.episode_step_count >= self.max_episode_steps:
            reward -= self.max_episode_timeout_penalty
    
        return reward

    def _check_checkpoint_coords(self, agent_x, agent_y, agent_z):
        if not self._checkpoint_debug_done:
            self._checkpoint_debug_done = True

        if self.current_target_checkpoint_idx >= len(self.checkpoints):
            return 0

        max_lookahead = 1
        furthest_hit_idx = None

        for checkpoint_idx in range(
            self.current_target_checkpoint_idx,
            min(self.current_target_checkpoint_idx + max_lookahead + 1, len(self.checkpoints))
        ):
            target = self.checkpoints[checkpoint_idx]
            checkpoint_dist = np.sqrt((target[0] - agent_x) ** 2 + (target[1] - agent_z) ** 2)

            if checkpoint_dist < self.checkpoint_threshold:
                furthest_hit_idx = checkpoint_idx

        if furthest_hit_idx is not None:
            advanced = furthest_hit_idx - self.current_target_checkpoint_idx + 1

            if advanced == 1:
                self._mc_print(f"Reached checkpoint {furthest_hit_idx}!")
            else:
                self._mc_print(
                    f"Skipped ahead to checkpoint {furthest_hit_idx}. "
                    f"Auto-crediting missed checkpoint(s)."
                )

            return advanced

        return 0

    def _is_in_lava_coords(self, observation):
        y = observation.get('YPos', 227)
        return y < 226.5

    def _build_obs_from_raw(self, observation):
        x = observation.get('XPos', 0.0)
        z = observation.get('ZPos', 0.0)
        vx = observation.get('XVel', 0.0)
        vz = observation.get('ZVel', 0.0)

        yaw = observation.get('Yaw', 0.0)
        if 'entities' in observation:
            for entity in observation['entities']:
                if entity.get('name') == 'Boat':
                    yaw = entity.get('yaw', yaw)
                    break

        obs_values = []

        for i in range(3):
            checkpoint_idx = self.current_target_checkpoint_idx + i
            if checkpoint_idx < len(self.checkpoints):
                target = self.checkpoints[checkpoint_idx]
                dx = target[0] - x
                dz = target[1] - z
            else:
                dx, dz = 0.0, 0.0
            obs_values.extend([dx, dz])

        obs_values.extend([vx, vz])

        if self.current_target_checkpoint_idx < len(self.checkpoints):
            target = self.checkpoints[self.current_target_checkpoint_idx]
            dx_to_target = target[0] - x
            dz_to_target = target[1] - z

            angle_to_target = math.atan2(dz_to_target, dx_to_target)
            yaw_radians = math.radians(-yaw + 90)

            relative_angle = angle_to_target - yaw_radians
            relative_angle = math.atan2(math.sin(relative_angle), math.cos(relative_angle))

            cos_angle = math.cos(relative_angle)
            sin_angle = math.sin(relative_angle)
        else:
            cos_angle = 1.0
            sin_angle = 0.0

        obs_values.extend([cos_angle, sin_angle])

        geom = self._segment_geometry(x, z)
        if geom is not None:
            seg_len = geom["seg_len"]
            progress = geom["progress"]
            lateral = geom["lateral"]

            progress_to_end = (seg_len - progress) / max(seg_len, 1e-6)
            lateral_norm = lateral / max(seg_len, 1.0)
        else:
            progress_to_end = 0.0
            lateral_norm = 0.0

        obs_values.extend([progress_to_end, lateral_norm])

        return np.array(obs_values, dtype=np.float32)

    def _get_observation(self, world_state=None):
        if world_state is None:
            world_state = self.agent_host.getWorldState()

        if world_state.number_of_observations_since_last_state > 0:
            msg = world_state.observations[-1].text
            observation = json.loads(msg)
            self.last_raw_obs = observation
            return self._build_obs_from_raw(observation)

        if self.last_raw_obs is not None:
            return self._build_obs_from_raw(self.last_raw_obs)

        return np.zeros(12, dtype=np.float32)

    def close(self):
        if self._mission_running:
            try:
                self.agent_host.sendCommand("quit")
            except Exception:
                pass

# "C:\Users\dkdlwkr\AppData\Local\Programs\Python\Python37\python.exe" train.py
import os

if __name__ == "__main__" and not TESTING:
    env = MalmoBoatEnv()

    resume_path = "boat_racing_ppo_interrupted.zipasas"
    #resume_path = "./models/boat_racing_ppo_348865_steps.zip"

    if os.path.exists(resume_path):
        print(f"Resuming training from {resume_path}")
        model = PPO.load(resume_path, env=env)
    else:
        print("No saved model found. Starting fresh.")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=1e-4,
            n_steps=4096,
            batch_size=128,
            n_epochs=10,
            gamma=0.995,
            gae_lambda=0.95,
            ent_coef=0.005,
        )

    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./models/",
        name_prefix="boat_racing_ppo"
    )

    print("Starting training.")

    try:
        model.learn(
            total_timesteps=500000,
            callback=checkpoint_callback,
            reset_num_timesteps=False
        )
        print("Training complete! Saving final model...")
        model.save("boat_racing_ppo_final")
        print("Model saved as 'boat_racing_ppo_final'")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user (Ctrl+C)!")
        print("Saving model before exit...")
        model.save("boat_racing_ppo_interrupted")
        print("Model saved as 'boat_racing_ppo_interrupted'")
        print("Latest checkpoint also available in ./models/ directory")

    finally:
        env.close()
        print("Environment closed.")