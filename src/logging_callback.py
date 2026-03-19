import csv
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from pathlib import Path

class BoatLoggingCallback(BaseCallback):
    """
    Logs per-rollout stats to CSV for post-training analysis.

    Columns written:
        timestep         — total env steps so far
        ep_rew_mean      — mean episode reward over last rollout
        ep_rew_std       — std of episode rewards
        ep_len_mean      — mean episode length
        checkpoints_reached — mean checkpoints per episode
        lava_deaths      — fraction of episodes ending in lava
        timeouts         — fraction of episodes ending in timeout
        explained_var    — SB3's explained variance (value fn quality)
        entropy_loss     — policy entropy
    """

    def __init__(self, log_path="training_log.csv", verbose=0):
        super().__init__(verbose)
        self.log_path = log_path
        self._ep_rewards    = []
        self._ep_lengths    = []
        self._ep_checkpoints = []
        self._lava_deaths   = []
        self._timeouts      = []

        # Write header
        if not Path(self.log_path).exists():
            with open(self.log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestep", "ep_rew_mean", "ep_rew_std", "ep_len_mean",
                    "checkpoints_reached", "lava_fraction", "timeout_fraction",
                    "explained_variance", "entropy_loss"
                ])

    def _on_step(self) -> bool:
        # Collect episode info from infos dict (SB3 populates this on episode end)
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self._ep_rewards.append(info["episode"]["r"])
                self._ep_lengths.append(info["episode"]["l"])
            if "checkpoint" in info:
                self._ep_checkpoints.append(info["checkpoint"])
            if "lava" in info:
                self._lava_deaths.append(float(info.get("lava", False)))
            if "timeout" in info:
                self._timeouts.append(float(info.get("timeout", False)))
        return True

    def _on_rollout_end(self) -> None:
        if not self._ep_rewards:
            return

        # Pull SB3 internal metrics if available
        explained_var = self.logger.name_to_value.get("train/explained_variance", float('nan'))
        entropy_loss  = self.logger.name_to_value.get("train/entropy_loss", float('nan'))

        row = [
            self.num_timesteps,
            np.mean(self._ep_rewards),
            np.std(self._ep_rewards),
            np.mean(self._ep_lengths),
            np.mean(self._ep_checkpoints) if self._ep_checkpoints else 0.0,
            np.mean(self._lava_deaths)    if self._lava_deaths    else 0.0,
            np.mean(self._timeouts)       if self._timeouts       else 0.0,
            explained_var,
            entropy_loss,
        ]

        with open(self.log_path, 'a', newline='') as f:
            csv.writer(f).writerow(row)

        if self.verbose:
            print(f"[Log] t={self.num_timesteps:,} | "
                  f"rew={row[1]:.2f}±{row[2]:.2f} | "
                  f"len={row[3]:.0f} | "
                  f"chk={row[4]:.2f} | "
                  f"ev={row[7]:.3f}")

        # Reset buffers
        self._ep_rewards     = []
        self._ep_lengths     = []
        self._ep_checkpoints = []
        self._lava_deaths    = []
        self._timeouts       = []
