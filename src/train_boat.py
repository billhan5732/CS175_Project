import os
import signal
import time
from datetime import datetime

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

from malmo_boat_env import MalmoBoatEnv
from logging_callback import BoatLoggingCallback

RUN_NAME = datetime.now().strftime("dqn_%Y%m%d_%H%M%S")

SAVE_DIR = os.path.join("./models/", RUN_NAME)
LOG_DIR  = os.path.join("./tensorboard_logs/", RUN_NAME)

TOTAL_TIMESTEPS = 10_000
CHECKPOINT_FREQ = 2_000

ENV_CONFIG = {
    "mission_xml_path":  "boat_mission.xml",
    "millisec_per_tick": 20,
    "num_tracks":        5,
}

DQN_CONFIG = dict(
    policy                 = "MlpPolicy",
    learning_rate          = 1e-3,
    buffer_size            = 50_000,
    learning_starts        = 1_000,
    batch_size             = 32,
    gamma                  = 0.99,
    target_update_interval = 1_000,
    exploration_fraction   = 0.2,
    exploration_final_eps  = 0.05,
    verbose                = 1,
    tensorboard_log        = LOG_DIR,
)

_env   = None
_model = None


def _shutdown(signum, frame):
    print("\n[Shutdown] Ctrl+C — saving and cleaning up...")
    if _model is not None:
        path = os.path.join(SAVE_DIR, "dqn_boat_interrupted")
        _model.save(path)
        print(f"[Shutdown] Saved → {path}")
    if _env is not None:
        _env.close()
        print("[Shutdown] Waiting 8s for Malmo server to go DORMANT...")
        time.sleep(8)
    print("[Shutdown] Safe to run next script.")
    os._exit(0)


signal.signal(signal.SIGINT,  _shutdown)
signal.signal(signal.SIGTERM, _shutdown)


def main():
    global _env, _model

    os.makedirs(SAVE_DIR, exist_ok=True)

    print("Waiting 8s for Malmo server to be ready...")
    time.sleep(8)

    print("Initializing environment...")
    _env = MalmoBoatEnv(**ENV_CONFIG)

    print("Building DQN model...")
    _model = DQN(env=_env, **DQN_CONFIG)

    log_path = os.path.join(SAVE_DIR, "training_log.csv")
    callbacks = CallbackList([
        CheckpointCallback(
            save_freq   = CHECKPOINT_FREQ,
            save_path   = SAVE_DIR,
            name_prefix = "dqn_boat",
            verbose     = 1,
        ),
        BoatLoggingCallback(log_path=log_path, verbose=1),
    ])

    print(f"Logging to: {log_path}")
    print(f"Tensorboard: tensorboard --logdir ./tensorboard_logs/")
    print(f"Starting DQN training for {TOTAL_TIMESTEPS:,} timesteps...")

    _model.learn(
        total_timesteps = TOTAL_TIMESTEPS,
        callback        = callbacks,
        tb_log_name     = "dqn_boat",
    )

    path = os.path.join(SAVE_DIR, "dqn_boat_final")
    _model.save(path)
    print(f"Done. Model saved → {path}")
    _env.close()


if __name__ == "__main__":
    main()