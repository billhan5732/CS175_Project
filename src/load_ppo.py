import os
import signal
import time
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.logger import configure

from malmo_boat_env import MalmoBoatEnv
from logging_callback import BoatLoggingCallback

RUN_NAME = "ppo_20260225_161200"
# RUN_NAME = datetime.now().strftime("ppo_%Y%m%d_%H%M%S")

SAVE_DIR = os.path.join("./models/", RUN_NAME)
LOG_DIR  = os.path.join("./tensorboard_logs/", RUN_NAME)

TOTAL_TIMESTEPS = 90_000
CHECKPOINT_FREQ = 2_500

LOAD_PATH = os.path.join("./models/ppo_20260225_161200/ppo_boat_interrupted.zip")

_env   = None
_model = None


def _shutdown(signum, frame):
    print("\n[Shutdown] Ctrl+C -- saving and cleaning up...")
    if _model is not None:
        path = os.path.join(SAVE_DIR, "ppo_boat_interrupted")
        _model.save(path)
        print(f"[Shutdown] Saved -> {path}")
    if _env is not None:
        _env.close()
        print("[Shutdown] Waiting 8s for Malmo server to go DORMANT...")
        time.sleep(8)
    print("[Shutdown] Safe to run next script.")
    os._exit(0)


signal.signal(signal.SIGINT,  _shutdown)
signal.signal(signal.SIGTERM, _shutdown)

ENV_CONFIG = {
    "mission_xml_path":  "boat_mission.xml",
    "millisec_per_tick": 20,
    "num_tracks":        5,
}

PPO_CONFIG = dict(
    policy        = "MlpPolicy",
    verbose       = 1,
    learning_rate = 3e-4,
    n_steps       = 2048,
    batch_size    = 64,
    n_epochs      = 10,
    gamma         = 0.99,
    gae_lambda    = 0.95,
    ent_coef      = 0.01,
    clip_range    = 0.2,
    tensorboard_log = LOG_DIR,
)


def main():
    global _env, _model

    os.makedirs(SAVE_DIR, exist_ok=True)

    print("Initializing environment...")
    _env = MalmoBoatEnv(**ENV_CONFIG)

    if LOAD_PATH and os.path.exists(LOAD_PATH):
        print(f"Resuming from: {LOAD_PATH}")
        _model = PPO.load(LOAD_PATH, env=_env)
        new_logger = configure(LOG_DIR, ["stdout", "tensorboard"])
        _model.set_logger(new_logger)
    else:
        print("Building fresh PPO model...")
        _model = PPO(env=_env, **PPO_CONFIG)

    log_path = os.path.join(SAVE_DIR, "training_log.csv")
    callbacks = CallbackList([
        CheckpointCallback(
            save_freq   = CHECKPOINT_FREQ,
            save_path   = SAVE_DIR,
            name_prefix = "ppo_boat",
            verbose     = 1,
        ),
        BoatLoggingCallback(log_path=log_path, verbose=1),
    ])

    print(f"Logging to: {log_path}")
    print(f"Tensorboard: tensorboard --logdir ./tensorboard_logs/")
    print(f"Starting PPO training for {TOTAL_TIMESTEPS:,} timesteps...")

    _model.learn(
        total_timesteps     = TOTAL_TIMESTEPS,
        reset_num_timesteps = False if (LOAD_PATH and os.path.exists(LOAD_PATH)) else True,
        callback            = callbacks,
        tb_log_name         = "ppo_boat",
    )

    path = os.path.join(SAVE_DIR, "ppo_boat_final")
    _model.save(path)
    print(f"Done. Model saved -> {path}")
    _env.close()


if __name__ == "__main__":
    main()