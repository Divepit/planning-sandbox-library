import os
import signal
import torch
import torch.cuda.amp as amp
import numpy as np
from RL_mover_env import RLEnv
from planning_sandbox.environment_class import Environment
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.utils import get_schedule_fn
import resource

# Try to increase the limit of open files
soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
try:
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
    print(f"Increased file limit from {soft} to {hard}")
except ValueError:
    print(f"Unable to increase file limit. Current limit: {soft}")

# Function to determine safe number of environments
def get_safe_num_envs():
    soft, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
    return min(1024, max(64, soft // 4))  # Use at most 1/4 of available file descriptors, between 64 and 1024

class EarlyStoppingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(EarlyStoppingCallback, self).__init__(verbose)
        self.should_stop = False
        signal.signal(signal.SIGINT, self.signal_handler)

    def signal_handler(self, sig, frame):
        print("\nEarly stopping initiated. Completing the current update...")
        self.should_stop = True

    def _on_step(self) -> bool:
        return not self.should_stop

class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []

    def _on_step(self) -> bool:
        for info in self.locals['infos']:
            if 'episode' in info:
                self.episode_rewards.append(info['episode']['r'])
        return True

def make_env(rank, num_agents, num_goals, num_obstacles, width, height, num_skills, seed=0):
    def _init():
        sandbox_env = Environment(width=width, height=height, num_agents=num_agents, num_goals=num_goals, num_obstacles=num_obstacles, num_skills=num_skills)
        env = RLEnv(env=sandbox_env)
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init

if __name__ == "__main__":
    # CUDA settings
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Environment parameters
    num_agents = 3
    num_goals = 3
    num_obstacles = 0
    width = 8
    height = 8
    num_skills = 2
    max_steps = width * height

    # Determine safe number of environments
    num_envs = get_safe_num_envs()
    print(f"Using {num_envs} environments")

    env = SubprocVecEnv([make_env(rank=i, num_agents=num_agents, num_goals=num_goals, num_obstacles=num_obstacles, width=width, height=height, num_skills=num_skills) for i in range(num_envs)])
    env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_obs=10.)

    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)

    # PPO model setup with optimizations
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=2048,
        batch_size=min(65536, num_envs * max_steps),  # Adjust batch size based on num_envs
        n_epochs=10,
        learning_rate=1e-4,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs = dict(
            net_arch=dict(pi=[512, 512], vf=[512, 512]),
            activation_fn=torch.nn.ReLU
        ),
        device=device,
        tensorboard_log=log_dir,
    )

    # Enable mixed precision training
    scaler = amp.GradScaler()

    # Modify the PPO's _train_step method to use mixed precision
    def mixed_precision_train_step(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Record original train step method
        original_train_step = self._train_step

        # Define new train step method with mixed precision
        def new_train_step(gradient_steps: int, batch_size: int = 64) -> None:
            for _ in range(gradient_steps):
                with amp.autocast():
                    loss = original_train_step(1, batch_size)
                scaler.scale(loss).backward()
                scaler.step(self.policy.optimizer)
                scaler.update()

        # Replace train step method
        self._train_step = new_train_step

    # Apply mixed precision to model
    mixed_precision_train_step(model)

    total_timesteps = 100000000  # Increased total timesteps

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[EarlyStoppingCallback(), TensorboardCallback()],
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving the model...")
    except Exception as e:
        print(f"An error occurred during training: {e}")
    finally:
        model.save("ppo_custom_env_optimized_gpu")
        print("Training completed and model saved.")