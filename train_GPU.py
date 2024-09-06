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
from tqdm import tqdm

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
    return min(256, max(64, soft // 8))  # Reduced max environments to 256

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

class MixedPrecisionPPO(PPO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scaler = amp.GradScaler(enabled=(self.device.type == 'cuda'))

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, gym.spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                with amp.autocast(enabled=(self.device.type == 'cuda')):
                    values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                    values = values.flatten()
                    # Normalize advantage
                    advantages = rollout_data.advantages
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                    # ratio between old and new policy, should be one at the first iteration
                    ratio = torch.exp(log_prob - rollout_data.old_log_prob)

                    # clipped surrogate loss
                    policy_loss_1 = advantages * ratio
                    policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                    policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                    # Logging
                    pg_losses.append(policy_loss.item())
                    clip_fraction = torch.mean((torch.abs(ratio - 1) > clip_range).float()).item()
                    clip_fractions.append(clip_fraction)

                    if self.clip_range_vf is None:
                        # No clipping
                        values_pred = values
                    else:
                        # Clip the different between old and new value
                        # NOTE: this depends on the reward scaling
                        values_pred = rollout_data.old_values + torch.clamp(
                            values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                        )
                    # Value loss using the TD(gae_lambda) target
                    value_loss = torch.nn.functional.mse_loss(rollout_data.returns, values_pred)
                    value_losses.append(value_loss.item())

                    # Entropy loss favor exploration
                    if entropy is None:
                        # Approximate entropy when no analytical form
                        entropy_loss = -torch.mean(-log_prob)
                    else:
                        entropy_loss = -torch.mean(entropy)

                    entropy_losses.append(entropy_loss.item())

                    loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Optimization step
                self.policy.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                # Clip grad norm
                self.scaler.unscale_(self.policy.optimizer)
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.scaler.step(self.policy.optimizer)
                self.scaler.update()

                approx_kl_divs.append(torch.mean(rollout_data.old_log_prob - log_prob).detach().cpu().numpy())

            if self.target_kl is not None and np.mean(approx_kl_divs) > 1.5 * self.target_kl:
                continue_training = False
                print(f"Early stopping at step {epoch} due to reaching max kl: {np.mean(approx_kl_divs):.2f}")
                break

        self._n_updates += self.n_epochs
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", torch.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

    def learn(self, *args, **kwargs):
        return super().learn(*args, **kwargs)

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
    print(f"Initializing {num_envs} environments...")

    # Create environments with progress bar
    envs = []
    for i in tqdm(range(num_envs), desc="Creating environments"):
        envs.append(make_env(rank=i, num_agents=num_agents, num_goals=num_goals, num_obstacles=num_obstacles, width=width, height=height, num_skills=num_skills))

    print("Initializing SubprocVecEnv...")
    env = SubprocVecEnv(envs)
    print("Initializing VecNormalize...")
    env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_obs=10.)

    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)

    print("Setting up PPO model...")
    # PPO model setup with optimizations
    model = MixedPrecisionPPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=2048,
        batch_size=min(65536, num_envs * max_steps),  # Adjust batch size based on num_envs
        n_epochs=100,
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

    total_timesteps = 100000000  # Increased total timesteps

    print("Starting training...")
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