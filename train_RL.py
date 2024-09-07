import os
import signal
import torch
from RL_mover_env import RLEnv
from planning_sandbox.environment_class import Environment
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
from stable_baselines3 import PPO

def linear_schedule(initial_value: float):
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func



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
                self.logger.record("env/claimed_goals", info['episode']['claimed_goals']) 
                self.logger.record("env/invalid_actions", info['episode']['invalid_actions']) 
                self.logger.record("env/stay_actions", info['episode']['stay_actions'])
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    num_envs = 12
    print(f"Number of environments: {num_envs}")

    num_agents = 3
    num_goals = 5
    num_obstacles = 0
    width = 8
    height = 8
    num_skills = 2
    max_steps = width * height


    env = SubprocVecEnv([make_env(rank=i, num_agents=num_agents, num_goals=num_goals, num_obstacles=num_obstacles, width=width, height=height, num_skills=num_skills) for i in range(num_envs)])
    env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_obs=10.)

    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=max_steps,
        batch_size=max_steps*num_envs,
        n_epochs=24,
        learning_rate=linear_schedule(2.5e-4),
        clip_range=0.8,
        ent_coef=0.025,
        policy_kwargs = dict(
            net_arch=dict(pi=[512, 512, 512], vf=[256, 256]),
            activation_fn=torch.nn.ReLU
        ),
        device=device,
        tensorboard_log=log_dir,
    )

    total_timesteps = 50000000

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
        model.save("ppo_custom_env_improved_goal_assignment")
        print("Training completed and model saved.")