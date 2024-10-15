import os
import logging
import copy
import torch
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback,BaseCallback
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from IL_env import ILEnv
from planning_sandbox.environment_class import Environment

dir_path = os.path.dirname(os.path.realpath(__file__))
device = "cuda" if torch.cuda.is_available() else "mps"

def make_env(rank, template_env, seed=0):
    logging.info(f"Creating environment {rank+1}")
    def _init():
        sandbox_env = copy.deepcopy(template_env)
        env = ILEnv(sandboxEnv=sandbox_env)
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init

class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.episode_rewards = []

    def _on_step(self) -> bool:
        self.logger.dump(self.num_timesteps)
        for info in self.locals['infos']:
            if 'episode' in info:
                logging.debug(info['episode'])
                self.logger.record("env/ep_av_reward", info['episode']['r'])
                self.logger.record("env/ep_av_distributed_goals", info['episode']['distributed_goals'])
                self.logger.record("env/ep_av_unclaimed_goals", info['episode']['unclaimed_goals'])
                self.logger.record("env/ep_av_cost", info['episode']['cost'])
                self.logger.record("env/ep_av_length", info['episode']['episode_attempts'])
                self.logger.record("env/ep_av_deadlocks", info['episode']['deadlocks'])
                self.episode_rewards.append(info['episode']['r'])
            if 'terminal_observation' in info:
                logging.debug(info['terminal_observation'])
        return True
    
checkpoint_callback = CheckpointCallback(
  save_freq=200,
  save_path=dir_path+"/model_logs/",
  name_prefix="rl_model",
  save_replay_buffer=True,
  save_vecnormalize=True,
)

def main():

    logging.info("Creating template environment...")
    eval_sandbox_env: Environment = Environment(size=100, num_agents=3, num_goals=5, num_skills=2, use_geo_data=True)
    evalEnv = ILEnv(sandboxEnv=eval_sandbox_env)
    logging.info("Checking environment...")
    check_env(evalEnv, warn=True)

    n_envs = 12
    n_timesteps = 500000
    logging.info(f"Creating {n_envs} environments...")


    subproc_env = SubprocVecEnv([make_env(rank=i, template_env=eval_sandbox_env) for i in range(n_envs)])
    norm_env = VecNormalize(subproc_env, norm_obs=False, norm_obs_keys=["map_elevations"])

    policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                     net_arch=[64,64])

    model = A2C(
        "MultiInputPolicy",
        norm_env,
        n_steps=200,
        verbose=1,
        tensorboard_log=dir_path+"/tensorboard_logs/",
        device=device,
        policy_kwargs=policy_kwargs
        )

    logging.info(f"Training model for {n_timesteps} timesteps...")
    model.learn(
        total_timesteps=n_timesteps,
        progress_bar=True,
        callback=[checkpoint_callback, TensorboardCallback()],
        )


if __name__ == "__main__":
    main()