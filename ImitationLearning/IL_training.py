import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import signal
import torch
import logging
import pickle
import numpy as np

from ImitationLearning.IL_env import ILEnv
from planning_sandbox.environment_class import Environment
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
from stable_baselines3 import PPO

from imitation.algorithms.bc import BC
from imitation.data.types import Transitions

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
        logging.debug("\nEarly stopping initiated. Completing the current update...")
        self.should_stop = True

    def _on_step(self) -> bool:
        return not self.should_stop

class TensorboardCallback(BaseCallback):
    def __init__(self, save_freq=200000, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.save_freq = save_freq
        self.last_save = 0

    def _on_step(self) -> bool:
        for info in self.locals['infos']:
            if 'episode' in info:
                self.episode_rewards.append(info['episode']['r'])
                self.logger.record("env/claimed_goals", info['episode']['claimed_goals'])
                self.logger.record("env/invalid_actions", info['episode']['invalid_actions'])
                self.logger.record("env/stay_actions", info['episode']['stay_actions'])
                self.logger.record("env/goal_stay_actions", info['episode']['goal_stay_actions'])
        
        # Check if it's time to save the model
        if self.num_timesteps - self.last_save >= self.save_freq:
            model_path = f"ppo_custom_env_improved_goal_assignment_intermediate"
            self.model.save(model_path)
            logging.info(f"Model saved at {model_path}")
            self.last_save = self.num_timesteps

        return True

def make_env(rank, num_agents, num_goals, size, num_skills, seed=0):
    def _init():
        sandbox_env = Environment(size=size, num_agents=num_agents, num_goals=num_goals, num_skills=num_skills, use_geo_data=True)
        env = ILEnv(env=sandbox_env)
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    num_envs = 12
    logging.info(f"Number of environments: {num_envs}")

    num_agents = 1
    num_goals = 2
    size = 32
    num_skills = 1
    max_steps = size*3



    log_dir = "/Users/marco/Programming/PlanningEnvironmentLibrary/ImitationLearning/logs"
    os.makedirs(log_dir, exist_ok=True)

    pkl_save_file = '/Users/marco/Programming/PlanningEnvironmentLibrary/ImitationLearning/expert_data.pkl'
    os.makedirs(os.path.dirname(pkl_save_file), exist_ok=True)

    with open(pkl_save_file, 'rb') as f:
        data = pickle.load(f)

    raw_observations = data['observations']
    raw_actions = data['actions']
    
    # Assuming raw_observations and raw_actions are already available
    expert_observations = np.array(raw_observations)
    expert_actions = np.array(raw_actions)

    # Generating next_obs as a shifted version of obs (or a valid placeholder if needed)
    next_observations = np.roll(expert_observations, -1, axis=0)  # Shift observations by 1 for next_obs

    # Assuming you want to end the episode at the last transition, generate `dones`
    dones = np.array([False] * (len(expert_observations) - 1) + [True])  # Convert list to NumPy array

    # Now create the Transitions object
    transitions = Transitions(
        obs=expert_observations,
        acts=expert_actions,
        infos=[None for _ in expert_observations],
        next_obs=next_observations,  # Provide valid next_obs here
        dones=dones  # NumPy array for `dones`
    )

    env = SubprocVecEnv([make_env(rank=i, num_agents=num_agents, num_goals=num_goals, size=size, num_skills=num_skills) for i in range(num_envs)])
    env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_obs=10.)

    rng = np.random.default_rng(0)
    bc_trainer = BC(
    observation_space=env.observation_space,
    action_space=env.action_space,
    demonstrations=transitions,
    rng=rng,
    )

    bc_trainer.train(n_epochs=100)
    bc_trainer.policy.save("/Users/marco/Programming/PlanningEnvironmentLibrary/ImitationLearning/bc_policy")

    obs = env.reset()
    action, _ = bc_trainer.policy.predict(obs)
    obs, reward, done, info = env.step(action)
    env.render()  # Optional: visualize the policy in action

    # model = PPO(
    #     "MlpPolicy",
    #     env,
    #     verbose=1,
    #     n_steps=max_steps,
    #     batch_size=max_steps*num_envs,
    #     n_epochs=24,
    #     learning_rate=linear_schedule(0.000075),
    #     clip_range=0.8,
    #     ent_coef=0.025,
    #     policy_kwargs = dict(
    #         net_arch=dict(pi=[128, 64, 32], vf=[64, 32]),
    #         activation_fn=torch.nn.ReLU
    #     ),
    #     device=device,
    #     tensorboard_log=log_dir,
    # )

    # total_timesteps = 500000000

    # try:
    #     model.learn(
    #         total_timesteps=total_timesteps,
    #         callback=[EarlyStoppingCallback(), TensorboardCallback(save_freq=10000)],
    #         progress_bar=True
    #     )
    # except KeyboardInterrupt:
    #     logging.info("\nTraining interrupted. Saving the model...")
    # except Exception as e:
    #     logging.error(f"An error occurred during training: {e}")
    # finally:
    #     model.save("ppo_custom_env_improved_goal_assignment")
    #     logging.info("Training completed and model saved.")