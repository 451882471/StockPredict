import argparse
import gym
import numpy as np
import time
import parl
from TD3_Agent import MujocoAgent
from TD3_model import MujocoModel
from parl.utils import logger, summary, action_mapping, ReplayMemory
import pandas as pd
from StockTradingEnvss import StockTradingEnv

MAX_EPISODES = 5000
ACTOR_LR = 3e-4
CRITIC_LR = 3e-4
GAMMA = 0.9
TAU = 0.005
MEMORY_SIZE = int(1e6)
WARMUP_SIZE = 1e4
BATCH_SIZE = 256
ENV_SEED = 1
EXPL_NOISE = 0.1  # Std of Gaussian exploration noise


def run_train_episode(env, agent, rpm):
    obs = env.reset()
    total_reward = 0
    steps = 0
    max_action = float(env.action_space.high[0])
    while True:
        steps += 1
        batch_obs = np.expand_dims(obs, axis=0)

        if rpm.size() < WARMUP_SIZE:
            action = env.action_space.sample()
        else:
            action = agent.predict(batch_obs.astype('float32'))
            action = np.squeeze(action)

            # Add exploration noise, and clip to [-max_action, max_action]
            action = np.clip(
                np.random.normal(action, EXPL_NOISE * max_action), -max_action,
                max_action)

        next_obs, reward, done, info = env.step(action)

        rpm.append(obs, action, reward, next_obs, done)

        if rpm.size() > WARMUP_SIZE:
            batch_obs, batch_action, batch_reward, batch_next_obs, batch_terminal = rpm.sample_batch(
                BATCH_SIZE)
            agent.learn(batch_obs, batch_action, batch_reward, batch_next_obs,
                        batch_terminal)

        obs = next_obs
        total_reward += reward

        if done:
            break
    return total_reward, steps


def run_evaluate_episode(env, agent):
    obs = env.reset()
    total_reward = 0
    while True:
        batch_obs = np.expand_dims(obs, axis=0)
        action = agent.predict(batch_obs.astype('float32'))
        action = np.squeeze(action)
        action = np.clip(action, -1.0, 1.0)  ## special
        action = action_mapping(action, env.action_space.low[0],
                                env.action_space.high[0])

        next_obs, reward, done, info = env.step(action)

        obs = next_obs
        total_reward += reward

        if done:
            break
    return total_reward


def main():
    # 加载数据
    df = pd.read_csv('TD3gupiao/DATA/AAPL.csv')
    df = df.sort_values('Date')
    # 创建环境
    env = StockTradingEnv(df)
    env.reset()

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    model = MujocoModel(act_dim, max_action)
    algorithm = parl.algorithms.TD3(
        model,
        max_action=max_action,
        gamma=GAMMA,
        tau=TAU,
        actor_lr=ACTOR_LR,
        critic_lr=CRITIC_LR)
    agent = MujocoAgent(algorithm, obs_dim, act_dim)

    rpm = ReplayMemory(MEMORY_SIZE, obs_dim, act_dim)

    test_flag = 0
    total_steps = 0
    while total_steps < args.train_total_steps:
        train_reward, steps = run_train_episode(env, agent, rpm)
        total_steps += steps
        # logger.info('Steps: {} Reward: {}'.format(total_steps, train_reward))
        summary.add_scalar('train/episode_reward', train_reward, total_steps)

        if total_steps // args.test_every_steps >= test_flag:
            while total_steps // args.test_every_steps >= test_flag:
                test_flag += 1
            evaluate_reward = run_evaluate_episode(env, agent)
            logger.info('Steps {}, Evaluate reward: {}'.format(
                total_steps, evaluate_reward))
            summary.add_scalar('eval/episode_reward', evaluate_reward,
                               total_steps)
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     '--env', help='Mujoco environment name', default='HalfCheetah-v2')
    parser.add_argument(
        '--train_total_steps',
        type=int,
        default=int(1e6),
        help='maximum training steps')
    parser.add_argument(
        '--test_every_steps',
        type=int,
        default=int(1e4),
        help='the step interval between two consecutive evaluations')

    args = parser.parse_args()

    main()