import os
import numpy as np
import parl
from parl.utils import action_mapping, ReplayMemory
import pandas as pd
from parl.utils import logger
from stockenv import StockTradingEnv
from mymodel import WudigupiaoModel
from myAgent import MyStockAgent
# from parl.algorithms import TD3
from parl.algorithms import DDPG
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

GAMMA = 0.99  # reward 的衰减因子，一般取 0.9 到 0.999 不等
TAU = 0.001  # target_model 跟 model 同步参数 的 软更新参数
ACTOR_LR = 3e-4  # Actor网络更新的 learning rate
CRITIC_LR = 3e-4  # Critic网络更新的 learning rate
MEMORY_SIZE = 1e6  # replay memory的大小，越大越占用内存
MEMORY_WARMUP_SIZE = 1e4  # replay_memory 里需要预存一些经验数据，再从里面sample一个batch的经验让agent去learn
REWARD_SCALE = 0.01  # reward 的缩放因子
BATCH_SIZE = 256  # 每次给agent learn的数据数量，从replay memory随机里sample一批数据出来
TRAIN_TOTAL_STEPS = 1e6  # 总训练步数
TEST_EVERY_STEPS = 1e4  # 每个N步评估一下算法效果，每次评估5个episode求平均reward


def run_episode(env, agent, rpm):
    obs = env.reset()
    total_reward, steps = 0, 0
    while True:
        steps += 1
        batch_obs = np.expand_dims(obs, axis=0)
        action = agent.predict(batch_obs.astype('float32'))
        action = np.squeeze(action)
        # Add exploration noise, and clip to [-1.0, 1.0]
        action = np.clip(np.random.normal(action, 1.0), -1.0, 1.0)
        action = action_mapping(action, env.action_space.low[0],
                                env.action_space.high[0])

        next_obs, reward, done, info = env.step(action)
        rpm.append(obs, action, REWARD_SCALE * reward, next_obs, done)

        if rpm.size() > MEMORY_WARMUP_SIZE:
            batch_obs, batch_action, batch_reward, batch_next_obs, \
                    batch_terminal = rpm.sample_batch(BATCH_SIZE)
            # critic_cost = agent.learn(batch_obs, batch_action, batch_reward,
            #                           batch_next_obs, batch_terminal)

        obs = next_obs
        total_reward += reward

        if done:
            break
    return total_reward, steps


# 评估 agent, 跑 5 个episode，总reward求平均
def evaluate(env, agent, render=False, evaldraw=False):
    eval_reward = []
    data_info = []
    for i in range(5):
        obs = env.reset()
        total_reward, steps = 0, 0
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
            steps += 1

            if render:
                reward_info = env.render(Opener=render)
                data_info.append(reward_info)
            
            if done:
                break
        eval_reward.append(total_reward)
    if evaldraw:
        return data_info
    
    else:
        return np.mean(eval_reward)

def main(train=True):
    if train:
        # 加载数据
        df = pd.read_csv('wudigushi/DATA/AAPL.csv')
        df = df.sort_values('Date')
        # 创建环境
        env = StockTradingEnv(df)
        env.reset()
        act_dim = env.action_space.shape[0]
        obs_dim = env.observation_space.shape[0]
        
        print(act_dim)
        print(obs_dim)
        
        model = WudigupiaoModel(act_dim)
        algorithm = DDPG(
            model, gamma=GAMMA, tau=TAU, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR)
        agent = MyStockAgent(algorithm, obs_dim, act_dim)
        

        rpm = ReplayMemory(int(MEMORY_SIZE), obs_dim , act_dim)

        

        test_flag = 0
        total_steps = 0
        while total_steps < TRAIN_TOTAL_STEPS:
            train_reward, steps = run_episode(env, agent, rpm)
            total_steps += steps
            # logger.info('Steps: {} Reward: {}'.format(total_steps, train_reward))

            if total_steps // TEST_EVERY_STEPS >= test_flag:
                # print('s1')
                while total_steps // TEST_EVERY_STEPS >= test_flag:
                    test_flag += 1

                evaluate_reward = evaluate(env, agent)
                logger.info('Steps {}, Test reward: {}'.format(total_steps,
                                                        evaluate_reward))


                

                ckpt = 'wudigushi/ckpt/steps_{}.ckpt'.format(total_steps)
                agent.save(ckpt)
    else:
        ckpt = 'wudigushi/ckpt/steps_980117.ckpt'  # 请设置ckpt为你训练中效果最好的一次评估保存的模型文件名称
        df = pd.read_csv('wudigushi/DATA/AAPL.csv')
        df = df.sort_values('Date')
        # 创建环境
        env = StockTradingEnv(df)
        env.reset()
        act_dim = env.action_space.shape[0]
        obs_dim = env.observation_space.shape[0]
        # obs_dim = 36
        print(act_dim)
        print(obs_dim)
        
        model = WudigupiaoModel(act_dim)
        algorithm = DDPG(
            model, gamma=GAMMA, tau=TAU, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR)
        agent = MyStockAgent(algorithm, obs_dim, act_dim)
        agent.restore(ckpt)
        evaluate_reward = evaluate(env, agent)
        logger.info('Evaluate reward: {}'.format(evaluate_reward)) # 打印评估的reward

if __name__ == '__main__':

    main(train=True)




