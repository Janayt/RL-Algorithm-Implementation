import numpy as np
import pandas as pd
import gym
import time
from gridworld import CliffWalkingWapper, FrozenLakeWapper


class QLearning(object):
    def __init__(self, num_states, num_actions, gamma, learning_rate, epsilon):
        self.actions = num_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q_table = np.zeros((num_states, num_actions))

    def choose_action(self, observation):
        """
        探索式采取动作
        :param observation:
        :return:
        """
        if np.random.uniform() < (1 - self.epsilon):
            action = self.predict(observation)
        else:
            action = np.random.choice(self.actions)
        return action

    def predict(self, observation):
        """
        可能存在多个最大的Q值
        :param observation:
        :return:
        """
        Q_list = self.Q_table[observation, :]
        maxQ = np.max(Q_list)
        action_list = np.where(Q_list == maxQ)[0]  # maxQ可能对应多个action
        action = np.random.choice(action_list)
        return action

    def learn(self, obs, action, reward, next_obs, done):
        predict_Q = self.Q_table[obs, action]
        if done:
            target_Q = reward
        else:
            target_Q = reward + self.gamma * np.max(self.Q_table[next_obs, :])
        self.Q_table[obs, action] += self.learning_rate * (target_Q - predict_Q)  ### update


def run(env, agent, render=False):
    total_reward = 0
    total_steps = 0
    obs = env.reset()

    while True:
        if render:
            env.render()
        action = agent.choose_action(obs)
        next_obs, reward, done, _ = env.step(action)
        agent.learn(obs, action, reward, next_obs, done)

        obs = next_obs
        total_reward += reward
        total_steps += 1

        if done:
            break
    return total_reward, total_steps

def test_episode(env, agent):
    total_reward = 0
    obs = env.reset()
    while True:
        action = agent.predict(obs) # greedy
        next_obs, reward, done, _ = env.step(action)
        total_reward += reward
        obs = next_obs
        time.sleep(0.5)
        env.render()
        if done:
            break
    return total_reward

if __name__ == "__main__":
    env = gym.make("CliffWalking-v0")
    env = CliffWalkingWapper(env)
    agent = QLearning(env.observation_space.n, env.action_space.n, learning_rate=0.1, gamma=0.9, epsilon=0.1)

    render = False
    for episode in range(500):
        episode_reward, episode_steps = run(env, agent, render=render)
        print('Episode %s: steps = %s , reward = %.1f' % (episode, episode_steps, episode_reward))
        # 每隔20个episode渲染一下看看效果
        if episode % 20 == 0:
            render = True
        else:
            render = False

    test_reward = test_episode(env, agent)
    print('test reward = %.1f' % (test_reward))