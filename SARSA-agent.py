import gym
import numpy as np
import time
from gridworld import CliffWalkingWapper

class SarsaAgent(object):
    def __init__(self, num_states, num_actions, gamma, learning_rate, epsilon):
        self.actions = num_actions
        self.lr = learning_rate
        self.epsilon = epsilon
        self.gamma = gamma
        self.Q_table = np.zeros((num_states, num_actions))

    def choose_action(self, observation):
        if np.random.uniform() < (1 - self.epsilon):
            action = self.predict(observation)
        else:
            action = np.random.choice(self.actions)
        return action

    def predict(self, observation):
        Q_list = self.Q_table[observation, :]
        maxQ = np.max(Q_list)
        action_list = np.where(Q_list == maxQ)[0]  # maxQ 可能对应多个action
        action = np.random.choice(action_list)
        return action

    def learn(self, observation, action, reward, next_obs, next_action, done):
        predict_Q = self.Q_table[observation, action]
        if done:
            target_Q = reward
        else:
            target_Q = reward + self.gamma * self.Q_table[next_obs, next_action]
        self.Q_table[observation, action] = self.Q_table[observation, action] + self.lr * (target_Q - predict_Q)

def run(env, agent, render=False):
    total_reward = 0
    total_steps = 0
    observation = env.reset()
    action = agent.choose_action(observation)
    while True:
        if render:
            env.render()
        next_obs, reward, done, _ = env.step(action)
        next_action = agent.choose_action(next_obs)
        agent.learn(observation, action, reward, next_obs, next_action, done)

        observation = next_obs
        action = next_action
        total_steps += 1
        total_reward += reward
        if done:
            break
    return total_reward, total_steps


def test_episode(env, agent):
    total_reward = 0
    observation = env.reset()
    while True:
        action = agent.choose_action(observation)
        next_obs, reward, done, _ = env.step(action)
        total_reward += reward
        observation = next_obs
        time.sleep(0.5)
        env.render()
        if done:
            print('test reward = %.1f' % (total_reward))
            break

if __name__ == "__main__":
    env = gym.make('CliffWalking-v0')
    env = CliffWalkingWapper(env)
    agent = SarsaAgent(num_states=env.observation_space.n,
                       num_actions=env.action_space.n,
                       learning_rate=0.1,
                       gamma=0.9,
                       epsilon=0.1)
    render = False
    for episode in range(500):
        episode_reward, episode_steps = run(env, agent, render=render)
        print('Episode {}s: steps: {}, reward: {}'.format(episode, episode_steps, episode_reward))
        if episode % 20 == 0:
            render = True
        else:
            render =False

    test_episode(env, agent)