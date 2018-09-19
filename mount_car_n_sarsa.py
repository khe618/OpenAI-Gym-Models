import gym
import math
import numpy as np
import random
import matplotlib.pyplot as plt

class Policy:
    def __init__(self):
        self.left_v = np.zeros((6, 4))
        self.right_v = np.zeros((6, 4))
        self.x_buckets = [-0.9, -0.6, -0.3, 0, 0.3, 0.6]
        self.velocity_buckets = [-0.035, 0, 0.035, 0.07]
        self.explore_rate = 0.1
        self.discount_rate = 0.95
        self.exploration_decay = 0.9995
        self.n_steps = 5
    def update(self, state, action, rewards, next_state, next_action):
        if next_action == 0:
            rewards.append(self.left_v[next_state[0]][next_state[1]])
        elif next_action == 2:
            rewards.append(self.right_v[next_state[0]][next_state[1]])
        reward = 0
        for i in range(len(rewards)):
            reward += rewards[i] * (self.discount_rate ** i)
        if action == 0:
            self.left_v[state[0]][state[1]] += 0.01 * (reward - self.left_v[state[0]][state[1]])
        else:
            self.right_v[state[0]][state[1]] += 0.01 * (reward - self.right_v[state[0]][state[1]])
        self.explore_rate *= self.exploration_decay
            
    def determine_move(self, state):
        options =  [(0, self.left_v[state[0]][state[1]]), (2, self.right_v[state[0]][state[1]])]
        options.sort(key = lambda x: x[1], reverse = True)
        return options[0][0]
    
    def explore_move(self):
        if random.random() > 0.5:
            return 0
        return 2

    def process_input(self, arr):
        x = observation[0]
        velocity = observation[1]
                                 
        for i in range(len(self.x_buckets)):
            if x <= self.x_buckets[i]:
                state1 = i
                break
        for i in range(len(self.velocity_buckets)):
            if velocity <= self.velocity_buckets[i]:
                state2 = i
                break
        return (state1, state2)
        
policy = Policy()
results = []
env = gym.make('MountainCar-v0')
for i_episode in range(400):
    observation = env.reset()
    state0 = policy.process_input(observation)
    rewards = []
    actions = []
    states = []
    for t in range(1000):
        env.render()
        print(observation)
        if random.random() < policy.explore_rate:
            action = policy.explore_move()
        else:
            action = policy.determine_move(state0)
        observation, reward, done, info = env.step(action)
        state1 = policy.process_input(observation)
        rewards.append(reward)
        states.append(state0)
        actions.append(action)
        if len(states) > policy.n_steps:
            policy.update(states[t-5], actions[t-5], rewards[-5:], state1, action)
        if done:
            for i in range(len(states)-4, len(states)):
                policy.update(states[i], actions[i], rewards[i:], None, None)
            print("Episode finished after {} timesteps".format(t+1))
            results.append(t+1)
            break
        state0 = state1

x = [i for i in range(400)]
plt.plot(x, results)
plt.show()
