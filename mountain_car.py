import gym
import math
import numpy as np
import random
import matplotlib.pyplot as plt

class Policy:
    def __init__(self):
        self.left_q = np.zeros((6, 4))
        self.right_q = np.zeros((6, 4))
        self.x_buckets = [-0.9, -0.6, -0.3, 0, 0.3, 0.6]
        self.velocity_buckets = [-0.035, 0, 0.035, 0.07]
        self.explore_rate = 0.2
        self.discount_rate = 0.95
        self.exploration_decay = 0.999
    def update_q(self, state, next_state, move, reward):
        if move == 0:
            self.left_q[state[0]][state[1]] += 0.1 * (reward + self.discount_rate * self.max_q(next_state) - self.left_q[state[0]][state[1]])
        else:
            self.right_q[state[0]][state[1]] += 0.1 * (reward + self.discount_rate * self.max_q(next_state) - self.right_q[state[0]][state[1]])
        self.explore_rate *= self.exploration_decay
            
    def determine_move(self, state):
        options =  [(0, self.left_q[state[0]][state[1]]), (2, self.right_q[state[0]][state[1]])]
        options.sort(key = lambda x: x[1], reverse = True)
        return options[0][0]
    
    def explore_move(self):
        if random.random() > 0.5:
            return 0
        return 2
    
    def max_q(self, state):
        return max(self.left_q[state[0]][state[1]], self.right_q[state[0]][state[1]])

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
    for t in range(200):
        env.render()
        if random.random() < policy.explore_rate:
            action = policy.explore_move()
        else:
            action = policy.determine_move(state0)
        observation, reward, done, info = env.step(action)
        state1 = policy.process_input(observation)
        print(reward)
        if done:
            policy.update_q(state0, state1, action, 1)
            print("Episode finished after {} timesteps".format(t+1))
            results.append(t+1)
            break
        else:
            policy.update_q(state0, state1, action, -1)
        state0 = state1

x = [i for i in range(400)]
plt.plot(x, results)
plt.show()
