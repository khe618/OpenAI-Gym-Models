import gym
import math
import numpy as np
import random
import matplotlib.pyplot as plt

class Policy:
    def __init__(self):
        self.left_q = np.zeros((6, 3, 3))
        self.right_q = np.zeros((6, 3, 3))
        self.x_buckets = [-0.8, 0.8, 2.5]
        self.theta_buckets = [-2*math.pi/45, math.pi/-45, 0, math.pi/45, 2*math.pi/45,math.pi]
        self.theta_dot_buckets = [math.pi/-6, math.pi/6, math.pi / 2]
        self.explore_rate = 0.2
        self.discount_rate = 0.95
        self.exploration_decay = 0.999
    def update_q(self, state, next_state, move, reward):
        if move == 0:
            self.left_q[state[0]][state[1]][state[2]] += 0.1 * (reward + self.discount_rate * self.max_q(next_state) - self.left_q[state[0]][state[1]][state[2]])
        elif move == 1:
            self.right_q[state[0]][state[1]][state[2]] += 0.1 * (reward + self.discount_rate * self.max_q(next_state) - self.right_q[state[0]][state[1]][state[2]])
        self.explore_rate *= self.exploration_decay
            
    def determine_move(self, state):
        options =  [(0, self.left_q[state[0]][state[1]][state[2]]), (1, self.right_q[state[0]][state[1]][state[2]])]
        options.sort(key = lambda x: x[1], reverse = True)
        return options[0][0]
    
    def explore_move(self):
        return env.action_space.sample()
    
    def max_q(self, state):
        return max(self.left_q[state[0]][state[1]][state[2]], self.right_q[state[0]][state[1]][state[2]])

    def process_input(self, theta, theta_dot, x):
        theta_dot = math.tanh(theta_dot)
        for i in range(len(self.theta_buckets)):
            if theta <= self.theta_buckets[i]:
                state1 = i
                break
        for i in range(len(self.theta_dot_buckets)):
            if theta_dot <= self.theta_dot_buckets[i]:
                state2 = i
                break
        for i in range(len(self.x_buckets)):
            if x <= self.x_buckets[i]:
                state3 = i
                break
        return (state1, state2, state3)
        
policy = Policy()
results = []
env = gym.make('CartPole-v0')
for i_episode in range(400):
    observation = env.reset()
    theta = observation[2]
    theta_dot = observation[3]
    x = observation[0]
    state0 = policy.process_input(theta, theta_dot, x)
    for t in range(200):
        env.render()
        print(observation)
        if random.random() < policy.explore_rate:
            action = policy.explore_move()
        else:
            action = policy.determine_move(state0)
        observation, reward, done, info = env.step(action)
        theta = observation[2]
        theta_dot = observation[3]
        x = observation[0]
        state1 = policy.process_input(theta, theta_dot, x)
        if done:
            policy.update_q(state0, state1, action, -1)
            print("Episode finished after {} timesteps".format(t+1))
            results.append(t+1)
            break
        else:
            policy.update_q(state0, state1, action, 0)
        state0 = state1

x = [i for i in range(400)]
plt.plot(x, results)
plt.show()
