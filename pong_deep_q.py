# -*- coding: utf-8 -*-
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv3D, MaxPooling3D
from keras.optimizers import Adam

EPISODES = 1000


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Conv3D(32, kernel_size=(1, 5, 5), strides=(1, 1, 1),
                 activation='relu',
                 input_shape=self.state_size))
        model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))
        model.add(Conv3D(64, (1, 5, 5), activation='relu'))
        model.add(MaxPooling3D(pool_size=(1, 2, 2)))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

def process_image(arr):
    arr = arr[34:194]
    arr = arr.astype(float)
    new_arr = np.zeros((arr.shape[0], arr.shape[1]))
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            for k in range(arr.shape[2]):
                arr[i][j][k] /= 255.0
            new_arr[i][j] = arr[i][j][0] * 0.2126 + arr[i][j][1] * 0.7152 + arr[i][j][2] * 0.0722
    compressed_arr = np.zeros((int(new_arr.shape[0]/4), int(new_arr.shape[1]/4)))
    for m in range(0, new_arr.shape[0], 4):
        for n in range(0, new_arr.shape[1], 4):
            for i in range(m, m+4):
                for j in range(n, n+4):
                    compressed_arr[int(m/4)][int(n/4)] += new_arr[i][j]
            compressed_arr[int(m/4)][int(n/4)] /= 16.0        
            
    return compressed_arr        

if __name__ == "__main__":
    env = gym.make('Pong-v0')
    state_size = (2, 40, 40, 1)
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    agent.load("drive/app/cartpole-dqn.h5")
    done = False
    batch_size = 32
    image_size = (80, 80, 1)
    for e in range(EPISODES):
        state = env.reset()
        state = process_image(state)
        buffer = [state]
        for time in range(10000):
            #env.render()
            if len(buffer) > 1:
                action = agent.act(state)
            else:
                action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)
            next_state = process_image(next_state)
            buffer.append(next_state)
            next_state = np.reshape(buffer[-2:], (1,) + state_size)
            if len(buffer) > 2:
                agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, EPISODES, time, agent.epsilon))
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        if e % 10 == 0:
            agent.save("cartpole-dqn.h5")
