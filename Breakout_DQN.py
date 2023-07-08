import random
import numpy as np
import pandas as pd
import gym
import time
from collections import deque
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt


class DQN:
    def __init__(self, env):
        # Initialize the DQN agent
        self.env = env
        self.memory = deque(maxlen=400000)  # Replay memory
        self.gamma = 0.9  # Discount factor for future rewards
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01  # Minimum exploration rate
        self.epsilon_decay = self.epsilon_min / 500000  # Decay rate for exploration
        
        self.batch_size = 16  # Batch size for training
        self.train_start = 1000  # Number of experiences required before starting training
        self.state_size = self.env.observation_space.shape[0] * 4  # Size of the state vector
        self.action_size = self.env.action_space.n  # Number of possible actions
        self.learning_rate = 0.0025  # Learning rate for the optimizer
        
        self.evaluation_model = self.create_model()  # Neural network for evaluation
        self.target_model = self.create_model()  # Neural network as a target for stable training
        
    def create_model(self):
        # Create a neural network model for the DQN
        model = Sequential()
        model.add(Dense(128 * 2, input_dim=self.state_size, activation='relu'))
        model.add(Dense(128 * 2, activation='relu'))
        model.add(Dense(128 * 2, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mean_squared_error', optimizer=optimizers.RMSprop(lr=self.learning_rate, decay=0.99, epsilon=1e-3))
        return model
    
    def choose_action(self, state, steps):
        # Choose an action using epsilon-greedy exploration strategy
        if steps > 50000:
            if self.epsilon > self.epsilon_min:
                self.epsilon -= self.epsilon_decay
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.evaluation_model.predict(state)[0])
        
    def remember(self, cur_state, action, reward, new_state, done):
        # Store the experience in the replay memory
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = (cur_state, action, reward, new_state, done)
        self.memory.extend([transition])
        self.memory_counter += 1
    
    def replay(self):
        # Train the DQN by replaying experiences from the replay memory
        if len(self.memory) < self.train_start:
            return
        mini_batch = random.sample(self.memory, self.batch_size)
        update_input = np.zeros((self.batch_size, self.state_size))
        update_target = np.zeros((self.batch_size, self.action_size))
        
        for i in range(self.batch_size):
            state, action, reward, new_state, done = mini_batch[i]
            target = self.evaluation_model.predict(state)[0]
            if done:
                target[action] = reward
            else:
                target[action] = reward + self.gamma * np.amax(self.target_model.predict(new_state)[0])
            update_input[i] = state
            update_target[i] = target
    
        self.evaluation_model.fit(update_input, update_target, batch_size=self.batch_size, epochs=1, verbose=0)
    
    def target_train(self):
        # Update the target model with the weights of the evaluation model
        self.target_model.set_weights(self.evaluation_model.get_weights())
        return
    
    def visualize(self, reward, episode):
        # Visualize the average reward per episode
        plt.plot(episode, reward, 'ob-')
        plt.title('Average reward each 100 episode')
        plt.ylabel('Reward')
        plt.xlabel('Episodes')
        plt.grid()
        plt.show()
    
    def transform(self, state):
        # Transform the state representation if necessary
        if state.shape[1] == 512:
            return state
        a = [np.binary_repr(x, width=8) for x in state[0]]
        res = []
        for x in a:
            res.extend([x[:2], x[2:4], x[4:6], x[6:]])
        res = [int(x, 2) for x in res]
        return np.array(res)


def main():
    # Initialize the environment
    env = gym.make('Breakout-ram-v0')
    env = env.unwrapped
    
    # Print environment information
    print(env.action_space)
    print(env.observation_space.shape[0])
    print(env.observation_space.high)
    print(env.observation_space.low)
    
    episodes = 100000
    trial_len = 10000
    
    tmp_reward = 0
    sum_rewards = 0
    total_steps = 0
    
    graph_reward = []
    graph_episodes = []
    time_record = []
    
    dqn_agent = DQN(env=env)
    
    for i_episode in range(episodes):
        start_time = time.time()
        total_reward = 0
        cur_state_tuple = env.reset()
        cur_state = np.array(cur_state_tuple[0]).reshape(1, 128)
        cur_state = dqn_agent.transform(cur_state).reshape(1, 128 * 4) / 4
        
        for _ in range(trial_len):
            # Choose action, take a step in the environment
            action = dqn_agent.choose_action(cur_state, total_steps)
            step_result = env.step(action)
            if len(step_result) == 5:
                new_state, reward, done, _, info = step_result
            else:
                new_state, reward, done, _, info = step_result + (None,) * (4 - len(step_result))
            
            new_state = new_state.reshape(1, 128)
            new_state = dqn_agent.transform(new_state).reshape(1, 128 * 4) / 4
            total_reward += reward
            sum_rewards += reward
            tmp_reward += reward
            if reward > 0:
                reward = 1  # Testing whether it is good.
            
            # Store the experience in the replay memory
            dqn_agent.remember(cur_state, action, reward, new_state, done)
            
            if total_steps > 10000:
                if total_steps % 4 == 0:
                    # Train the DQN by replaying experiences from the replay memory
                    dqn_agent.replay()
                if total_steps % 5000 == 0:
                    # Update the target model with the weights of the evaluation model
                    dqn_agent.target_train()
            
            cur_state = new_state
            total_steps += 1
            if done:
                env.reset()
                break
        
        if (i_episode + 1) % 100 == 0:
            graph_reward.append(sum_rewards / 100)
            graph_episodes.append(i_episode + 1)
            sum_rewards = 0
        
        end_time = time.time()
        time_record.append(end_time - start_time)
        tmp_reward = 0
    
    print("Reward: ")
    print(graph_reward)
    print("Episode: ")
    print(graph_episodes)
    print("Average_time: ")
    print(sum(time_record) / 5000)
    dqn_agent.visualize(graph_reward, graph_episodes)


main()    