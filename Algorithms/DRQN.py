# Code adapted from https://github.com/rlcode/reinforcement-learning/tree/master/2-cartpole
import random
import gym
import numpy as np
from collections import deque
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import sys
import config
import time
from Algorithms.algorithms import encode_input_nn

# Define the agent that is going to be used for training
class DRQNAgent:
    def __init__(self, state_features, action_size, trace_length, model='', gamma=config.gamma,
                 learning_rate=config.learning_rate_dqn, decay=config.epsilon_decay_dqn):
        # Define state and action space sizes
        self.state_features = state_features
        self.action_size = action_size
        self.trace_length=trace_length
        # Hyper-parameters for the Double-DQN architecture
        self.discount_factor = gamma # Discount factor for Bellman equation
        self.learning_rate = learning_rate # Learning rate for ADAM optimizer
        self.epsilon = 1.0 # Initial epsilon value (for epsilon greedy policy)
        self.epsilon_decay = decay # Epsilon decay (for epsilon greedy policy)
        self.epsilon_min = 0.01 # Minimal epsilon value (for epsilon greedy policy)
        self.batch_size = 64 # Batch size for replay
        self.train_start = 1000 # Adds a delay, for the memory to have data before starting the training
        # Create a replay memory using deque
        self.memory = deque(maxlen=2000)
        # create main model and target model
        if model:
            self.model = model
        else:  # when it needs to be trained
            # create main model and target model
            self.model = self.build_model()
            self.target_model = self.build_model() # The target model is a NN used to increase stability
            # Initialize target model
            self.update_target_model()

    # NN input is the state, output is the estimated Q value for each action
    def build_model(self):
        # We build a model with 3 layers
        model = Sequential()
        #model.add(TimeDistributed(Dense(24, activation='relu',kernel_initializer='he_uniform'),batch_input_shape=(1,self.trace_length,self.state_features)))
        model.add(LSTM(32, input_shape=(self.trace_length,self.state_features)))
        model.add(Dense(self.action_size, activation='linear'))
        #model.summary() # Uncomment to see the model summary provided by Keras
        # Compile the model: use Mean Squared Error as loss function, ADAM as optimizer
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    # Function to update the target model
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # Epsilon greedy policy 
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(state)
            return np.argmax(q_value[0])

    # Save sample <s,a,r,s'> to the replay memory
    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train_model(self):
        if len(self.memory) < self.train_start:
            return # Start training only when there are some samples in the memory
        # Pick samples randomly from replay memory (with batch_size)
        batch_size = min(self.batch_size, len(self.memory))
        mini_batch = random.sample(self.memory, batch_size)
        # Preprocess the batch by storing the data in different vectors
        update_input = np.zeros([batch_size, self.trace_length, self.state_features]) 
        update_target = np.zeros([batch_size, self.trace_length, self.state_features]) 
        action, reward, done = [], [], []

        for i in range(batch_size):
            update_input[i] = mini_batch[i][0]
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            update_target[i] = mini_batch[i][3]
            done.append(mini_batch[i][4])
        # Obtain the targets for the NN training phase
        target = self.model.predict(update_input)
        target_val = self.target_model.predict(update_target) # Use the target network HERE for further stability

        for i in range(self.batch_size):
            # Get maximum Q value at s' from target model
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                target[i][action[i]] = reward[i] + self.discount_factor * np.amax(target_val[i])
        # Fit the model!
        self.model.reset_states() # This order is actually redundant: state is reset during training
        self.model.fit(update_input, target, batch_size=self.batch_size, epochs=1, verbose=0)

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


def drqn(env, algorithm, trace_length=config.trace_length, EPISODES=config.episodes_drqn):

    # Get size of state and action
    if algorithm == 'original':
        state_size = env.observation_space.n
    elif algorithm == 'expanded':
        state_size = env.observation_space.shape[0]

    action_size = env.action_space.n
     # Temporal dimension: length of sequence to feed the neural network
    # Create the agent
    agent = DRQNAgent(state_size, action_size, trace_length)
    scores, episodes, q_val = [], [], [] # To store values
    totals = np.array([])  # to store stats of the process
    transmissions = np.array([])
    collisions = np.array([])
    plot_flag = False  # To see the learning process (If set to False, it trains faster)

    for e in range(EPISODES):
        done = False
        tic = time.perf_counter()  # time the episode
        score = 0
        state = np.zeros([1, trace_length, state_size])
        state[0, -1, :] = encode_input_nn(algorithm, env.reset(), state_size) # Set the initial state
        q_val.append(max(agent.model.predict(state)[0]))
        agent.model.reset_states()
        while not done:  # Iterate while the game has not finished
            if plot_flag:
                env.render()
            # Get action for the current state and go one step in environment
            action = agent.get_action(state)  # Using epsilon-greedy policy
            next_statee, reward, done, info = env.step(action)  # Note that next_statee is used to update next_state!!
            next_state = np.zeros([1, trace_length, state_size])
            next_state[0, 0:trace_length-1, :] = state[0, 1:, :]
            next_state[0, -1, :]= encode_input_nn(algorithm, next_statee, state_size)
            # If an action makes the episode end before time (i.e, before 499 time steps), then give a penalty of -100
            score += reward * agent.discount_factor ** env.counter
            # Save the sample <s, a, r, s'> to the replay memory
            agent.append_sample(state, action, reward, next_state, done)
            # Train
            agent.train_model()
            state = next_state.copy()
            
            if done:
                # Update target model after each episode
                agent.update_target_model()
                # Store values for plotting
                scores.append(score)
                episodes.append(e)
                e_totals, e_transmissions, e_collisions = env.get_stats()
                totals = np.append(totals, e_totals)
                transmissions = np.append(transmissions, e_transmissions)
                collisions = np.append(collisions, e_collisions)
                toc = time.perf_counter()
                # Output the results of the episode
                if e % 5 == 0:
                    print("episode:", e, "  score:", score, "  time spent:", toc-tic, "  memory length:",
                          len(agent.memory), "  epsilon:", agent.epsilon, "  q_value:", q_val[-1])

    stats = [totals, transmissions, collisions]
    return agent, scores, stats


def drqn_evaluate(env, agent, algorithm, trace_length=config.trace_length, rep=config.drqn_evaluate_reps):
    # Get size of state and action
    if algorithm == 'original':
        state_size = env.observation_space.n
    elif algorithm == 'expanded':
        state_size = env.observation_space.shape[0]

    score_emp = []
    totals = np.array([])  # to store stats of the process
    transmissions = np.array([])
    collisions = np.array([])
    agent.epsilon = 0  # Use greedy policy!!
    for sim in range(rep):
        score = 0
        done = False
        state = np.zeros([1, trace_length, state_size])
        state[0, -1, :] = encode_input_nn(algorithm, env.reset(), state_size) # Set the initial state
        agent.model.reset_states() 
        sys.stdout.write('\r %s %f %%' % ('Simulating : ', sim/rep*100))
        sys.stdout.flush()
        while not done:  # Iterate while the game has not finished
            # Get action for the current state and go one step in environment
            action = agent.get_action(state) # Using epsilon-greedy policy
            next_statee, reward, done, info = env.step(action)
            next_state = np.zeros([1, trace_length, state_size])
            next_state[0, 0:trace_length-1, :] = state[0, 1:, :]
            next_state[0, -1, :] = encode_input_nn(algorithm, next_statee, state_size)
            score += reward * agent.discount_factor ** env.counter
            state = next_state.copy()

            if done:
                score_emp.append(score)
                e_totals, e_transmissions, e_collisions = env.get_stats()
                totals = np.append(totals, e_totals)
                transmissions = np.append(transmissions, e_transmissions)
                collisions = np.append(collisions, e_collisions)
    mean_score = np.mean(score_emp)
    stats = [score_emp, totals, transmissions, collisions]
    return mean_score, stats
