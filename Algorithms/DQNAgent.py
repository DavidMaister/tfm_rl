import random
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.optimizers import Adam
from collections import deque
import numpy as np


# Define the agent that is going to be used for training
class DoubleDQNAgent:
    def __init__(self, state_size, action_size, config, model=''):
        # Define state and action space sizes
        self.decay = config.epsilon_decay_dqn
        self.state_size = state_size
        self.encoded = to_categorical(np.array([0, 1]))
        self.action_size = action_size
        # Hyper-parameters for the DQN architecture
        self.discount_factor = config.gamma     # Discount factor for Bellman equation
        self.learning_rate = config.learning_rate_dqn   # Learning rate for ADAM optimizer
        self.epsilon = 1.0      # Initial epsilon value (for epsilon greedy policy)
        self.epsilon_decay = config.epsilon_decay_dqn      # Epsilon decay (for epsilon greedy policy)
        self.epsilon_min = 0.01     # Minimal epsilon value (for epsilon greedy policy)
        self.batch_size = 64    # Batch size for replay
        self.train_start = 1000     # Adds a delay, for the memory to have data before starting the training
        # Create a replay memory using deque
        self.memory = deque(maxlen=5000)
        # it can be specified the trained model so we instantly load it
        if model:
            self.model = model
        else:   # when it needs to be trained
            # create main model and target model
            self.model = self.build_model()
            self.target_model = self.build_model()  # The target model is a NN used to increase stability
            # Initialize target model
            self.update_target_model()

    # NN input is the state, output is the estimated Q value for each action
    def build_model(self):
        # We build a model with 3 layers
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(24, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='linear', kernel_initializer='he_uniform'))
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
        update_input = np.zeros((batch_size, self.state_size))
        update_target = np.zeros((batch_size, self.state_size))
        action, reward, done = [], [], []

        for i in range(batch_size):
            update_input[i] = mini_batch[i][0]
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            update_target[i] = mini_batch[i][3]
            done.append(mini_batch[i][4])
        # Obtain the targets for the NN training phase
        target = self.model.predict(update_input)
        target_next = self.model.predict(update_target)
        target_val = self.target_model.predict(update_target) # Use the target network HERE for further stability

        for i in range(self.batch_size):
            # Get maximum Q value at s' from target model
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                a = np.argmax(target_next[i])
                target[i][action[i]] = reward[i] + self.discount_factor * (target_val[i][a])
        # Fit the model!
        self.model.fit(update_input, target, batch_size=self.batch_size, epochs=1, verbose=0)