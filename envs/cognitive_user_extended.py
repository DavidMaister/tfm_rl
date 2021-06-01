import math
import random
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import config


class CognitiveUserExtended(gym.Env):

    def __init__(self, L=config.L, max_iterations=config.env_max_iterations, length_state=config.length_state):
        self.action_space = spaces.Discrete(2)
        self.length_state = length_state
        self.observation_space = spaces.Box(np.array([0, 0]), np.array([1, self.length_state - 1]), dtype=np.int)
        self.states = config.states
        self.counter = 0
        self.transmissions = 0
        self.collisions = 0
        self.max_iterations = max_iterations
        self.current_state = np.array([0, 0])

        self.L = L  # frame duration
        self.P = config.P
        # reward
        self.rnt = config.rnt  # no transmission reward when the primary is not transmitting
        self.rt = config.rt  # transmitting when primary not transmitting
        self.rp = config.rp  # not transmitting when primary transmitting
        self.rc = config.rc  # transmitting when primary transmitting -> collision
        self.instant_reward = config.instant_reward


    def reset(self, initial_state = None):
        if initial_state is not None:
            self.current_state = np.array([initial_state, 0])
        else:
            self.current_state = self.observation_space.sample()
            self.current_state[1] = 0      # set the duration of the state to '0'

        self.counter = 0
        self.transmissions = 0
        self.collisions = 0
        return self.current_state

    def step(self, action):

        done = False
        # Not Transmitting #
        if action == 0:
            # next_state = random.choices(self.states,
            #                           weights=self.P[self.action_space.n * self.current_state[0] + action, :], k=1)[0]

            # multiply the probability p00 or p11 by the factor p00^i or p11^i, and the other probability is 1-P
            prob_stay_same_state = self.P[self.action_space.n * self.current_state[0], self.current_state[0]]
            next_state_factor = prob_stay_same_state * prob_stay_same_state ** self.current_state[1]
            next_state_probs = [next_state_factor, 1 - next_state_factor]
            # flip the probs if the next state is 1
            if self.current_state[0] == 1:
                next_state_probs = np.flip(next_state_probs)
            next_state = random.choices(self.states, weights=next_state_probs, k=1)[0]  # get the next state
            reward = self.instant_reward[self.action_space.n * self.current_state[0] + action, next_state]
            if self.current_state[0] == next_state:
                if self.current_state[1] < self.length_state - 1:
                    self.current_state[1] += 1
            else:
                self.current_state[1] = 0
            self.current_state[0] = next_state

        # Transmitting #
        else:  # when transmitting, check that during the whole transmission there is no collision
            collision_detected = False
            transmission_ended = False
            transmission_duration = 0
            first_slot = True

            while collision_detected == False and transmission_ended == False:
                if first_slot:
                    prob_stay_same_state = self.P[self.action_space.n * self.current_state[0], self.current_state[0]]
                    next_state_factor = prob_stay_same_state * prob_stay_same_state ** self.current_state[1]
                    next_state_probs = [next_state_factor, 1 - next_state_factor]
                    # flip the probs if the next state is 1
                    if self.current_state[0] == 1:
                        next_state_probs = np.flip(next_state_probs)
                    first_slot = False
                else:
                    # if starts from 0
                    # next_state_probs[0] *= config.p00
                    # next_state_probs = [next_state_probs[0], 1 - next_state_probs[0]]
                    next_state_probs = [config.p00, 1 - config.p00]
                    #print(self.current_state, next_state_probs)
                #print(self.current_state, next_state_probs, transmission_duration)
                next_state = random.choices(self.states, weights=next_state_probs, k=1)[0]  # get the next state
                if next_state == 0:  # no collision detected
                    transmission_duration += 1
                    if transmission_duration == self.L + 1:
                        reward = self.rt    # Successful transmission reward
                        self.transmissions += 1
                        transmission_ended = True

                else:  # collision detected
                    reward = self.rc    # Collision reward
                    self.collisions += 1
                    collision_detected = True

                if self.current_state[0] == next_state:
                    if self.current_state[1] < self.length_state - 1:
                        self.current_state[1] += 1
                else:
                    self.current_state[1] = 0
                self.current_state[0] = next_state

        self.counter += 1
        if self.counter == self.max_iterations:
            done = True
        else:
            done = False
        return self.current_state, reward, done, {}

    def render(self):
        print('Current state', self.current_state)

    def get_stats(self):
        return self.counter, self.transmissions, self.collisions
