import random
import gym
from gym import spaces
import numpy as np


class CognitiveUser(gym.Env):

    def __init__(self, config):
        self.L = config.L
        self.max_iterations = config.env_max_iterations
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(2)
        self.states = config.states
        self.counter = 0
        self.transmissions = 0
        self.collisions = 0
        self.n_rnt = 0
        self.n_rp = 0
        self.current_state = 0

        self.P = config.P
        # reward
        self.rnt = config.rnt  # no transmission reward when the primary is not transmitting
        self.rt = config.rt  # transmitting when primary not transmitting
        self.rp = config.rp  # not transmitting when primary transmitting
        self.rc = config.rc  # transmitting when primary transmitting -> collision
        self.instant_reward = config.instant_reward

    def reset(self, initial_state=None):
        if initial_state is not None:
            self.current_state = initial_state
        else:
            self.current_state = self.observation_space.sample()
        self.counter = 0
        self.transmissions = 0
        self.collisions = 0
        self.n_rnt = 0
        self.n_rp = 0
        return self.current_state

    def step(self, action):

        done = False
        ## Not Transmitting
        if action == 0:
            next_state = random.choices(self.states,    #self.action_space.n * self.current_state + action --- previous version
                                        weights=self.P[self.action_space.n * self.current_state, :], k=1)[0]
            reward = self.instant_reward[self.action_space.n * self.current_state + action, next_state]
            if reward == self.rp:
                self.n_rp += 1
            elif reward == self.rnt:
                self.n_rnt += 1
            self.current_state = next_state

        ## Transmitting
        else:  # when transmitting, check that during the whole transmission there is no collision
            collision_detected = False
            transmission_ended = False
            transmission_duration = 0

            while collision_detected == False and transmission_ended == False:
                next_state = random.choices(self.states,
                                            weights=self.P[self.action_space.n * self.current_state, :], k=1)[0]
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

                self.current_state = next_state

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
