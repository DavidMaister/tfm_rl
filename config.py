#   Global variables for the program
import numpy as np
import math


class Config:
    # tuning parameters
    n_jobs = 1  # number of simultaneous threads !!always less than processor has!!!
    flag_train_all = True

    steps_vi = 200     # steps for value iteration

    # environment
    env_max_iterations = 50

    # Q learning
    epochs_ql = 100    # number of epochs for q_learning
    alpha_ql = 0.002    # learning rate
    epsilon_ql = 1      # initial value of epsilon
    epsilon_min_ql = 0.1
    epsilon_decay_ql = 0.9995    # epsilon decay each iteration

    # DQN
    flag_train_dqn = True
    episodes_dqn = 2      # number of episodes
    n_seeds = 2
    train_wait = 30   # train_wait = 10   # number of steps it waits to train the model
    learning_rate_dqn = 0.0001
    epsilon_decay_dqn = 0.999
    dqn_evaluate_reps = 2    # times the dqn model is evaluated to average the score afterwards

    # Evaluate policies
    rep_eval_pol = 3

    # DRQN
    trace_length = 5    # Temporal dimension: length of sequence to feed the neural network
    episodes_drqn = episodes_dqn
    drqn_evaluate_reps = dqn_evaluate_reps

    # original states
    N_states = 2
    N_actions = 2
    states = np.array([0, 1])
    actions = np.array([0, 1])

    def __init__(self):
        self.gamma = 0.99  # discount factor
        self.L = 2  # length of the frame
        self.lambda_on = 6
        self.lambda_off = 3
        self.ts = 0.1  # period of sample

        # reward
        self.rnt = -.5  # no transmission reward when the primary is not transmitting
        self.rt = 1  # transmitting when primary not transmitting
        self.rp = .5  # not transmitting when primary transmitting
        self.rc = -1.5  # transmitting when primary transmitting -> collision

        self.update()

    def update(self):
        # update the transition and rewards matrix
        self.exponent = math.exp(- (self.lambda_on + self.lambda_off) * self.ts)
        self.p00 = self.lambda_on / (self.lambda_on + self.lambda_off) + self.lambda_off / (
                    self.lambda_on + self.lambda_off) * self.exponent
        self.p01 = 1 - self.p00
        self.p11 = self.lambda_off / (self.lambda_on + self.lambda_off) + self.lambda_on / (
                    self.lambda_on + self.lambda_off) * self.exponent
        self.p10 = 1 - self.p11

        self.P = np.array([[self.p00, self.p01],
                           [self.p00 ** (self.L + 1), (1 - self.p00 ** (self.L + 1))],
                           [self.p10, self.p11],
                           [self.p10 * self.p00 ** self.L, (1 - self.p10 * self.p00 ** self.L)]])

        self.R = np.array(
            [self.rnt * self.p00 + self.rp * self.p01,
             self.rt * self.p00 ** (self.L + 1) + self.rc * (1 - self.p00 ** (self.L + 1)),
             self.rp * self.p11 + self.rnt * self.p10,
             self.rc * (1 - self.p10 * self.p00 ** self.L) + self.rt * self.p10 * self.p00 ** self.L])
        self.instant_reward = np.array([[self.rnt, self.rp],  # r(0,0,0) r(0,0,1)      # this is the reward of r(s,a,s')
                                        [self.rt, self.rc],  # r(0,1,0) r(0,1,1)
                                        [self.rnt, self.rp],  # r(1,0,0) r(1,0,1)
                                        [self.rt, self.rc]])  # r(1,1,0) r(1,1,1)