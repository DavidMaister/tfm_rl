#   Global variables for the program
import numpy as np
import math
import os
import datetime

# tuning parameters
n_jobs = 1  # number of simultaneous threads !!always less than processor has!!!

gamma = 0.99   # discount factor
L = 2  # length of the frame
lambda_on = 6
lambda_off = 3
ts = 0.1    # period of sample

steps_vi = 1200     # steps for value iteration

# reward
rnt = -.5  # no transmission reward when the primary is not transmitting
rt = 1  # transmitting when primary not transmitting
rp = .5  # not transmitting when primary transmitting
rc = -1.5  # transmitting when primary transmitting -> collision

# environment
env_max_iterations = 200

# Q learning
epochs_ql = 3000    # number of epochs for q_learning
alpha_ql = 0.002    # learning rate
epsilon_ql = 1      # initial value of epsilon
epsilon_min_ql = 0.1
epsilon_decay_ql = 0.9995    # epsilon decay each iteration

# DQN
flag_train_dqn = False
episodes_dqn = 2      # number of episodes
n_seeds = 1
train_wait = 10   # train_wait = 10   # number of steps it waits to train the model
learning_rate_dqn = 0.0001
epsilon_decay_dqn = 0.999
dqn_evaluate_reps = 2    # times the dqn model is evaluated to average the score afterwards

# Evaluate policies
rep_eval_pol = 1

# DRQN
trace_length = 5    # Temporal dimension: length of sequence to feed the neural network
episodes_drqn = 2
drqn_evaluate_reps = dqn_evaluate_reps


#####
# extended space state
length_state = 3   # max duration the environment can stay in the same state

epochs_ql_ext = 3500

# # create directories for weights and results
# date_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
# print(date_time)
# results_path = 'results/' + date_time + '/images/'
# if not os.path.exists(results_path):
#     os.makedirs(results_path)  # create directory if not exists
#
# weights_path = 'results/' + date_time + '/weights/'
# if not os.path.exists(weights_path):
#     os.makedirs(weights_path)  # create directory if not exists



#############
# probability transition matrix

# original states
N_states = 2
N_actions = 2
states = np.array([0, 1])
actions = np.array([0, 1])

exponent = math.exp(- (lambda_on + lambda_off) * ts)
p00 = lambda_on/(lambda_on+lambda_off) + lambda_off/(lambda_on+lambda_off)*exponent
p01 = 1 - p00
p11 = lambda_off/(lambda_on+lambda_off) + lambda_on/(lambda_on+lambda_off)*exponent
p10 = 1 - p11
#P = np.array([[p00, p01], [p10, p11]])
#P = np.repeat(P, 2, axis=0)     # we repeat the rows of the matrix because it is not dependent on the action
# try P with different probabilities when transmitting
P = np.array([[p00, p01],
              [p00 ** (L + 1), (1 - p00 ** (L + 1))],
              [p10, p11],
              [p10 * p00 ** L, (1 - p10 * p00 ** L)]])
# so it is the same for both options
R = np.array(
    [rnt * p00 + rp * p01, rt * p00 ** (L + 1) + rc * (1 - p00 ** (L + 1)), rp *p11 + rnt * p10,
     rc * (1 - p10 * p00 ** L) + rt * p10 * p00 ** L])
instant_reward = np.array([[rnt, rp],  # r(0,0,0) r(0,0,1)      # this is the reward of r(s,a,s')
                           [rt, rc],  # r(0,1,0) r(0,1,1)
                           [rnt, rp],  # r(1,0,0) r(1,0,1)
                           [rt, rc]])  # r(1,1,0) r(1,1,1)

# transition matrix and reward
N_states_ext = length_state * N_states
# each state is represented as s0[i], where i is the time in the same state, an '0' for idle and '1' for occupied
P_ext = np.zeros([N_states_ext * N_actions, N_states_ext])
R_ext = np.zeros(N_states_ext * N_actions)
num_rows_ext, num_cols_ext = P_ext.shape
option_P_A = True
if option_P_A:
    # transistion probability with expected decay
    for row in range(num_rows_ext):
        if row < length_state * N_actions:  # different treatment to set of '0' states and set of '1' states
            if row % 2 == 0: # these are the rows corresponding to action 'not transmit'
                aux_col = row / 2 + 1
                if row / 2 + 1 >= length_state: # we cant surpass the max duration in a state, all probabilities that
                    aux_col = row / 2           # exceeds are truncated to the max duration

                aux_col = int(aux_col)  # force to int
                P_ext[row, aux_col] = p00**(row/2) * p00    # situation when it remains in the same state
                P_ext[row, length_state] = 1 - P_ext[row, aux_col]  # situation when the state changes

                R_ext[row] = rnt * P_ext[row, aux_col] + rp * P_ext[row, length_state]

            else:   # rows corresponding to action 'transmit'
                aux_col = row//2 + L + 1
                if aux_col >= length_state:     # deal when surpassing the max duration of a state
                    aux_col = length_state - 1

                aux_col = int(aux_col)  # force to int
                P_ext[row, aux_col] = p00**(row//2) * p00**(L+1)    # situation of a correct transmission
                P_ext[row, length_state] = 1 - P_ext[row, aux_col]  # situation of a collision
                R_ext[row] = rt * P_ext[row, aux_col] + rc * P_ext[row, length_state]

        else:   # treat the set of '1' states

            if row % 2 == 0:  # these are the rows corresponding to action 'not transmit'
                aux_col = row / 2 + 1
                if aux_col >= num_cols_ext:
                    aux_col = row / 2

                aux_col = int(aux_col)  # force to int
                P_ext[row, aux_col] = p11**(row/2 - length_state) * p11 # when it remains in the same state
                P_ext[row, 0] = 1 - P_ext[row, aux_col]
                R_ext[row] = rp * P_ext[row, aux_col] + rnt * P_ext[row, 0]

            else:   # rows corresponding to action 'transmit'


                # we have to discern two situation for collisions:
                # 1.- the next state after taking the action of transmit is 'busy', so the duration of the busy state
                # increases by one
                # 2.- if the first case does not  happen, any return to the busy state in the remaining transition, will
                # make to the state '1'('busy') with 0 duration
                aux_col = row // 2 + 1
                if aux_col >= length_state * N_states:
                    aux_col = row // 2

                # in the case L>=length state, the corresponding column varies to be the maximum of the length state col
                aux_col_succesful_tx = L
                if L >= length_state:
                    aux_col_succesful_tx = length_state - 1
                aux_col = int(aux_col)  # force to int
                P_ext[row, aux_col] = p11**(row//2 - length_state) * p11    # situation 1.- explained above
                P_ext[row, aux_col_succesful_tx] = (1 - P_ext[row, aux_col]) * p00**L #p11 ** (row // 2 - length_state) * p10 * p00 ** L  # successful transmission
                P_ext[row, length_state] += 1 - (P_ext[row, aux_col] + P_ext[row, aux_col_succesful_tx])    # situation 2.- explained above
                R_ext[row] = rt * P_ext[row, aux_col_succesful_tx] + rc * (P_ext[row, aux_col] + P_ext[row, length_state])

# probability transition with heavy decay. option B(is wrong
else:
    for row in range(num_rows_ext):
        if row < length_state * N_actions:  # different treatment to set of '0' states and set of '1' states
            if row % 2 == 0:    # these are the rows corresponding to action 'not transmit'
                aux_col = row / 2 + 1
                if row / 2 + 1 >= length_state: # we cant surpass the max duration in a state, all probabilities that
                    aux_col = row / 2           # exceeds are truncated to the max duration

                aux_col = int(aux_col)  # force to int
                P_ext[row, aux_col] = p00**(row/2) * p00    # situation when it remains in the same state
                P_ext[row, length_state] = 1 - P_ext[row, aux_col]  # situation when the state changes

                R_ext[row] = rnt * P_ext[row, aux_col] + rp * P_ext[row, length_state]

    for row in range(num_rows_ext):
        if row < length_state * N_actions:  # different treatment to set of '0' states and set of '1' states
            if row % 2 != 0:   # rows corresponding to action 'transmit'
                aux_col = row//2 + L + 1
                exceed_slots = 0
                if aux_col >= length_state:     # truncate when surpassing the max duration of a state
                    aux_col = length_state - 1
                    exceed_slots = (row//2 + L + 1) - aux_col  # slots the transmission exceeds the max length state
                    if exceed_slots == 3:
                        exceed_slots = 2    # the last case has to be cut to 2 in order to work
                aux_col = int(aux_col)  # force to int
                #P_ext[row, aux_col] = p00**(row//2) * p00**(L+1)    # situation of a correct transmission
                # correct transmission
                P_ext[row, aux_col] = P_ext[row - 1, aux_col - L + exceed_slots]   # initialize with staying in the same state next slot
                # multiplicate the prob of staying in the same state the next slots
                for i in range(L - exceed_slots):
                    P_ext[row, aux_col] *= P_ext[row + 1 + 2*i, int((row + 1 + 2*i) / 2 + 1)]
                # if it reaches the max length state with remaining slots to transmit, multiply the necessary slots
                # by the probability of staying in the same state in the max length (truncated probability)
                P_ext[row, aux_col] *= P_ext[length_state * N_actions - N_actions, length_state - 1] ** exceed_slots
                P_ext[row, length_state] = 1 - P_ext[row, aux_col]  # situation of a collision
                R_ext[row] = rt * P_ext[row, aux_col] + rc * P_ext[row, length_state]

        else:   # treat the set of '1' states

            if row % 2 == 0:  # these are the rows corresponding to action 'not transmit'
                aux_col = row / 2 + 1
                if aux_col >= num_cols_ext:
                    aux_col = row / 2

                aux_col = int(aux_col)  # force to int
                P_ext[row, aux_col] = p11**(row/2 - length_state) * p11 # when it remains in the same state
                P_ext[row, 0] = 1 - P_ext[row, aux_col]
                R_ext[row] = rp * P_ext[row, aux_col] + rnt * P_ext[row, 0]

            else:   # rows corresponding to action 'transmit'

                # we have to discern two situation for collisions:
                # 1.- the next state after taking the action of transmit is 'busy', so the duration of the busy state
                # increases by one
                # 2.- if the first case does not  happen, any return to the busy state in the remaining transition, will
                # make to the state '1'('busy') with 0 duration
                aux_col = row // 2 + 1
                if aux_col >= length_state * N_states:
                    aux_col = row // 2

                aux_col = int(aux_col)  # force to int
                P_ext[row, aux_col] = p11**(row//2 - length_state) * p11    # situation 1.- explained above
                P_ext[row, L] = (1 - P_ext[row, aux_col]) #* p00**L # successful transmission
                for i in range(L):
                    P_ext[row, L] *= P_ext[i*2, i+1]
                P_ext[row, length_state] = 1 - (P_ext[row, aux_col] + P_ext[row, L])    # situation 2.- explained above
                R_ext[row] = rt * P_ext[row, L] + rc * (P_ext[row, aux_col] + P_ext[row, length_state])

print('\n')






