import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter1d
import time
import os
import copy
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
from keras.utils import to_categorical

from Algorithms.DQNAgent import DoubleDQNAgent
from envs.cognitive_user import CognitiveUser
import random


def plot_stats(ax, totals, transmissions, collisions, label='', plot_style='plot', alpha=0.65,
               hist_bins=np.arange(0, 1, 0.025)):
    aux = transmissions + collisions
    if plot_style == 'plot':
        ax.plot(aux / totals, label='T+C/Total '+label, alpha=0.6)
        ax.plot(uniform_filter1d(aux/totals, size=40, mode='nearest'), label='T+C/Total MA '+label)
        aux[aux == 0] = 1  # avoid division by 0 by converting the denominator to 1 so the division will be '0'.
        ax.plot(transmissions / aux, label='T/T+C ' + label, alpha=0.6)
        ax.plot(uniform_filter1d(transmissions / aux, size=40, mode='nearest'), label='T/T+C MA ' + label)
        ax.set_ylim([0, 1])

    elif plot_style == 'hist':  # if
        ax.hist(aux / totals, label='T+C/Total ' + label, alpha=alpha, bins=hist_bins)
        aux[aux == 0] = 1  # avoid division by 0 by converting the denominator to 1 so the division will be '0'.
        ax.hist(transmissions / aux, label='T/T+C ' + label, alpha=alpha, bins=hist_bins)

    ax.legend()
    return ax


def encode_input_nn(algorithm, state, state_size):
    if algorithm == 'original':
        encoded = to_categorical(np.array([0, 1]))
        state = encoded[state]
        state = np.reshape(state, [1, state_size])
    elif algorithm == 'expanded':
        state = np.reshape(state, [1, state_size])
    return state


def value_iteration(gamma, N_states, N_actions, N_steps, P, R):
    # Value iteration
    v = np.zeros((N_states, N_steps))
    q = np.zeros((N_states * N_actions, N_steps))

    for k in range(N_steps-1):
        aux_q = np.zeros(N_states)  # aux variable to calculate the array of maximums needed for the q function
        for kk in range(N_states):
            v[kk, k + 1] = max(R[N_actions * kk: N_actions * (kk + 1)] + gamma *
                               P[N_actions * kk: N_actions * (kk + 1), :] @ v[:, k])
            # the indexes takes the part of the matrix of P and R that involves the kk state
            aux_q[kk] = max(q[N_actions * kk: N_actions * (kk + 1), k])
        q[:, k + 1] = R + gamma * P @ aux_q
    return v, q


def dqn(env, algorithm, EPISODES, train_wait, config):
    # Get size of state and action
    if algorithm == 'original':
        state_size = env.observation_space.n
    elif algorithm == 'expanded':
        state_size = env.observation_space.shape[0]
    action_size = env.action_space.n



    # Create the agent
    agent = DoubleDQNAgent(state_size, action_size, config)
    scores, episodes = [], []  # To store values for plotting
    totals = np.array([])   # to store stats of the process
    transmissions = np.array([])
    collisions = np.array([])
    break_flag = False  # To stop training when the agent has successfully learned
    for e in range(EPISODES):
        if break_flag:
            break
        done = False
        score = 0
        counter = 0
        tic = time.perf_counter()   # time the episode
        # input dimension for the network is shape of the state size
        state = copy.deepcopy(env.reset())  # Set the initial state
        state = encode_input_nn(algorithm, state, state_size)

        while not done:  # Iterate while the game has not finished
            # Get action for the current state and go one step in environment
            action = agent.get_action(state)  # Using epsilon-greedy policy
            next_state, reward, done, info = env.step(action)
            next_state = encode_input_nn(algorithm, next_state, state_size)
            # If an action makes the episode end before time (i.e, before 499 time steps), then give a penalty of -100
            #reward = reward if not done or score == 499 else -100

            # Save the sample <s, a, r, s'> to the replay memory
            agent.append_sample(state, action, reward, next_state, done)
            # Train
            counter += 1
            if counter % train_wait == 0:
                #tic = time.perf_counter()
                agent.train_model()     # only is trained after a wait of train_wait steps
                #print('pred values', agent.model.predict(encode_input_nn(algorithm, 0, state_size)), agent.model.predict(encode_input_nn(algorithm, 1, state_size)))
                #toc = time.perf_counter()
                #print(f"Train duration:  {toc - tic:0.4f} seconds")
            score += reward * agent.discount_factor ** env.counter
            state = next_state.copy()
            if done:
                # Update target model after each episode
                agent.update_target_model()
                # Store values for plotting
                #score = score if score == 500 else score + 100
                scores.append(score)
                episodes.append(e)
                e_totals, e_transmissions, e_collisions = env.get_stats()
                totals = np.append(totals, e_totals)
                transmissions = np.append(transmissions, e_transmissions)
                collisions = np.append(collisions, e_collisions)
                toc = time.perf_counter()
                # Output the results of the episode
                state_0 = 0 if algorithm == 'original' else np.array([0, 0])
                state_1 = 1 if algorithm == 'original' else np.array([1, 0])
                print("episode:", e, "  score:", score, "  memory length:",
                      len(agent.memory), "  epsilon:", agent.epsilon, "  time spent:", toc-tic,
                      'pred values', agent.model.predict(encode_input_nn(algorithm, state_0, state_size)),
                      agent.model.predict(encode_input_nn(algorithm, state_1, state_size)))

                # Stop if the network converges and does not improve the score in the last 10 episodes
                if len(scores) > 10:
                    if np.var(scores[-min(10, len(scores)):]) < 1:
                        break_flag = True
    # Output whether the agent learnt before time or not
    # if break_flag:
    #     print("Training finished successfully")
    # else:
    #     print("Training finished unsuccessfully")

    stats = [totals, transmissions, collisions]
    # save the figure outside the function if needed
    return agent, scores, stats


def dqn_evaluate(env, agent, algorithm, rep):
    # Get size of state and action
    if algorithm == 'original':
        state_size = env.observation_space.n
    elif algorithm == 'expanded':
        state_size = env.observation_space.shape[0]

    scores = []
    totals = np.array([])  # to store stats of the process
    transmissions = np.array([])
    collisions = np.array([])
    agent.epsilon = 0
    for e in range(rep):
        done = False
        score = 0
        counter = 0
        # input dimension for the network is shape of the state size
        state = copy.deepcopy(env.reset())  # Set the initial state
        state = encode_input_nn(algorithm, state, state_size)
        while not done:
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            next_state = encode_input_nn(algorithm, next_state, state_size)
            score += reward * agent.discount_factor ** counter
            state = next_state.copy()
            counter += 1
            if done:
                scores.append(score)
                e_totals, e_transmissions, e_collisions = env.get_stats()
                totals = np.append(totals, e_totals)
                transmissions = np.append(transmissions, e_transmissions)
                collisions = np.append(collisions, e_collisions)
    mean_score = np.mean(scores)
    stats = [scores, totals, transmissions, collisions]
    return mean_score, stats


def evaluate(config, q, env, rep):
    scores = []
    totals = []  # to store stats of the process
    transmissions = []
    collisions = []
    for e in range(rep):
        done = False
        score = 0
        counter = 0
        state = copy.deepcopy(env.reset())
        while not done:
            action = np.argmax(q[state, :])
            next_state, instant_reward, done, _ = env.step(action)
            if not config.flag_train_all:
                score += instant_reward
            else:
                score += instant_reward * config.gamma ** counter
            state = next_state.copy()
            counter += 1
            if done:
                scores.append(score)
                e_totals, e_transmissions, e_collisions = env.get_stats()
                totals.append(e_totals)
                transmissions.append(e_transmissions)
                collisions.append(e_collisions)

                # if not config.flag_train_all: # print stats
                #     print("N_collisions: ", e_collisions, "N_transmission: ", e_transmissions, "N_no_transmissions: ",
                #           env.n_rnt, "N_rp: ", env.n_rp, "N_totals: ", e_totals, "score", score)

    transmissions = np.array(transmissions)
    collisions = np.array(collisions)
    totals = np.array(totals)
    aux = transmissions + collisions
    stats1 = aux / totals
    aux[aux == 0] = 1
    stats2 = transmissions / aux
    return scores, stats1, stats2


def evaluate_policies(states, q_vi, ql, dqn_agent, algorithm, results_path, gamma, rep, config):
    env_vi = CognitiveUser(config) #if algorithm == 'original' else CognitiveUserExtended()    # value iteration
    env_ql = CognitiveUser(config) #if algorithm == 'original' else CognitiveUserExtended()    # q learning
    env_dqn = CognitiveUser(config) #if algorithm == 'original' else CognitiveUserExtended()   # deep q network
    scores_vi, scores_ql, scores_dqn = (np.zeros(rep), np.zeros(rep)), (np.zeros(rep), np.zeros(rep)), (np.zeros(rep), np.zeros(rep))
    totals_vi, totals_ql, totals_dqn = (np.zeros(rep), np.zeros(rep)), (np.zeros(rep), np.zeros(rep)), (np.zeros(rep), np.zeros(rep))
    tx_vi, tx_ql, tx_dqn = (np.zeros(rep), np.zeros(rep)), (np.zeros(rep), np.zeros(rep)), (np.zeros(rep), np.zeros(rep))  # transmissions
    cx_vi, cx_ql, cx_dqn = (np.zeros(rep), np.zeros(rep)), (np.zeros(rep), np.zeros(rep)), (np.zeros(rep), np.zeros(rep))  # collisions
    for i in states:  # iterate over the states
        # scores_vi[i], scores_ql[i], scores_dqn[i] = np.zeros(rep), np.zeros(rep), np.zeros(rep)
        # totals_vi[i], totals_ql[i], totals_dqn[i] = np.zeros(rep), np.zeros(rep), np.zeros(rep) # to save value of each repetition
        # tx_vi[i], tx_ql[i], tx_dqn[i] = np.zeros(rep), np.zeros(rep), np.zeros(rep)
        # cx_vi[i], cx_ql[i], cx_dqn[i] = np.zeros(rep), np.zeros(rep), np.zeros(rep)
        for n in range(rep):
            state = i  # initial state
            done = False
            t = 0   # counter of the while loop
            state_vi = copy.deepcopy(env_vi.reset(i))
            state_ql = copy.deepcopy(env_ql.reset(i))
            state_dqn = copy.deepcopy(env_dqn.reset(i))
            while not done:
                # take the action maximizing q
                action_vi = np.argmax(q_vi[state_vi, :]) if algorithm == 'original' else np.argmax(q_vi[tuple(state_vi)])
                action_ql = np.argmax(ql[state_ql, :]) if algorithm == 'original' else np.argmax(ql[tuple(state_ql)])
                action_dqn = dqn_agent.get_action(encode_input_nn(algorithm, state_dqn, len(states)))

                # get the next state and reward
                next_state_vi, instant_reward_vi, done_vi, _ = env_vi.step(action_vi)
                next_state_ql, instant_reward_ql, done_ql, _ = env_ql.step(action_ql)
                next_state_dqn, instant_reward_dqn, done_dqn, _ = env_dqn.step(action_dqn)

                #   accumulate the reward
                scores_vi[i][n] = scores_vi[i][n] + gamma**t * instant_reward_vi
                scores_ql[i][n] = scores_ql[i][n] + gamma ** t * instant_reward_ql
                scores_dqn[i][n] = scores_dqn[i][n] + gamma ** t * instant_reward_dqn
                state_vi = copy.deepcopy(next_state_vi)
                state_ql = next_state_ql.copy()
                state_dqn = next_state_dqn.copy()
                t += 1
                if done_vi or done_ql or done_dqn:
                    totals_vi[i][n], tx_vi[i][n], cx_vi[i][n] = env_vi.get_stats()
                    totals_ql[i][n], tx_ql[i][n], cx_ql[i][n] = env_ql.get_stats()
                    totals_dqn[i][n], tx_dqn[i][n], cx_dqn[i][n] = env_dqn.get_stats()
                    done = True

    fig_evaluate_policy, ax_eval = plt.subplots(2, 2)
    plt.suptitle('Evaluation of algorithms '+algorithm)
    ax_eval[0][0].hist(scores_vi[0], label='Value Iteration', alpha=0.7, bins='auto')
    ax_eval[0][0].hist(scores_ql[0], label='Q learning', alpha=0.6, bins='auto')
    ax_eval[0][0].hist(scores_dqn[0], label='DQN', alpha=0.6, bins='auto')
    ax_eval[0][0].set_title('Value of state 0')
    ax_eval[0][0].set_xlabel('Value')
    ax_eval[0][0].set_ylabel('frequency')
    ax_eval[0][0].legend()

    ax_eval[0][1].hist(scores_vi[1], label='Value Iteration', alpha=0.7, bins='auto')
    ax_eval[0][1].hist(scores_ql[1], label='Q learning', alpha=0.6, bins='auto')
    ax_eval[0][1].hist(scores_dqn[1], label='DQN', alpha=0.6, bins='auto')
    ax_eval[0][1].set_title('Value of state 1')
    ax_eval[0][1].set_xlabel('Value')
    ax_eval[0][1].set_ylabel('frequency')
    ax_eval[0][1].legend()

    plot_stats(ax_eval[1][0], totals_vi[0], tx_vi[0], cx_vi[0], label='vi', plot_style='hist')
    plot_stats(ax_eval[1][0], totals_ql[0], tx_ql[0], cx_ql[0], label='ql', plot_style='hist')
    plot_stats(ax_eval[1][0], totals_dqn[0], tx_dqn[0], cx_dqn[0], label='dqn', plot_style='hist')
    ax_eval[1][0].set_title('Stats of state 0')
    ax_eval[1][0].set_xlabel('ratio')
    ax_eval[1][0].set_ylabel('frequency')

    plot_stats(ax_eval[1][1], totals_vi[1], tx_vi[1], cx_vi[1], label='vi', plot_style='hist')
    plot_stats(ax_eval[1][1], totals_ql[1], tx_ql[1], cx_ql[1], label='ql', plot_style='hist')
    plot_stats(ax_eval[1][1], totals_dqn[1], tx_dqn[1], cx_dqn[1], label='dqn', plot_style='hist')
    ax_eval[1][1].set_title('Stats of state 1')
    ax_eval[1][1].set_xlabel('ratio')
    ax_eval[1][1].set_ylabel('frequency')
    plt.savefig(results_path + 'evaluation_'+algorithm+'.png')

    print('\n')
    return

