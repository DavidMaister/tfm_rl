import numpy as np
import random
import config
import datetime
import matplotlib.pyplot as plt  # using backend Qt5Agg in matplotlibrc conf file
from matplotlib.cm import get_cmap
from envs.cognitive_user import CognitiveUser
from envs.cognitive_user_extended import CognitiveUserExtended
from Algorithms.algorithms import dqn, evaluate_policies, value_iteration, plot_stats, q_learning_ext, dqn_evaluate
from Algorithms.DQNAgent import DoubleDQNAgent
from Algorithms.DRQN import DRQNAgent
from Algorithms.DRQN import drqn, drqn_evaluate
from joblib import Parallel, delayed
import os
import itertools
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
from tensorflow import keras


def q_learning(env, Q, epsilon, alpha=config.alpha_ql, gamma=config.gamma, min_epsilon=config.epsilon_min_ql,
               epsilon_decay=config.epsilon_decay_ql):
    done = False
    state = env.reset()

    while not done:
        exploit_explore_tradeoff = random.uniform(0, 1)

        # Choose an action following exploitation. Select the best action
        if exploit_explore_tradeoff > epsilon:
            action = np.argmax(Q[state, :])

        # instead, following the exploration. Choosing a random action
        else:
            action = env.action_space.sample()

        # Perform the action, observe the environment and get the reward
        new_state, reward, done, info = env.step(action)

        ### Update the Q-table
        Q[state, action] = Q[state, action] + alpha * (reward + gamma * np.max(Q[new_state, :]) - Q[state, action])

        # update the state for the next iteration
        state = new_state

        # update epsilon
        if epsilon > min_epsilon:
            epsilon *= epsilon_decay

    return Q, epsilon


def process(env, algorithm, weights_path, seed):
    agent, score_dqn, stats = dqn(env, algorithm)
    score_eval, _ = dqn_evaluate(env, agent, algorithm, rep=config.dqn_evaluate_reps)
    agent.model.save(weights_path + algorithm + '/model_weights_' + str(seed))

    return score_dqn, stats, score_eval


def process_drqn(env, algorithm, weights_path, seed):
    agent, score_dqn, stats = drqn(env, algorithm)
    score_eval, _ = drqn_evaluate(env, agent, algorithm, rep=config.dqn_evaluate_reps)
    agent.model.save(weights_path + 'drqn_' + algorithm + '/model_weights_' + str(seed))

    return score_dqn, stats, score_eval


if __name__ == '__main__':
    # create directories for weights and results
    date_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    print(date_time)
    results_path = 'results/' + date_time + '/images/'
    if not os.path.exists(results_path):
        os.makedirs(results_path)  # create directory if not exists

    weights_path = 'results/' + date_time + '/weights/'
    if not os.path.exists(weights_path):
        os.makedirs(weights_path)  # create directory if not exists
    plt.style.use('ggplot')
    plt.rcParams["figure.figsize"] = (15, 7)
    plt.rcParams['figure.dpi'] = 280
    plt.rcParams['figure.constrained_layout.use'] = True
    # plt.rcParams['savefig.bbox'] = 'tight'
    plt.ioff()

    # optimal policy
    # deterministic policies
    pi1 = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
    pi2 = np.array([[0, 1, 0, 0], [0, 0, 1, 0]])
    pi3 = np.array([[1, 0, 0, 0], [0, 0, 0, 1]])
    pi4 = np.array([[0, 1, 0, 0], [0, 0, 0, 1]])
    pi_list = [pi1, pi2, pi3, pi4]

    # v = inv(I - gamma* pi* P) * pi * R
    v_analytical = [np.linalg.inv(np.identity(2) - config.gamma*pi1 @ config.P) @  pi1 @ config.R,
                    np.linalg.inv(np.identity(2) - config.gamma*pi2 @ config.P) @  pi2 @ config.R,
                    np.linalg.inv(np.identity(2) - config.gamma*pi3 @ config.P) @  pi3 @ config.R,
                    np.linalg.inv(np.identity(2) - config.gamma*pi4 @ config.P) @  pi4 @ config.R]
    # q = inv(I - gamma * P * pi) * R
    q_analytical = [np.linalg.inv(np.identity(4) - config.gamma * config.P @ pi1) @ config.R,
                    np.linalg.inv(np.identity(4) - config.gamma * config.P @ pi2) @ config.R,
                    np.linalg.inv(np.identity(4) - config.gamma * config.P @ pi3) @ config.R,
                    np.linalg.inv(np.identity(4) - config.gamma * config.P @ pi4) @ config.R]
    # the optimal policy is the one deterministic policy with the maximum value in any
    pi_opt_number = np.argmax(np.max(v_analytical, axis=1))
    pi_opt = pi_list[pi_opt_number]
    v_opt = v_analytical[pi_opt_number]
    q_opt = q_analytical[pi_opt_number]
    # check that the value is correct
    if all(np.isclose(v_opt, pi_opt @ q_opt)) and all(np.isclose(q_opt, config.R + config.gamma * config.P @ v_opt)):
        print('Equations for optimal policy verified')
    print('v optimal: ', v_opt)
    print('q optimal: ', q_opt)
    # reshape q_opt
    q_opt = np.reshape(q_opt, (2, 2))


    # Value iteration
    v_vi, q_vi = value_iteration(config.gamma, config.N_states, config.N_actions, config.steps_vi, config.P, config.R)
    print("Value iteration.\nv:", v_vi[:, config.steps_vi - 1])
    print("q:", q_vi[:, config.steps_vi - 1])

    fig1, ax1 = plt.subplots(1, 2)
    ax1[0].plot(v_vi[0, :], label='v_vi[0]', color='r')
    ax1[0].plot(v_vi[1, :], label='v_vi[1]', color='b')
    ax1[0].axhline(v_opt[0], linewidth=0.5, label='v_opt[0]', color='r')
    ax1[0].axhline(v_opt[1], linewidth=0.5, label='v_opt[1]', color='b')
    ax1[0].legend()
    ax1[0].set_xlabel('step')
    ax1[0].set_title('V value iteration')

    ax1[1].plot(q_vi[0, :], label='q_vi[0,0]', color='r')
    ax1[1].plot(q_vi[1, :], label='q_vi[0,1]', color='b')
    ax1[1].plot(q_vi[2, :], label='q_vi[1,0]', color='g')
    ax1[1].plot(q_vi[3, :], label='q_vi[1,1]', color='y')
    ax1[1].axhline(q_opt[0][0], linewidth=0.5, label='q_opt[0,0]', color='r')
    ax1[1].axhline(q_opt[0][1], linewidth=0.5, label='q_opt[0,1]', color='b')
    ax1[1].axhline(q_opt[1][0], linewidth=0.5, label='q_opt[1,0]', color='g')
    ax1[1].axhline(q_opt[1][1], linewidth=0.5, label='q_opt[1,1]', color='y')

    ax1[1].set_xlabel('step')
    ax1[1].set_title('Q value iteration')
    ax1[1].legend()
    if not os.path.exists('results'):
        os.makedirs('results')  # create directory if not exists
    plt.savefig(results_path +'value_iteration.png')

    # instance the gym environemnet
    environment = CognitiveUser()
    environment.reset()
    # reshape the q value to a matrix form
    q_vi = np.reshape(q_vi[:, config.steps_vi - 1], (2, 2))

    # Q-Learning  #

    # Initializing the Q-table
    Q = np.zeros((environment.observation_space.n, environment.action_space.n, config.epochs_ql))
    Q_transmissions = np.zeros(config.epochs_ql - 1)
    Q_collisions = np.zeros(config.epochs_ql - 1)
    Q_totals = np.zeros(config.epochs_ql - 1)
    epsilon_inst = config.epsilon_ql  # instantaneous value of epsilon, it decays every iteration
    for i in range(config.epochs_ql - 1):
        environment.reset()
        Q[:, :, i + 1], epsilon_inst = q_learning(environment, Q[:, :, i], epsilon_inst, config.alpha_ql, config.gamma)
        Q_totals[i], Q_transmissions[i], Q_collisions[i] = environment.get_stats()

    print('Q-Learning obtained Q table:\n', Q[:, :, config.epochs_ql - 1])

    fig_ql, axql = plt.subplots(1, 2)
    plot_stats(axql[1], Q_totals, Q_transmissions, Q_collisions)
    axql[1].set_title('Q learning stats')
    axql[1].set_xlabel('step')

    # plot q-learning
    axql[0].plot(Q[0, 0, :], label='q(0,0)', color='r')
    axql[0].plot(Q[0, 1, :], label='q(0,1)', color='b')
    axql[0].plot(Q[1, 0, :], label='q(1,0)', color='g')
    axql[0].plot(Q[1, 1, :], label='q(1,1)', color='y')
    axql[0].axhline(q_opt[0][0], linewidth=0.5, label='q_opt[0,0]', color='r')  # horizontal lines with the
    axql[0].axhline(q_opt[0][1], linewidth=0.5, label='q_opt[1,0]', color='b')  # corresponding optimal value.
    axql[0].axhline(q_opt[1][0], linewidth=0.5, label='q_opt[0,1]', color='g')
    axql[0].axhline(q_opt[1][1], linewidth=0.5, label='q_opt[1,1]', color='y')
    axql[0].legend()
    axql[0].set_title('Q-learning values evolution')
    axql[0].set_xlabel('step')
    plt.savefig(results_path +'ql.png')

    algorithm = 'original'
    # DQN #
    if config.flag_train_dqn:
        # for dqn, we select the best agent among different seeds
        n_jobs = config.n_jobs  # number of simultaneous threads !!always less than processor has!!!
        n_seeds = config.n_seeds  # number of seeds

        if not os.path.exists(results_path + 'dqn_' + algorithm):
            os.makedirs(results_path + 'dqn_' + algorithm)  # create directory if not exists

        # figures as seeds, each element in the list contains first the figure and second the axis
        process_return = Parallel(n_jobs=n_jobs, verbose=10) \
            (delayed(process)(environment, algorithm, weights_path, seed=seed) for seed in range(n_seeds))
        # the return consists in a list for every seed with the structure:
        # 0 - score of the training process
        # 1 - the stats of the training process
        # 2 - the mean score of the evaluation process
        scores_eval = []
        # plot the results of the training
        for i in range(n_seeds):
            aux_fig, aux_ax = plt.subplots(2, 1, num='dqn_original_' + str(i))
            aux_fig.suptitle('DQN ' + algorithm + ' ' + str(i))
            aux_ax[0].plot(process_return[i][0], label='Score of episode ' + str(i))
            aux_ax[0].set_xlabel('episode')
            aux_ax[0].set_title('Score')
            aux_ax[0].legend()

            plot_stats(aux_ax[1], process_return[i][1][0], process_return[i][1][1], process_return[i][1][2], str(i))
            aux_ax[1].set_xlabel('episode')
            aux_ax[1].set_title('Stats')

            # save the figure
            aux_fig.savefig(results_path + 'dqn_' + algorithm + '/seed_' + str(i) + '.png')
            scores_eval.append(process_return[i][2])  # store all the scores in the same list to select the best

        # get the agent that has better value
        max_score_eval = np.argmax(scores_eval)
        print('Best model is: ' + str(max_score_eval))
        # load the model with higher score
        model = keras.models.load_model(weights_path + algorithm + '/model_weights_' + str(max_score_eval))

    else:   # load pretrained model
        model = keras.models.load_model('model_weights_original')
        print('Model loaded')
    # initialize a DQNAgent class with the trained model
    dqn_agent = DoubleDQNAgent(environment.observation_space.n, environment.action_space.n, model=model)
    dqn_agent.epsilon = 0
    # evaluate
    evaluate_policies(config.states, q_vi, Q[:, :, config.epochs_ql - 1], dqn_agent, algorithm, results_path)

    #                                                   #
    #                                                   #
    #                                                   #
    # ---------- Extended state space ------------------#
    #                                                   #
    #                                                   #
    #                                                   #
    #                                                   #
    print('---------- Extended space -----------')

    # Optimal policy calculated analytically
    zero = np.array([0, 1])
    one = np.array([1, 0])

    policies = []  # list of all policies
    combinations = [list(i) for i in itertools.product([0, 1], repeat=config.N_states_ext)]
    for combination in combinations:  # every different bit combination -> result in a different policy
        pi = np.zeros([config.N_states_ext, config.N_states_ext * config.N_actions])
        for i in range(pi.shape[0]):  # construct each policy
            pi[i, i * 2:i * 2 + config.N_actions] = zero if combination[i] == 0 else one
        policies.append(pi)
    v_analytical_ext = []
    q_analytical_ext = []
    for pi in policies:
        v = np.linalg.inv(np.identity(config.N_states_ext) - config.gamma * pi @ config.P_ext) @ pi @ config.R_ext
        q = np.linalg.inv(np.identity(config.N_states_ext * config.N_actions) - config.gamma * config.P_ext @ pi) @ config.R_ext
        v_analytical_ext.append(v)
        q_analytical_ext.append(q)
    sums_v = []
    for v in v_analytical_ext:
        sums_v.append(sum(v))
    v_opt_ext = v_analytical_ext[np.argmax(sums_v)]
    q_opt_ext = q_analytical_ext[np.argmax(sums_v)]
    pi_opt_ext = policies[np.argmax(sums_v)]
    if all(np.isclose(v_opt_ext, pi_opt_ext @ q_opt_ext)) and all(
            np.isclose(q_opt_ext, config.R_ext + config.gamma * config.P_ext @ v_opt_ext)):
        print('Equations for optimal policy verified')


    env_ext = CognitiveUserExtended()

    # value iteration
    v_vi_ext, q_vi_ext = value_iteration(config.gamma, config.N_states_ext, config.N_actions, config.steps_vi,
                                         config.P_ext, config.R_ext)

    fig_vi_ext, ax_vi_ext = plt.subplots(1, 2)
    plt.suptitle('Value Iteration Extended')
    # name = "tab10"  # a way to use a colour palette
    # colors = get_cmap(name).colors
    # ax_vi_ext[0].set_prop_cycle(color=colors)
    color = iter(plt.cm.rainbow(np.linspace(0, 1, config.N_states_ext)))
    for i in range(config.N_states_ext):
        c = next(color)
        ax_vi_ext[0].plot(v_vi_ext[i, :], color=c, label='v_vi[' + str(i) + ']')
        ax_vi_ext[0].axhline(v_opt_ext[i], linewidth=0.5, color=c)
    ax_vi_ext[0].legend()
    ax_vi_ext[0].set_title('V value')
    ax_vi_ext[0].set_xlabel('step')

    # a way to make the necessary colours we need
    color = iter(plt.cm.rainbow(np.linspace(0, 1, config.N_states_ext * config.N_actions)))
    for i in range(config.N_states_ext * config.N_actions):
        c = next(color)
        ax_vi_ext[1].plot(q_vi_ext[i, :], color=c, label='q_vi[' + str(i) + ']')
        ax_vi_ext[1].axhline(q_opt_ext[i], linewidth=0.5, color=c)
    ax_vi_ext[1].legend()
    ax_vi_ext[1].set_title('Q value')
    ax_vi_ext[1].set_xlabel('step')

    plt.savefig(results_path + 'vi_ext.png')

    # reshape the q_vi_ext to a shape of 3D matrix
    q_vi_ext = q_vi_ext[:, -1].reshape(env_ext.observation_space.high[0] + 1, env_ext.observation_space.high[1] + 1,
                                       env_ext.action_space.n)
    q_opt_ext = q_opt_ext.reshape(env_ext.observation_space.high[0] + 1, env_ext.observation_space.high[1] + 1,
                                       env_ext.action_space.n)

    # Q learning extended
    # Initializing the Q-table
    Q_ext = np.zeros((env_ext.observation_space.high[0] + 1, env_ext.observation_space.high[1] + 1,
                      env_ext.action_space.n, config.epochs_ql_ext))

    # initialize values
    Q_transmissions_ext = np.zeros(config.epochs_ql_ext - 1)
    Q_collisions_ext = np.zeros(config.epochs_ql_ext - 1)
    Q_totals_ext = np.zeros(config.epochs_ql_ext - 1)
    epsilon_inst = config.epsilon_ql  # instantaneous value of epsilon, it decays every iteration

    for i in range(config.epochs_ql_ext - 1):
        env_ext.reset()
        Q_ext[:, :, :, i + 1], epsilon_inst = q_learning_ext(env_ext, Q_ext[:, :, :, i], epsilon_inst, config.alpha_ql,
                                                             config.gamma)  # Q-learning formula
        Q_totals_ext[i], Q_transmissions_ext[i], Q_collisions_ext[i] = env_ext.get_stats()

    # plot the results
    fig_ql_ext, ax_ql_ext = plt.subplots(1, 2)
    plt.suptitle('Q learning extended')
    # ax_ql_ext[0].plot(Q_ext[0, 0, 0, :], label='Q[(0,0),0]')
    # ax_ql_ext[0].plot(Q_ext[0, 0, 1, :], label='Q[(0,0),1]')
    # ax_ql_ext[0].plot(Q_ext[1, 0, 0, :], label='Q[(1,0),0]')
    # ax_ql_ext[0].plot(Q_ext[1, 0, 1, :], label='Q[(0,0),1]')
    # a way to make the necessary colours we need
    color = iter(plt.cm.rainbow(np.linspace(0, 1, config.N_states_ext * config.N_actions)))
    for i in range(config.N_states):
        for j in range(config.length_state):
            for t in range(config.N_actions):
                c = next(color)
                ax_ql_ext[0].plot(Q_ext[i, j, t, :], color=c, label='ql_q[' + str(i) + ',' + str(j) + ','+str(t) + ']')
                ax_ql_ext[0].axhline(q_opt_ext[i, j, t], linewidth=0.5,color=c)

    ax_ql_ext[0].set_title('Q values evolution')
    ax_ql_ext[0].set_xlabel('step')
    ax_ql_ext[0].legend(fontsize="x-small", loc= 'upper left')

    plot_stats(ax_ql_ext[1], Q_totals_ext, Q_transmissions_ext, Q_collisions_ext)
    ax_ql_ext[1].set_title('Q stats evolution')
    ax_ql_ext[1].set_xlabel('step')
    plt.savefig(results_path + 'ql_ext.png')

    # DQN
    algorithm = 'expanded'
    # DQN #
    if config.flag_train_dqn:
        # for dqn, we select the best agent among different seeds
        n_jobs = config.n_jobs  # number of simultaneous threads !!always less than processor has!!!
        n_seeds = config.n_seeds  # number of seeds
        algorithm = 'expanded'
        if not os.path.exists(results_path + 'dqn_' + algorithm):
            os.makedirs(results_path + 'dqn_' + algorithm)  # create directory if not exists

        # figures as seeds, each element in the list contains first the figure and second the axis
        process_return_ext = Parallel(n_jobs=n_jobs, verbose=10) \
            (delayed(process)(env_ext, algorithm, weights_path, seed=seed) for seed in range(n_seeds))
        # the return consists in a list for every seed with the structure:
        # 0 - score of the training process
        # 1 - the stats of the training process
        # 2 - the mean score of the evaluation process
        scores_eval_ext = []
        # plot the results of the training
        for i in range(n_seeds):
            aux_fig, aux_ax = plt.subplots(2, 1, num='dqn_extended_' + str(i))
            aux_fig.suptitle('DQN ' + algorithm + ' ' + str(i))
            aux_ax[0].plot(process_return_ext[i][0], label='Score of episode ' + str(i))
            aux_ax[0].set_xlabel('episode')
            aux_ax[0].set_title('Score')
            aux_ax[0].legend()

            plot_stats(aux_ax[1], process_return_ext[i][1][0], process_return_ext[i][1][1],
                       process_return_ext[i][1][2], str(i))
            aux_ax[1].set_xlabel('episode')
            aux_ax[1].set_title('Stats')

            # save the figure
            aux_fig.savefig(results_path + 'dqn_' + algorithm + '/seed_' + str(i) + '.png')
            scores_eval_ext.append(process_return_ext[i][2])  # store all the scores in the same list to select the best

        # get the agent that has better value
        max_score_eval_ext = np.argmax(scores_eval_ext)
        print('Best model is: ' + str(max_score_eval_ext))
        # load the model with higher score
        model_ext = keras.models.load_model(weights_path + algorithm + '/model_weights_' + str(max_score_eval_ext))

    else:   # load a pretrained model
        model_ext = keras.models.load_model('model_weights_expanded')
        print('Model loaded')
    # initialize a DQNAgent class with the trained model
    dqn_agent_ext = DoubleDQNAgent(env_ext.observation_space.shape[0], env_ext.action_space.n, model=model_ext)
    dqn_agent_ext.epsilon = 0

    # evaluate
    evaluate_policies(config.states, q_vi_ext, Q_ext[:, :, :, - 1], dqn_agent_ext, algorithm, results_path)


    # ------- #
    #  DRQN   #
    #         #
    #drqn(environment, 'original')
    if config.flag_train_dqn:
        # for dqn, we select the best agent among different seeds
        n_jobs = config.n_jobs    # number of simultaneous threads !!always less than processor has!!!
        n_seeds = config.n_seeds  # number of seeds
        algorithm = 'original'
        if not os.path.exists(results_path + 'drqn_' + algorithm):
            os.makedirs(results_path + 'drqn_' + algorithm)  # create directory if not exists

        # figures as seeds, each element in the list contains first the figure and second the axis
        process_return_drqn = Parallel(n_jobs=n_jobs, verbose=10) \
            (delayed(process_drqn)(environment, algorithm, weights_path, seed=seed) for seed in range(n_seeds))
        # the return consists in a list for every seed with the structure:
        # 0 - score of the training process
        # 1 - the stats of the training process
        # 2 - the mean score of the evaluation process
        scores_eval_drqn = []
        # plot the results of the training
        for i in range(n_seeds):
            aux_fig, aux_ax = plt.subplots(2, 1, num='drqn_' + str(i))
            aux_fig.suptitle('DRQN ' + algorithm + ' ' + str(i))
            aux_ax[0].plot(process_return_drqn[i][0], label='Score of episode ' + str(i))
            aux_ax[0].set_xlabel('episode')
            aux_ax[0].set_title('Score')
            aux_ax[0].legend()

            plot_stats(aux_ax[1], process_return_drqn[i][1][0], process_return_drqn[i][1][1],
                       process_return_drqn[i][1][2], str(i))
            aux_ax[1].set_xlabel('episode')
            aux_ax[1].set_title('Stats')

            # save the figure
            aux_fig.savefig(results_path + 'drqn_' + algorithm + '/seed_' + str(i) + '.png')
            scores_eval_drqn.append(process_return_drqn[i][2])  # store all the scores in the same list to select the best
        # get the agent that has better value
        max_score_eval_drqn = np.argmax(scores_eval_drqn)
        print('Best model is: ' + str(max_score_eval_drqn))
        # load the model with higher score
        model_drqn = keras.models.load_model(
            weights_path + 'drqn_' + algorithm + '/model_weights_' + str(max_score_eval_drqn))

    else:
        model_drqn = keras.models.load_model('model_weights_drqn')

    agent_drqn = DRQNAgent(environment.observation_space.n, environment.action_space.n, config.trace_length,
                           model=model_drqn)

    # Evaluate all deep networks together

    algorithm = 'original'
    _, drqn_stats = drqn_evaluate(environment, agent_drqn, algorithm,
                                       trace_length=config.trace_length, rep=config.drqn_evaluate_reps)
    _, dqn_stats = dqn_evaluate(environment, dqn_agent, algorithm, rep=config.dqn_evaluate_reps)
    _, dqn_ext_stats = dqn_evaluate(env_ext, dqn_agent_ext, 'expanded', rep=config.dqn_evaluate_reps)

    fig_nn_comp, ax_nn_comp = plt.subplots(1, 2, num='nn_comp')
    fig_nn_comp.suptitle('Deep networks comparison')
    ax_nn_comp[0].hist(drqn_stats[0], label='drqn', bins='auto')
    ax_nn_comp[0].hist(dqn_stats[0], label='dqn', bins='auto')
    ax_nn_comp[0].hist(dqn_ext_stats[0], label='dqn_ext', bins='auto')
    ax_nn_comp[0].set_xlabel('episode')
    ax_nn_comp[0].set_ylabel('score')
    ax_nn_comp[0].legend()
    #ax_nn_comp[0].set_title('Score')

    plot_stats(ax_nn_comp[1], drqn_stats[1], drqn_stats[2], drqn_stats[3], label='drqn', plot_style='hist')
    plot_stats(ax_nn_comp[1], dqn_stats[1], dqn_stats[2], dqn_stats[3], label='dqn', plot_style='hist')
    plot_stats(ax_nn_comp[1], dqn_ext_stats[1], dqn_ext_stats[2], dqn_ext_stats[3], label='dqn_ext', plot_style='hist')
    ax_nn_comp[1].set_title('Stats')

    fig_nn_comp.savefig(results_path + 'nn_comparison.png')
    print('\n')
    # plt.show()
