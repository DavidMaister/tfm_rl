import numpy as np
import random
from config import Config
import datetime
import matplotlib.pyplot as plt  # using backend Qt5Agg in matplotlibrc conf file
from matplotlib.cm import get_cmap
from envs.cognitive_user import CognitiveUser
from Algorithms.algorithms import dqn, evaluate_policies, value_iteration, plot_stats, dqn_evaluate, evaluate
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


def q_learning(env, Q, epsilon, alpha, gamma, min_epsilon, epsilon_decay):
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


def process(env, algorithm, weights_path, config, seed):
    agent, score_dqn, stats = dqn(env, algorithm, EPISODES=config.episodes_dqn, train_wait=config.train_wait,
                                  config=config)
    score_eval, _ = dqn_evaluate(env, agent, algorithm, rep=config.dqn_evaluate_reps)
    agent.model.save(weights_path + algorithm + '/model_weights_' + str(seed))

    return score_dqn, stats, score_eval


def process_drqn(env, algorithm, weights_path, config, seed):
    agent, score_dqn, stats = drqn(env, algorithm, trace_length=config.trace_length, EPISODES=config.episodes_drqn,
                                   config = config)
    score_eval, _ = drqn_evaluate(env, agent, algorithm, trace_length=config.trace_length,
                                  rep=config.drqn_evaluate_reps)
    agent.model.save(weights_path + 'drqn_' + algorithm + '/model_weights_' + str(seed))

    return score_dqn, stats, score_eval


if __name__ == '__main__':
    # create directories for weights and results
    date_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    print(date_time)

    plt.style.use('ggplot')
    plt.rcParams["figure.figsize"] = (15, 7)
    plt.rcParams['figure.dpi'] = 280
    plt.rcParams['figure.constrained_layout.use'] = True
    # plt.rcParams['savefig.bbox'] = 'tight'
    plt.ioff()

    # declare the variable to iterate
    changing_variable = 'lambda_on'
    changing_values = range(4, 8, 10)    # range has the format range(start, stop, step)
    #changing_values = np.linspace(0, 5, num=11) # for decimals

    # import config
    config = Config()
    scores_opt = []
    stats_opt = []

    scores_vi = []
    stats_vi = []

    scores_ql = []
    stats_ql = []

    scores_dqn = []
    stats_dqn = []

    scores_drqn = []
    stats_drqn = []



    print_values = []
    for value in changing_values:
        print_values.append(value)
        # import config
        # change the probabilities and rewards according to the new value
        setattr(config, changing_variable, value)
        config.update()

        environment = CognitiveUser(config)
        print(changing_variable, ' = ', value)
        print(environment.P)

        # create directories
        results_path = 'results/' + date_time + '/' + changing_variable + '_' + str(value) + '/images/'
        if not os.path.exists(results_path):
            os.makedirs(results_path)  # create directory if not exists

        weights_path = 'results/' + date_time + '/' + changing_variable + '_' + str(value) + '/weights/'
        if not os.path.exists(weights_path):
            os.makedirs(weights_path)  # create directory if not exists

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
        print("Optimal policy number: ", pi_opt_number)
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

        # only calculates the optimal policy if flag_train_all is set to False.
        if config.flag_train_all:
            # Value iteration ######

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
            ax1[0].set_ylabel('value')
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
            ax1[1].set_ylabel('value')
            ax1[1].set_title('Q value iteration')
            ax1[1].legend()
            if not os.path.exists('results'):
                os.makedirs('results')  # create directory if not exists
            plt.savefig(results_path +'value_iteration.png')

            # instance the gym environemnet

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
                Q[:, :, i + 1], epsilon_inst = q_learning(environment, Q[:, :, i], epsilon_inst, alpha=config.alpha_ql,
                                                          gamma=config.gamma, min_epsilon=config.epsilon_min_ql,
                                                          epsilon_decay=config.epsilon_decay_ql)
                Q_totals[i], Q_transmissions[i], Q_collisions[i] = environment.get_stats()

            print('Q-Learning obtained Q table:\n', Q[:, :, config.epochs_ql - 1])
            q_ql = Q[:, :, config.epochs_ql - 1]

            fig_ql, axql = plt.subplots(1, 2)
            plot_stats(axql[1], Q_totals, Q_transmissions, Q_collisions)
            axql[1].set_title('Q learning stats')
            axql[1].set_xlabel('step')
            axql[1].set_ylabel('ratio')

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
            axql[0].set_ylabel('value')
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
                    (delayed(process)(environment, algorithm, weights_path, config, seed=seed) for seed in range(n_seeds))
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
                    aux_ax[0].set_ylabel('score')
                    aux_ax[0].set_title('Score')
                    aux_ax[0].legend()

                    plot_stats(aux_ax[1], process_return[i][1][0], process_return[i][1][1], process_return[i][1][2], str(i))
                    aux_ax[1].set_xlabel('episode')
                    aux_ax[1].set_ylabel('ratio')
                    aux_ax[1].set_title('Stats')

                    # save the figure
                    aux_fig.savefig(results_path + 'dqn_' + algorithm + '/seed_' + str(i) + '.png')
                    plt.clf()
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
            dqn_agent = DoubleDQNAgent(environment.observation_space.n, environment.action_space.n, config, model=model)
            dqn_agent.epsilon = 0
            # evaluate
            print('Evaluating policies')
            evaluate_policies(config.states, q_vi, Q[:, :, config.epochs_ql - 1], dqn_agent, algorithm, results_path,
                              gamma=config.gamma, rep=config.rep_eval_pol, config=config)

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
                    (delayed(process_drqn)(environment, algorithm, weights_path, config, seed=seed) for seed in range(n_seeds))
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
                    aux_ax[0].set_ylabel('score')
                    aux_ax[0].set_title('Score')
                    aux_ax[0].legend()

                    plot_stats(aux_ax[1], process_return_drqn[i][1][0], process_return_drqn[i][1][1],
                               process_return_drqn[i][1][2], str(i))
                    aux_ax[1].set_xlabel('episode')
                    aux_ax[1].set_ylabel('ratio')
                    aux_ax[1].set_title('Stats')

                    # save the figure
                    aux_fig.savefig(results_path + 'drqn_' + algorithm + '/seed_' + str(i) + '.png')
                    scores_eval_drqn.append(process_return_drqn[i][2])  # store all the scores in the same list to select the best
                    plt.clf()
                # get the agent that has better value
                max_score_eval_drqn = np.argmax(scores_eval_drqn)
                print('Best model is: ' + str(max_score_eval_drqn))
                # load the model with higher score
                model_drqn = keras.models.load_model(
                    weights_path + 'drqn_' + algorithm + '/model_weights_' + str(max_score_eval_drqn))

            else:
                model_drqn = keras.models.load_model('model_weights_drqn')

            agent_drqn = DRQNAgent(environment.observation_space.n, environment.action_space.n, config.trace_length, config,
                                   model=model_drqn)

            # Evaluate all deep networks together
            print('Evaluating neural networks')
            algorithm = 'original'
            drqn_score, drqn_stats = drqn_evaluate(environment, agent_drqn, algorithm,
                                               trace_length=config.trace_length, rep=config.drqn_evaluate_reps)
            dqn_score, dqn_stats = dqn_evaluate(environment, dqn_agent, algorithm, rep=config.dqn_evaluate_reps)


            fig_nn_comp, ax_nn_comp = plt.subplots(1, 2, num='nn_comp')
            fig_nn_comp.suptitle('Deep networks comparison')
            ax_nn_comp[0].hist(drqn_stats[0], label='drqn', bins='auto')
            ax_nn_comp[0].hist(dqn_stats[0], label='dqn', bins='auto')
            ax_nn_comp[0].set_xlabel('score')
            ax_nn_comp[0].set_ylabel('frequency')
            ax_nn_comp[0].legend()
            #ax_nn_comp[0].set_title('Score')

            plot_stats(ax_nn_comp[1], drqn_stats[1], drqn_stats[2], drqn_stats[3], label='drqn', plot_style='hist')
            plot_stats(ax_nn_comp[1], dqn_stats[1], dqn_stats[2], dqn_stats[3], label='dqn', plot_style='hist')
            ax_nn_comp[1].set_title('Stats')
            ax_nn_comp[1].set_xlabel('ratio')
            ax_nn_comp[1].set_ylabel('frequency')

            fig_nn_comp.savefig(results_path + 'nn_comparison.png')

            # add the values to the global vars

            scores_dqn.append(dqn_score)
            stats_dqn.append([np.mean((dqn_stats[2] + dqn_stats[3])/ dqn_stats[1]),
                              np.mean(dqn_stats[2]/(dqn_stats[2] + dqn_stats[3]))])

            scores_drqn.append(drqn_score)
            stats_drqn.append([np.mean((drqn_stats[2] + drqn_stats[3]) / drqn_stats[1]),
                              np.mean(drqn_stats[2] / (drqn_stats[2] + drqn_stats[3]))])
            print('\n')

            print('Getting score and stats for ql and vi')
            # get the mean score and mean statistics for every algorithm
            vi_score, vi_stats1, vi_stats2 =  evaluate(config, q_vi, environment, config.rep_eval_pol)
            ql_score, ql_stats1, ql_stats2 = evaluate(config, q_ql, environment, config.rep_eval_pol)

            # add the mean values
            scores_vi.append(np.mean(vi_score))
            stats_vi.append([np.mean(vi_stats1), np.mean(vi_stats2)])

            scores_ql.append(np.mean(ql_score))
            stats_ql.append([np.mean(ql_stats1), np.mean(ql_stats2)])


        # always evaluate the optimal policy
        opt_score, opt_stats1, opt_stats2 = evaluate(config, q_opt, environment, config.rep_eval_pol)
        scores_opt.append(np.mean(opt_score))
        stats_opt.append([np.mean(opt_stats1), np.mean(opt_stats2)])

    print('Plotting results after evaluating all values')
    print('Variable : ', changing_variable)
    print(print_values)
    print('\n---- Optimal policy ----- \n Scores: \n', scores_opt, '\n (T+C)/Total: \n', [x[0] for x in stats_opt],
          '\n T/(T+C): \n', [x[1] for x in stats_opt])

    # plot the numbers

    fig_eval, ax_eval = plt.subplots(1, 2, num='evaluation')
    colors = plt.get_cmap('tab20').colors
    colors_tab10 = plt.get_cmap('tab10').colors
    ax_eval[0].set_prop_cycle(color=colors_tab10)
    ax_eval[1].set_prop_cycle(color=colors)
    fig_eval.suptitle('Evaluation over variable ' + changing_variable)
    # score
    ax_eval[0].plot(print_values, scores_opt, label='opt', marker='o', markersize=3)
    ax_eval[0].set_xlabel(changing_variable)
    ax_eval[0].set_ylabel('score')

    # stats
    ax_eval[1].plot(print_values, [x[0] for x in stats_opt], label='(T+C)/Total opt',marker='o', markersize=3)
    ax_eval[1].plot(print_values, [x[1] for x in stats_opt], label='T/(T+C) opt',marker='o', markersize=3)
    ax_eval[1].set_xlabel(changing_variable)
    ax_eval[1].set_ylabel('ratio')

    if config.flag_train_all:
        print('\n---- Value iteration ----- \n Scores: \n', scores_vi, '\n (T+C)/Total: \n', [x[0] for x in stats_vi],
              '\n T/(T+C): \n', [x[1] for x in stats_vi])
        ax_eval[0].plot(print_values, scores_vi, label='vi',marker='o', markersize=3)
        ax_eval[1].plot(print_values, [x[0] for x in stats_vi], label='(T+C)/Total vi',marker='o', markersize=3)
        ax_eval[1].plot(print_values, [x[1] for x in stats_vi], label='T/(T+C) vi',marker='o', markersize=3)

        print('\n---- Q-learning ----- \n Scores: \n', scores_ql, '\n (T+C)/Total: \n', [x[0] for x in stats_ql],
              '\n T/(T+C): \n', [x[1] for x in stats_ql])
        ax_eval[0].plot(print_values, scores_ql, label='ql',marker='o', markersize=3)
        ax_eval[1].plot(print_values, [x[0] for x in stats_ql], label='(T+C)/Total ql',marker='o', markersize=3)
        ax_eval[1].plot(print_values, [x[1] for x in stats_ql], label='T/(T+C) ql',marker='o', markersize=3)

        if config.flag_train_dqn:
            print('\n---- DQN ----- \n Scores: \n', scores_dqn, '\n (T+C)/Total: \n',
                  [x[0] for x in stats_dqn], '\n T/(T+C): \n', [x[1] for x in stats_dqn])
            ax_eval[0].plot(print_values, scores_dqn, label='dqn',marker='o', markersize=3)
            ax_eval[1].plot(print_values, [x[0] for x in stats_dqn], label='(T+C)/Total dqn',marker='o', markersize=3)
            ax_eval[1].plot(print_values, [x[1] for x in stats_dqn], label='T/(T+C) dqn',marker='o', markersize=3)

            print('\n---- DRQN ----- \n Scores: \n', scores_drqn, '\n (T+C)/Total: \n',
                  [x[0] for x in stats_drqn], '\n T/(T+C): \n', [x[1] for x in stats_drqn])
            ax_eval[0].plot(print_values, scores_drqn, label='drqn',marker='o', markersize=3)
            ax_eval[1].plot(print_values, [x[0] for x in stats_drqn], label='(T+C)/Total drqn',marker='o', markersize=3)
            ax_eval[1].plot(print_values, [x[1] for x in stats_drqn], label='T/(T+C) drqn',marker='o', markersize=3)

    ax_eval[0].legend()
    ax_eval[1].legend()

    # save fig
    fig_eval.savefig('results/' + date_time + '/' + 'eval_over_variable.png')
    print('\n')
