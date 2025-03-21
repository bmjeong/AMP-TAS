import matplotlib.pyplot as plt
import numpy as np

def plot_result(result):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 6), gridspec_kw={'width_ratios': [1, 2]})

    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # print(cycle)

    N_agent = result.N_agent
    N_task = result.N_task
    agent_position = result.agent_position
    task_position = result.task_position
    path = result.route
    map_size = result.map_size

    for id_a in range(N_agent):
        # ax[0].text(agent_position[id_a, 0], agent_position[id_a, 1],
        #            str(id_a),
        #            fontsize=5, color='b')
        ax[0].text(agent_position[id_a, 0]-15, agent_position[id_a, 1]-20,
                   str(id_a),
                   fontsize=10, color='white')
        for id_loc in range(len(path[id_a])):
            if id_loc == 0:
                xx = [agent_position[id_a, 0], task_position[path[id_a][id_loc], 0]]
                yy = [agent_position[id_a, 1], task_position[path[id_a][id_loc], 1]]
                ax[0].plot(xx, yy, color=cycle[id_a % len(cycle)], linewidth=1.5)
            else:
                xx = [task_position[path[id_a][id_loc - 1], 0], task_position[path[id_a][id_loc], 0]]
                yy = [task_position[path[id_a][id_loc - 1], 1], task_position[path[id_a][id_loc], 1]]
                ax[0].plot(xx, yy, color=cycle[id_a % len(cycle)], linewidth=1.5)

            ax[0].plot(task_position[path[id_a][id_loc], 0], task_position[path[id_a][id_loc], 1], 'o',
                       markersize=8, markerfacecolor='k', markeredgecolor=cycle[id_a % len(cycle)])

        if len(path[id_a]) > 0:
            ax[0].plot(agent_position[id_a, 0], agent_position[id_a, 1], 's', markersize=10, markerfacecolor='b',
                       markeredgecolor=cycle[id_a % len(cycle)])
        else:
            ax[0].plot(agent_position[id_a, 0], agent_position[id_a, 1], 's', markersize=10, markerfacecolor='b',
                       markeredgecolor='b')

    for id_t in range(N_task):
        ax[0].plot(task_position[id_t, 0], task_position[id_t, 1], 'o', markersize=6, markerfacecolor='k',
                   markeredgecolor='k')
        if id_t < 10:
            ax[0].text(task_position[id_t, 0] - 10, task_position[id_t, 1] - 10, str(id_t), fontsize=5, color='white')
        else:
            ax[0].text(task_position[id_t, 0] - 20, task_position[id_t, 1] - 10, str(id_t), fontsize=5, color='white')

        # ax[0].text(task_position[id_t, 0]+3, task_position[id_t, 1]+int(id_t/N_agent)*1-2, str(id_t),
        #            color='k', fontsize=8)
        # ax[0].plot([task_position[id_t, 0], task_position[id_t, 0]+3],
        #            [task_position[id_t, 1], task_position[id_t, 1]+int(id_t/N_agent)*1-2], 'k', linewidth=0.5)
    # ax[0].set_xlim([map_size[0, 0]+20, map_size[0, 1]+20])
    # ax[0].set_ylim([map_size[1, 0]-10, map_size[1, 1]-30])
    ax[0].set_xlim([map_size[0, 0]-10, map_size[0, 1]+10])
    ax[0].set_ylim([map_size[1, 0]-10, map_size[1, 1]+10])
    ax[0].set_xlabel('x (m)', fontsize=15)
    ax[0].set_ylabel('y (m)', fontsize=15)
    ax[0].grid(True)
    # plt.show()

    tt = np.zeros(N_agent)
    for k in range(N_agent):
        if path[k]:
            for l_ind, l in enumerate(path[k]):
                t_sta = result.record_time_arr[l]
                t_lea = t_sta + result.task_time[l]
                # a1 = [float(t_arr[l]), float(t_sta[l]), float(t_sta[l]), float(t_arr[l])]
                # print(l, a1)
                a2 = [k - 0.25, k - 0.25, k + 0.25, k + 0.25]
                a3 = [float(t_sta), float(t_lea), float(t_lea), float(t_sta)]
    #
    #             ax[1].fill(a1, a2, color='gray')
                ax[1].fill(a3, a2, color='skyblue')
    #             # ax[1].plot([float(t_arr[l]), float(t_sta[l])], [k, k], color='gray', linewidth=10)
    #             # ax[1].plot([float(t_sta[l]), float(t_lea[l])], [k, k], color='skyblue', linewidth=10)
                ax[1].text(sum([t_sta, t_lea]) / 2 + 5, k+0.35*((l_ind%2)*2-1)-0.05, str(l), fontsize=8, color='b')
                ax[1].plot([sum([t_sta, t_lea]) / 2, sum([t_sta, t_lea]) / 2+5], [k+0.25*((l_ind%2)*2-1), k+0.3*((l_ind%2)*2-1)],'k',linewidth=0.5)
                if l == path[k][-1]:
                    tt[k] = t_lea
    #
    ax[1].set_xlim([0, int(max(tt) / 1000 + 1)*1000])
    ax[1].set_ylim([-0.5, N_agent - 0.5])
    ax[1].set_xlabel('time (sec)', fontsize=15)
    ax[1].set_ylabel('Agent Index', fontsize=15)
    ax[1].grid(True)
