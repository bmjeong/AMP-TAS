import numpy as np
import numpy.linalg as LA
import copy
import sys
import math

class Setting:
    def __init__(self, prob, buff_size=2, N_iter=30):
        self.N_agent = prob.N_agent
        self.N_task = prob.N_task
        self.capacity = prob.capacity # 하나의 값
        self.max_depth = self.capacity

        self.discount = prob.discount

        # Agent Type, Task Type 관련된 것
        self.N_agent_type = prob.N_agent_type
        self.N_task_type = prob.N_task_type
        self.feasibility = prob.feasibility

        self.agent_type = prob.agent_type

        self.task_reward = prob.task_reward # vector size : N_task
        self.task_type = prob.task_type # vector size : N_task
        self.map_size = prob.map_size
        self.agent_position = prob.agent_position
        self.task_position = prob.task_position

        # tuning parameter for exp - value normalization
        self.base_dist = prob.base_dist
        self.agent_vel = prob.agent_vel # vector size : N_agent
        self.task_time = prob.task_time # vector size : N_task

        self.buff_size = buff_size
        self.N_iter = N_iter

        self.dist_a2t = np.zeros((self.N_agent, self.N_task))
        for i in range(self.N_agent):
            for j in range(self.N_task):
                self.dist_a2t[i, j] = np.sqrt((self.task_position[j, 0] - self.agent_position[i, 0]) ** 2
                                              + (self.task_position[j, 1] - self.agent_position[i, 1]) ** 2)

        self.dist_t2t = np.zeros((self.N_task, self.N_task))
        for i in range(self.N_task):
            for j in range(self.N_task):
                self.dist_t2t[i, j] = np.sqrt((self.task_position[i, 0] - self.task_position[j, 0]) ** 2
                                              + (self.task_position[i, 1] - self.task_position[j, 1]) ** 2)

        self.norm_a2t_cost = np.zeros((self.N_agent, self.N_task))
        self.norm_t2a_cost = np.zeros((self.N_agent, self.N_task))
        self.norm_t2t_cost = np.zeros((self.N_task, self.N_task))
        for i in range(self.N_task):
            for j in range(self.N_agent):
                self.norm_a2t_cost[j, i] = self.dist_a2t[j, i] / self.base_dist
                self.norm_t2a_cost[j, i] = self.norm_a2t_cost[j, i]
            for j in range(self.N_task):
                self.norm_t2t_cost[i, j] = self.dist_t2t[i, j] / self.base_dist

class Agent:
    def __init__(self, id_a, setting):
        task_list = np.arange(0, setting.N_task)
        self.id = id_a
        self.type = setting.agent_type[id_a]
        feasible_task = setting.feasibility[self.type, setting.task_type] == 1
        self.state = setting.agent_position[id_a, :]
        self.x = self.state[0]
        self.y = self.state[1]
        self.cap = setting.capacity
        self.task = task_list[feasible_task] # absolute index
        self.N_task = len(self.task)
        self.task_type = setting.task_type[feasible_task]
        self.task_reward = setting.task_reward[feasible_task]
        # self.value_task = np.exp(self.task_reward)
        temp_term = self.task_reward / 2

        self.route = []
        self.route_length = 0

        self.vel = setting.agent_vel[id_a]

        self.route_time = []

class Task:
    def __init__(self, id_t, setting):
        self.id = id_t
        self.type = setting.task_type[id_t]
        self.x = setting.task_position[id_t, 0]
        self.y = setting.task_position[id_t, 1]
        self.duration = setting.task_time[id_t]
        self.discount = setting.discount
        self.reward = setting.task_reward[id_t]
        self.start = 0

class Result:
    def __init__(self, setting):
        self.N_agent = setting.N_agent
        self.N_task = setting.N_task
        self.map_size = setting.map_size
        self.route = [[] for i in range(setting.N_agent)]
        self.agent_position = setting.agent_position
        self.task_position = setting.task_position
        self.task_time = setting.task_time
        self.cost = np.zeros(setting.N_iter)
        self.reward = np.zeros(setting.N_iter)
        self.cvg_msg_rate = np.zeros(setting.N_iter)
        self.msg_t2a = np.zeros((setting.N_iter, 1, setting.N_agent, setting.N_task))

        self.record_time_arr = np.zeros(setting.N_task)

        self.total_t = 0
        self.total_r = 0

        self.too_many = False

class CBBA:
    def __init__(self, prob):
        self.prob = prob
        setting = Setting(prob)
        self.task_reward = setting.task_reward
        self.agent_position = setting.agent_position
        self.task_position = setting.task_position
        self.task_time = setting.task_time

        self.setting = setting

        self.base_dist = setting.base_dist
        self.N_agent = setting.N_agent
        self.N_task = setting.N_task
        self.max_depth = setting.max_depth
        # print("max_depth", self.max_depth)

        self.graph = np.logical_not(np.identity(self.N_agent)).tolist()

        self.AgentList = []
        for i in range(self.N_agent):
            self.AgentList.append(Agent(i, setting))

        self.TaskList = []
        for i in range(self.N_task):
            self.TaskList.append(Task(i, setting))

        # initialize these properties
        self.bundle_list = [[-1] * self.max_depth for _ in range(self.N_agent)]
        self.path_list = [[-1] * self.max_depth for _ in range(self.N_agent)]
        self.times_list = [[-1] * self.max_depth for _ in range(self.N_agent)]
        self.scores_list = [[-1] * self.max_depth for _ in range(self.N_agent)]

        # fixed the initialization, from 0 vector to -1 vector
        self.bid_list = [[-1] * self.N_task for _ in range(self.N_agent)]
        self.winners_list = [[-1] * self.N_task for _ in range(self.N_agent)]
        self.winner_bid_list = [[-1] * self.N_task for _ in range(self.N_agent)]

        self.agent_index_list = []
        for n in range(self.N_agent):
            self.agent_index_list.append(self.AgentList[n].id)

        self.too_many = False

    def solve(self):
        """
        Main CBBA Function
        """

        # Initialize working variables
        # Current iteration
        iter_idx = 1
        # Matrix of time of updates from the current winners
        time_mat = [[0] * self.N_agent for _ in range(self.N_agent)]
        iter_prev = 0
        done_flag = False

        # Main CBBA loop (runs until convergence)
        while not done_flag:
            # 1. Communicate
            # Perform consensus on winning agents and bid values (synchronous)
            time_mat = self.communicate(time_mat, iter_idx)

            # 2. Run CBBA bundle building/updating
            # Run CBBA on each agent (decentralized but synchronous)
            # print(iter_idx)
            for idx_agent in range(self.N_agent):
                new_bid_flag = self.bundle(idx_agent)
                # Update last time things changed
                # needed for convergence but will be removed in the final implementation
                if new_bid_flag:
                    iter_prev = iter_idx

                # pp = []
                # bb = []
                # for p in self.path_list[idx_agent]:
                #     pp.append(int(p))
                #     if int(p) == -1:
                #         bb.append(-1)
                #     else:
                #         bb.append(self.bid_list[idx_agent][int(p)])
                # print(idx_agent, pp, bb, self.bid_list[idx_agent], new_bid_flag)

            # 3. Convergence Check
            # Determine if the assignment is over (implemented for now, but later this loop will just run forever)
            if (iter_idx - iter_prev) > self.N_agent:
                done_flag = True
            elif (iter_idx - iter_prev) > (2 * self.N_agent):
                print("Algorithm did not converge due to communication trouble")

                done_flag = True
            else:
                # Maintain loop
                iter_idx += 1

            if iter_idx > 1000:
                self.too_many = True
                print("Too many iterations")
                done_flag = True

            # print(iter_idx, new_bid_flag, iter_prev)
            # for idx_agent in range(self.N_agent):
            #     pp = []
            #     bb = []
            #     for p in self.path_list[idx_agent]:
            #         pp.append(int(p))
            #         if int(p) == -1:
            #             bb.append(-1)
            #         else:
            #             bb.append(self.bid_list[idx_agent][int(p)])
            #     print(idx_agent, pp, bb)


        # for idx_agent in range(self.N_agent):
        #     ss = []
        #     for idx_task in range(self.max_depth):
        #         ss.append(float(self.scores_list[idx_agent][idx_task]))
        #     bb = []
        #     for idx_task in range(self.N_task):
        #         bb.append(float(self.winner_bid_list[idx_agent][idx_task]))
        #     print(ss)
        #     print(bb)


        # Map path and bundle values to actual task indices
        for n in range(self.N_agent):
            for m in range(self.max_depth):
                if self.bundle_list[n][m] == -1:
                    break
                else:
                    self.bundle_list[n][m] = self.TaskList[self.bundle_list[n][m]].id

                if self.path_list[n][m] == -1:
                    break
                else:
                    self.path_list[n][m] = self.TaskList[self.path_list[n][m]].id
        # Compute the total score of the CBBA assignment
        score_total = 0
        for n in range(self.N_agent):
            for m in range(self.max_depth):
                if self.scores_list[n][m] > -1:
                    score_total += self.scores_list[n][m]
                else:
                    break

        # output the result path for each agent, delete all -1
        self.path_list = [list(filter(lambda a: a != -1, self.path_list[i]))
                          for i in range(len(self.path_list))]
        # delete redundant elements
        self.bundle_list = [list(filter(lambda a: a != -1, self.bundle_list[i]))
                            for i in range(len(self.bundle_list))]
        self.times_list = [list(filter(lambda a: a != -1, self.times_list[i]))
                           for i in range(len(self.times_list))]
        self.scores_list = [list(filter(lambda a: a != -1, self.scores_list[i]))
                            for i in range(len(self.scores_list))]

        # for p in self.path_list:
        #     print(p)

        setting = Setting(self.prob)
        result = Result(setting)

        total_t = 0
        total_r = 0
        for i, p in enumerate(self.path_list):
            # print(p)
            # print(i, self.times_list[i])
            result.route[i] = p
            r = 0
            t = 0
            for j in range(len(p)):

                if self.setting.discount > 0:
                    # print(self.task_reward[p[j]])
                    # r += self.task_reward[p[j]]
                    r += self.task_reward[p[j]] * np.exp(-self.setting.discount * self.times_list[i][j])
                    # print(p[j], self.times_list[i][j], self.task_reward[p[j]] * np.exp(-self.setting.discount * self.times_list[i][j]))
                    # r += self.task_reward[p[j]]
                    # print(p[j], self.task_reward[p[j]] * np.exp(-self.setting.discount * self.times_list[i][j]))
                else:
                    r += self.task_reward[p[j]]
                # r += self.task_reward[p[j]]
                if j == 0:
                    t += LA.norm(self.task_position[p[j], :] - self.agent_position[i, :]) / self.AgentList[i].vel
                else:
                    t += LA.norm(self.task_position[p[j], :] - self.task_position[p[j -1], :]) / self.AgentList[i].vel
                result.record_time_arr[p[j]] = t
                t += self.task_time[p[j]]

            total_t += t
            total_r += r
        # print(total_t, total_r)
        result.total_t = total_t
        result.total_r = total_r

        if self.too_many:
            result.too_many = True

        return result

    def bundle(self, idx_agent: int):
        """
        Main CBBA bundle building/updating (runs on each individual agent)
        """

        # Update bundles after messaging to drop tasks that are outbid
        self.bundle_remove(idx_agent)
        # Bid on new tasks and add them to the bundle
        new_bid_flag = self.bundle_add(idx_agent)

        return new_bid_flag

    def bundle_remove(self, idx_agent: int):
        """
        Update bundles after communication
        For outbid agents, releases tasks from bundles
        """

        out_bid_for_task = False
        for idx in range(self.max_depth):
            # If bundle(j) < 0, it means that all tasks up to task j are
            # still valid and in paths, the rest (j to MAX_DEPTH) are released
            if self.bundle_list[idx_agent][idx] < 0:
                break
            else:
                # Test if agent has been outbid for a task.  If it has, release it and all subsequent tasks in its path.
                if self.winners_list[idx_agent][self.bundle_list[idx_agent][idx]] != self.agent_index_list[idx_agent]:
                    out_bid_for_task = True

                if out_bid_for_task:
                    # The agent has lost a previous task, release this one too
                    if self.winners_list[idx_agent][self.bundle_list[idx_agent][idx]] == \
                            self.agent_index_list[idx_agent]:
                        # Remove from winner list if in there
                        self.winners_list[idx_agent][self.bundle_list[idx_agent][idx]] = -1
                        self.winner_bid_list[idx_agent][self.bundle_list[idx_agent][idx]] = -1

                    # Clear from path and times vectors and remove from bundle
                    path_current = copy.deepcopy(self.path_list[idx_agent])
                    idx_remove = path_current.index(self.bundle_list[idx_agent][idx])

                    # remove item from list at location specified by idx_remove, then append -1 at the end.
                    del self.path_list[idx_agent][idx_remove]
                    self.path_list[idx_agent].append(-1)
                    del self.scores_list[idx_agent][idx_remove]
                    self.scores_list[idx_agent].append(-1)
                    del self.times_list[idx_agent][idx_remove]
                    self.times_list[idx_agent].append(-1)

                    self.bundle_list[idx_agent][idx] = -1

    def bundle_add(self, idx_agent: int):
        """
        Create bundles for each agent
        """

        epsilon = 1e-5
        new_bid_flag = False

        # Check if bundle is full, the bundle is full when bundle_full_flag is True
        index_array = np.where(np.array(self.bundle_list[idx_agent]) == -1)[0]
        if len(index_array) > 0:
            bundle_full_flag = False
        else:
            bundle_full_flag = True

        # Initialize feasibility matrix (to keep track of which j locations can be pruned)
        # feasibility = np.ones((self.N_task, self.max_depth+1))
        feasibility = [[1] * (self.max_depth + 1) for _ in range(self.N_task)]

        while not bundle_full_flag:
            # Update task values based on current assignment
            [best_indices, task_times, feasibility] = self.compute_bid(idx_agent, feasibility)

            # Determine which assignments are available. array_logical_1, array_logical_2,
            # array_logical_13 are all numpy 1D bool array
            array_logical_1 = ((np.array(self.bid_list[idx_agent]) - np.array(self.winner_bid_list[idx_agent]))
                               > epsilon)
            # find the equal items
            array_logical_2 = (abs(np.array(self.bid_list[idx_agent]) - np.array(self.winner_bid_list[idx_agent]))
                               <= epsilon)
            # Tie-break based on agent index
            array_logical_3 = (self.agent_index_list[idx_agent] < np.array(self.winners_list[idx_agent]))

            array_logical_result = np.logical_or(array_logical_1, np.logical_and(array_logical_2, array_logical_3))

            # Select the assignment that will improve the score the most and place bid
            array_max = np.array(self.bid_list[idx_agent]) * array_logical_result
            best_task = array_max.argmax()
            value_max = max(array_max)

            if value_max > 0:
                # print(idx_agent, value_max, np.where(array_max == value_max)[0])
                # print(np.array(self.bid_list[idx_agent]))
                # print(np.array(self.winner_bid_list[idx_agent]), array_max)
                # Set new bid flag
                new_bid_flag = True

                # Check for tie, return a 1D numpy array
                all_values = np.where(array_max == value_max)[0]

                # Check if there's only one task with the best bid value
                if len(all_values) == 1:
                    best_task = all_values[0]
                else:
                    # Tie-break by choosing the task with the earliest start time
                    earliest = sys.float_info.max
                    # best_task = None
                    for i in all_values:
                        if self.TaskList[i].start < earliest:
                            earliest = self.TaskList[i].start
                            best_task = i

                self.winners_list[idx_agent][best_task] = self.AgentList[idx_agent].id
                self.winner_bid_list[idx_agent][best_task] = self.bid_list[idx_agent][best_task]

                # Insert value into list at location specified by index, and delete the last one of original list.
                self.path_list[idx_agent].insert(best_indices[best_task], best_task)
                del self.path_list[idx_agent][-1]
                self.times_list[idx_agent].insert(best_indices[best_task], task_times[best_task])
                del self.times_list[idx_agent][-1]
                self.scores_list[idx_agent].insert(best_indices[best_task], self.bid_list[idx_agent][best_task])
                del self.scores_list[idx_agent][-1]

                length = len(np.where(np.array(self.bundle_list[idx_agent]) > -1)[0])
                self.bundle_list[idx_agent][length] = best_task

                # Update feasibility
                # This inserts the same feasibility boolean into the feasibility matrix
                for i in range(self.N_task):
                    # Insert value into list at location specified by index, and delete the last one of original list.
                    feasibility[i].insert(best_indices[best_task], feasibility[i][best_indices[best_task]])
                    del feasibility[i][-1]
            else:
                break

            # Check if bundle is full
            index_array = np.where(np.array(self.bundle_list[idx_agent]) == -1)[0]
            # print(idx_agent, index_array)
            if len(index_array) > 0:
                bundle_full_flag = False
            else:
                bundle_full_flag = True
        return new_bid_flag

    def communicate(self, time_mat: list, iter_idx: int):
        """
        Runs consensus between neighbors. Checks for conflicts and resolves among agents.
        This is a message passing scheme described in Table 1 of: "Consensus-Based Decentralized Auctions for
        Robust Task Allocation", H.-L. Choi, L. Brunet, and J. P. How, IEEE Transactions on Robotics,
        Vol. 25, (4): 912 - 926, August 2009

        Note: Table 1 is the action rule for agent i based on communication with agent k regarding task j.
        The big for-loop with tons of if-else is the exact implementation of Table 1, for the sake of readability.
        """

        # time_mat is the matrix of time of updates from the current winners
        # iter_idx is the current iteration

        time_mat_new = copy.deepcopy(time_mat)

        # Copy data
        old_z = copy.deepcopy(self.winners_list)
        old_y = copy.deepcopy(self.winner_bid_list)
        z = copy.deepcopy(old_z)
        y = copy.deepcopy(old_y)

        epsilon = 10e-6

        # Start communication between agents
        # sender   = k
        # receiver = i
        # task     = j

        for k in range(self.N_agent):
            for i in range(self.N_agent):
                if self.graph[k][i] == 1:
                    for j in range(self.N_task):
                        # Implement table for each task

                        # Entries 1 to 4: Sender thinks he has the task
                        if old_z[k][j] == k:

                            # Entry 1: Update or Leave
                            if z[i][j] == i:
                                if (old_y[k][j] - y[i][j]) > epsilon:  # Update
                                    z[i][j] = old_z[k][j]
                                    y[i][j] = old_y[k][j]
                                elif abs(old_y[k][j] - y[i][j]) <= epsilon:  # Equal scores
                                    if z[i][j] > old_z[k][j]:  # Tie-break based on smaller index
                                        z[i][j] = old_z[k][j]
                                        y[i][j] = old_y[k][j]

                            # Entry 2: Update
                            elif z[i][j] == k:
                                z[i][j] = old_z[k][j]
                                y[i][j] = old_y[k][j]

                            # Entry 3: Update or Leave
                            elif z[i][j] > -1:
                                if time_mat[k][z[i][j]] > time_mat_new[i][z[i][j]]:  # Update
                                    z[i][j] = old_z[k][j]
                                    y[i][j] = old_y[k][j]
                                elif (old_y[k][j] - y[i][j]) > epsilon:  # Update
                                    z[i][j] = old_z[k][j]
                                    y[i][j] = old_y[k][j]
                                elif abs(old_y[k][j] - y[i][j]) <= epsilon:  # Equal scores
                                    if z[i][j] > old_z[k][j]:  # Tie-break based on smaller index
                                        z[i][j] = old_z[k][j]
                                        y[i][j] = old_y[k][j]

                            # Entry 4: Update
                            elif z[i][j] == -1:
                                z[i][j] = old_z[k][j]
                                y[i][j] = old_y[k][j]

                            else:
                                print(z[i][j])
                                raise Exception("Unknown winner value: please revise!")

                        # Entries 5 to 8: Sender thinks receiver has the task
                        elif old_z[k][j] == i:

                            # Entry 5: Leave
                            if z[i][j] == i:
                                # Do nothing
                                pass

                            # Entry 6: Reset
                            elif z[i][j] == k:
                                z[i][j] = -1
                                y[i][j] = -1

                            # Entry 7: Reset or Leave
                            elif z[i][j] > -1:
                                if time_mat[k][z[i][j]] > time_mat_new[i][z[i][j]]:  # Reset
                                    z[i][j] = -1
                                    y[i][j] = -1

                            # Entry 8: Leave
                            elif z[i][j] == -1:
                                # Do nothing
                                pass

                            else:
                                print(z[i][j])
                                raise Exception("Unknown winner value: please revise!")

                        # Entries 9 to 13: Sender thinks someone else has the task
                        elif old_z[k][j] > -1:

                            # Entry 9: Update or Leave
                            if z[i][j] == i:
                                if time_mat[k][old_z[k][j]] > time_mat_new[i][old_z[k][j]]:
                                    if (old_y[k][j] - y[i][j]) > epsilon:
                                        z[i][j] = old_z[k][j]  # Update
                                        y[i][j] = old_y[k][j]
                                    elif abs(old_y[k][j] - y[i][j]) <= epsilon:  # Equal scores
                                        if z[i][j] > old_z[k][j]:  # Tie-break based on smaller index
                                            z[i][j] = old_z[k][j]
                                            y[i][j] = old_y[k][j]

                            # Entry 10: Update or Reset
                            elif z[i][j] == k:
                                if time_mat[k][old_z[k][j]] > time_mat_new[i][old_z[k][j]]:  # Update
                                    z[i][j] = old_z[k][j]
                                    y[i][j] = old_y[k][j]
                                else:  # Reset
                                    z[i][j] = -1
                                    y[i][j] = -1

                            # Entry 11: Update or Leave
                            elif z[i][j] == old_z[k][j]:
                                if time_mat[k][old_z[k][j]] > time_mat_new[i][old_z[k][j]]:  # Update
                                    z[i][j] = old_z[k][j]
                                    y[i][j] = old_y[k][j]

                            # Entry 12: Update, Reset or Leave
                            elif z[i][j] > -1:
                                if time_mat[k][z[i][j]] > time_mat_new[i][z[i][j]]:
                                    if time_mat[k][old_z[k][j]] >= time_mat_new[i][old_z[k][j]]:  # Update
                                        z[i][j] = old_z[k][j]
                                        y[i][j] = old_y[k][j]
                                    elif time_mat[k][old_z[k][j]] < time_mat_new[i][old_z[k][j]]:  # Reset
                                        z[i][j] = -1
                                        y[i][j] = -1
                                    else:
                                        raise Exception("Unknown condition for Entry 12: please revise!")
                                else:
                                    if time_mat[k][old_z[k][j]] > time_mat_new[i][old_z[k][j]]:
                                        if (old_y[k][j] - y[i][j]) > epsilon:  # Update
                                            z[i][j] = old_z[k][j]
                                            y[i][j] = old_y[k][j]
                                        elif abs(old_y[k][j] - y[i][j]) <= epsilon:  # Equal scores
                                            if z[i][j] > old_z[k][j]:  # Tie-break based on smaller index
                                                z[i][j] = old_z[k][j]
                                                y[i][j] = old_y[k][j]

                            # Entry 13: Update or Leave
                            elif z[i][j] == -1:
                                if time_mat[k][old_z[k][j]] > time_mat_new[i][old_z[k][j]]:  # Update
                                    z[i][j] = old_z[k][j]
                                    y[i][j] = old_y[k][j]

                            else:
                                raise Exception("Unknown winner value: please revise!")

                        # Entries 14 to 17: Sender thinks no one has the task
                        elif old_z[k][j] == -1:

                            # Entry 14: Leave
                            if z[i][j] == i:
                                # Do nothing
                                pass

                            # Entry 15: Update
                            elif z[i][j] == k:
                                z[i][j] = old_z[k][j]
                                y[i][j] = old_y[k][j]

                            # Entry 16: Update or Leave
                            elif z[i][j] > -1:
                                if time_mat[k][z[i][j]] > time_mat_new[i][z[i][j]]:  # Update
                                    z[i][j] = old_z[k][j]
                                    y[i][j] = old_y[k][j]

                            # Entry 17: Leave
                            elif z[i][j] == -1:
                                # Do nothing
                                pass
                            else:
                                raise Exception("Unknown winner value: please revise!")

                            # End of table
                        else:
                            raise Exception("Unknown winner value: please revise!")

                    # Update timestamps for all agents based on latest comm
                    for n in range(self.N_agent):
                        if (n != i) and (time_mat_new[i][n] < time_mat[k][n]):
                            time_mat_new[i][n] = time_mat[k][n]
                    time_mat_new[i][k] = iter_idx

        # Copy data
        self.winners_list = copy.deepcopy(z)
        self.winner_bid_list = copy.deepcopy(y)
        return time_mat_new

    def compute_bid(self, idx_agent: int, feasibility: list):
        """
        Computes bids for each task. Returns bids, best index for task in
        the path, and times for the new path
        """

        # If the path is full then we cannot add any tasks to it
        empty_task_index_list = np.where(np.array(self.path_list[idx_agent]) == -1)[0]
        # print("empty_task_index_list", empty_task_index_list)
        if len(empty_task_index_list) == 0:
            best_indices = []
            task_times = []
            feasibility = []
            return best_indices, task_times, feasibility

        # Reset bids, best positions in path, and best times
        self.bid_list[idx_agent] = [-1] * self.N_task
        best_indices = [-1] * self.N_task
        task_times = [-2] * self.N_task

        # For each task
        for idx_task in range(self.N_task):
            # Check to make sure the path doesn't already contain task m
            index_array = np.where(np.array(self.path_list[idx_agent][0:empty_task_index_list[0]]) == idx_task)[0]
            if len(index_array) < 0.5:
                # this task not in my bundle yet
                # Find the best score attainable by inserting the score into the current path
                best_bid = 0
                best_index = -1
                best_time = -2

                # Try inserting task m in location j among other tasks and see if it generates a better new_path.
                for j in range(empty_task_index_list[0] + 1):
                    # print(best_bid)
                    if feasibility[idx_task][j] == 1:
                        skip = 0
                        # Check new path feasibility, true to skip this iteration, false to be feasible
                        if j == 0:
                            # insert at the beginning
                            task_prev = []
                            time_prev = []
                        else:
                            task_prev = self.TaskList[self.path_list[idx_agent][j - 1]]
                            time_prev = self.times_list[idx_agent][j - 1]
                        if j == (empty_task_index_list[0]):
                            task_next = []
                            time_next = []
                        else:
                            # task_id = self.path_list[idx_agent][j]
                            # task_next = Task(task_id, self.setting)
                            task_next = self.TaskList[self.path_list[idx_agent][j]]
                            time_next = self.times_list[idx_agent][j]

                        # Compute min and max start times and score
                        task_id = idx_task
                        task_cur = Task(task_id, self.setting)
                        score, min_start, max_start = self.scoring_compute_score(
                            idx_agent, task_cur, task_prev, task_next, time_prev, time_next)

                        if (min_start > max_start):
                            skip = 1

                        # no time window for tasks
                        # Save the best score and task position
                        # print(score, min_start, best_bid, skip)
                        if skip == 0:
                            if score > best_bid:
                                best_bid = score
                                best_index = j
                                # Select min start time as optimal
                                best_time = min_start
                        # print(j, best_bid, score, best_index)
                # save best bid information
                if best_bid > 0:
                    ### Bid-Warping
                    bids = []
                    for idx, task in enumerate(self.bundle_list[idx_agent]):
                        if task > -1:
                            bids.append(self.scores_list[idx_agent][idx])
                    # print(self.bundle_list[idx_agent][0], )
                    if bids:
                        if best_bid > min(bids):
                            print(min(bids), best_bid)
                    #         best_bid = min(best_bid, min(bids))



                    self.bid_list[idx_agent][idx_task] = best_bid
                    best_indices[idx_task] = best_index
                    task_times[idx_task] = best_time
        # print(self.bid_list[idx_agent], self.winner_bid_list[idx_agent], self.bundle_list[idx_agent])
            # this task is incompatible with my type
        # end loop through tasks
        # print(best_indices)
        return best_indices, task_times, feasibility

    # def compute_bid_nn(self, idx_agent: int, feasibility: list):
    #     """
    #     Computes bids for each task. Returns bids, best index for task in
    #     the path, and times for the new path
    #     """
    #
    #     # If the path is full then we cannot add any tasks to it
    #     empty_task_index_list = np.where(np.array(self.path_list[idx_agent]) == -1)[0]
    #     if len(empty_task_index_list) == 0:
    #         best_indices = []
    #         task_times = []
    #         feasibility = []
    #         return best_indices, task_times, feasibility
    #
    #     # Reset bids, best positions in path, and best times
    #     self.bid_list[idx_agent] = [-1] * self.N_task
    #     best_indices = [-1] * self.N_task
    #     task_times = [-2] * self.N_task
    #
    #     # For each task
    #     for idx_task in range(self.N_task):
    #         # Check to make sure the path doesn't already contain task m
    #         index_array = np.where(np.array(self.path_list[idx_agent][0:empty_task_index_list[0]]) == idx_task)[0]
    #         if len(index_array) < 0.5:
    #             # this task not in my bundle yet
    #             # Find the best score attainable by inserting the score into the current path
    #             best_bid = 0
    #             best_index = -1
    #             best_time = -2
    #
    #             # Try inserting task m in location j among other tasks and see if it generates a better new_path.
    #             for j in range(empty_task_index_list[0] + 1):
    #                 if feasibility[idx_task][j] == 1:
    #                     skip = 0
    #                     # Check new path feasibility, true to skip this iteration, false to be feasible
    #                     if j == 0:
    #                         # insert at the beginning
    #                         task_prev_id = -1
    #                         time_prev = -1
    #                     else:
    #                         task_prev_id = self.path_list[idx_agent][j - 1]
    #                         time_prev = self.times_list[idx_agent][j - 1]
    #                     if j == (empty_task_index_list[0]):
    #                         task_next_id = -1
    #                         time_next = -1
    #                     else:
    #                         task_next_id = self.path_list[idx_agent][j]
    #                         time_next = self.times_list[idx_agent][j]
    #
    #                     # Compute min and max start times and score
    #                     task_id = idx_task
    #                     # task_cur = Task(task_id, self.setting)
    #                     score, min_start, max_start = self.scoring_compute_score(
    #                         idx_agent, task_id, task_prev_id, task_next_id, time_prev, time_next)
    #
    #                     if (min_start > max_start):
    #                         # Infeasible path
    #                         skip = 1
    #                         # feasibility[idx_task, j] = 0
    #
    #                     # no time window for tasks
    #                     # Save the best score and task position
    #                     if skip == 0:
    #                         if score > best_bid:
    #                             best_bid = score
    #                             best_index = j
    #                             # Select min start time as optimal
    #                             best_time = min_start
    #
    #             # save best bid information
    #             if best_bid > 0:
    #                 self.bid_list[idx_agent][idx_task] = best_bid
    #                 best_indices[idx_task] = best_index
    #                 task_times[idx_task] = best_time
    #
    #         # this task is incompatible with my type
    #     # end loop through tasks
    #     return best_indices, task_times, feasibility
    #
    # def scoring_compute_score_(self, idx_agent: int, task_current: Task, task_prev: Task,
    #                           task_next: Task, time_prev, time_next):
    #     """
    #     Compute marginal score of doing a task and returns the expected start time for the task.
    #     """
    #     if not task_prev:
    #         # First task in path
    #         # Compute start time of task
    #         dt = math.sqrt((self.AgentList[idx_agent].x - task_current.x) ** 2 +
    #                        (self.AgentList[idx_agent].y - task_current.y) ** 2) / self.AgentList[
    #                  idx_agent].vel
    #     else:
    #         # Not first task in path
    #         dt = (math.sqrt((task_prev.x - task_current.x) ** 2 + (task_prev.y - task_current.y) ** 2)
    #               / self.AgentList[idx_agent].vel)
    #
    #     if not task_next:
    #         # Last task in path
    #         dt = 0.0
    #     else:
    #         # Not last task, check if we can still make promised task
    #         dt = (math.sqrt((task_next.x - task_current.x) ** 2 + (task_next.y - task_current.y) ** 2)
    #               / self.AgentList[idx_agent].vel)
    #
    #     # Compute score
    #     # no time window for tasks
    #     dt_current = math.sqrt((self.AgentList[idx_agent].x - task_current.x) ** 2 +
    #                            (self.AgentList[idx_agent].y - task_current.y) ** 2) / \
    #                  self.AgentList[idx_agent].vel
    #     # dt_current = dt
    #     reward = task_current.reward * math.exp((-task_current.discount) * dt_current)
    #
    #     # # Subtract fuel cost. Implement constant fuel to ensure DMG (diminishing marginal gain).
    #     # # This is a fake score since it double-counts fuel. Should not be used when comparing to optimal score.
    #     # # Need to compute real score of CBBA paths once CBBA algorithm has finished running.
    #     # penalty = self.AgentList[idx_agent].fuel * math.sqrt(
    #     #     (self.AgentList[idx_agent].x-task_current.x)**2 + (self.AgentList[idx_agent].y-task_current.y)**2 +
    #     #     (self.AgentList[idx_agent].z-task_current.z)**2)
    #     # #
    #     # score = reward - penalty
    #
    #     score = reward
    #
    #     return score
    #
    # def scoring_compute_score_nn(self, idx_agent, task_cur_id, task_prev_id, task_next_id, time_prev, time_next):
    #     if task_prev_id == -1:
    #
    #         # Compute start time of task
    #         dt = (math.sqrt((self.AgentList[idx_agent].x - self.TaskList[task_cur_id].x) ** 2 +
    #                        (self.AgentList[idx_agent].y - self.TaskList[task_cur_id].y) ** 2) /
    #               self.AgentList[idx_agent].vel)
    #         min_start = max(self.TaskList[task_cur_id].start, dt)
    #         penalty = dt
    #     else:
    #         # Not first task in path
    #         dt = (math.sqrt((self.TaskList[task_prev_id].x - self.TaskList[task_cur_id].x) ** 2 +
    #                         (self.TaskList[task_prev_id].y - self.TaskList[task_cur_id].y) ** 2) /
    #               self.AgentList[idx_agent].vel)
    #         min_start = max(self.TaskList[task_cur_id].start, time_prev + self.TaskList[task_prev_id].duration + dt)
    #         penalty = dt
    #
    #         # Last task in path
    #     if task_next_id == -1:
    #         max_start = self.TaskList[task_cur_id].start + self.TaskList[task_cur_id].duration
    #     else:
    #         # Not last task, check if we can still make promised task
    #         dt = (math.sqrt((self.TaskList[task_next_id].x - self.TaskList[task_cur_id].x) ** 2 +
    #                         (self.TaskList[task_next_id].y - self.TaskList[task_cur_id].y) ** 2)
    #               / self.AgentList[idx_agent].vel)
    #         max_start = min(self.TaskList[task_cur_id].start + self.TaskList[task_cur_id].duration,
    #                         time_next - self.TaskList[task_cur_id].duration - dt)
    #
    #         # Compute score
    #     reward = self.TaskList[task_cur_id].reward * math.exp(-self.TaskList[task_cur_id].discount *
    #                                                           (min_start - self.TaskList[task_cur_id].start))
    #     # print(min_start - task_current.start)
    #     # Subtract fuel cost.  Implement constant fuel to ensure DMG.
    #     # NOTE: This is a fake score since it double counts fuel.
    #     # print(min_start, reward * 1000, penalty)
    #     score = reward * 1000 - penalty
    #     # score = reward
    #
    #     return score, min_start, max_start

    def scoring_compute_score(self, idx_agent, task_current, task_prev, task_next, time_prev, time_next):
        if not task_prev:

            # Compute start time of task
            dt = (math.sqrt((self.AgentList[idx_agent].x - task_current.x) ** 2 +
                           (self.AgentList[idx_agent].y - task_current.y) ** 2) /
                  self.AgentList[idx_agent].vel)
            min_start = max(task_current.start, dt)
            penalty = dt + task_current.duration
        else:
            # Not first task in path
            dt = (math.sqrt((task_prev.x - task_current.x) ** 2 +
                            (task_prev.y - task_current.y) ** 2) /
                  self.AgentList[idx_agent].vel)
            min_start = max(task_current.start, time_prev + task_prev.duration + dt)
            penalty = dt + task_current.duration

            # Last task in path
        if not task_next:
            max_start = 10000
        else:
            # Not last task, check if we can still make promised task
            dt = (math.sqrt((task_next.x - task_current.x) ** 2 +
                            (task_next.y - task_current.y) ** 2)
                  / self.AgentList[idx_agent].vel)
            max_start = min(10000,
                            time_next - task_current.duration - dt)

            # Compute score
        reward = task_current.reward * math.exp(-task_current.discount *
                                                              (min_start - task_current.start))
        # print("task_current.reward", task_current.reward, reward)
        # print(min_start - task_current.start)
        # Subtract fuel cost.  Implement constant fuel to ensure DMG.
        # NOTE: This is a fake score since it double counts fuel.
        # print(min_start, reward * 1000, penalty)
        score = reward - penalty / self.base_dist
        # if score < 0:
        #     print(reward, penalty, min_start)
        # score = reward

        return score, min_start, max_start