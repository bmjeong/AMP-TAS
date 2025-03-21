import numpy as np
import copy
import numpy.linalg as LA
import time

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
    def __init__(self, id_a, setting, build_type, conf_res_type, refine_type, return_type):

        self.build_type = build_type
        self.conf_res_type = conf_res_type
        # if conf_res_type == 1:
        #     self.build_type = 1
        self.refine_type = refine_type
        self.return_type = return_type

        self.discount = setting.discount

        self.N_iter = setting.N_iter
        task_list = np.arange(0, setting.N_task)
        self.id = id_a
        self.type = setting.agent_type[id_a]
        feasible_task = setting.feasibility[self.type, setting.task_type] == 1
        self.state = setting.agent_position[id_a, :]
        self.cap = setting.capacity
        self.task = task_list[feasible_task] # absolute index
        self.N_task = len(self.task)

        self.base_dist = setting.base_dist

        self.vel = setting.agent_vel[id_a]

        self.task_type = setting.task_type[feasible_task]
        self.task_reward = setting.task_reward[feasible_task]
        self.task_time = setting.task_time[feasible_task]

        self.max_task_reward = max(max(self.task_reward), 1)
        # self.value_task = np.exp(self.task_reward)
        self.cost_a2t = setting.norm_a2t_cost[id_a, self.task]
        self.cost_t2t = np.zeros((self.N_task, self.N_task))

        self.value_a2t = np.zeros(self.N_task)
        self.value_t2t = np.zeros((self.N_task, self.N_task))
        self.value_t2a = np.zeros(self.N_task)

        self.value_a2t_cost = np.zeros(self.N_task)
        self.value_t2a_cost = np.zeros(self.N_task)
        self.value_t2t_cost = np.zeros((self.N_task, self.N_task))
        for i1, t1 in enumerate(self.task):
            self.value_a2t[i1] = np.exp(
                (self.task_reward[t1] - self.cost_a2t[i1] / self.vel
                 - self.task_time[t1] / self.base_dist) / self.max_task_reward)
            self.value_a2t_cost[i1] = np.exp(- self.cost_a2t[i1] / self.vel)

            if self.return_type:
                self.value_t2a[i1] = np.exp(
                    ( - self.cost_a2t[i1] / self.vel
                     - self.task_time[t1] / self.base_dist) / self.max_task_reward)
                self.value_t2a_cost[i1] = self.value_a2t_cost[i1]
            else:
                self.value_t2a[i1] = 1
                self.value_t2a_cost[i1] = 1

            for i2, t2 in enumerate(self.task):
                self.cost_t2t[i1, i2] = setting.norm_t2t_cost[t1, t2]
                self.value_t2t[i1, i2] = np.exp(
                    (self.task_reward[i2] - self.cost_t2t[i1, i2] / self.vel
                     - self.task_time[t2] / self.base_dist) / self.max_task_reward)
                self.value_t2t_cost[i1, i2] = np.exp(- self.cost_t2t[i1, i2] / self.vel)

        self.route = []
        self.route_time = []
        self.route_prev = []

        self.route_cnt = 0

        self.route_cta = [[] for i in range(setting.N_agent)]
        self.route_length = 0
        self.msg_a2t = np.ones(self.N_task) / 2

        # self.msg_a2t_prev = np.ones(self.N_task) / 2

        self.msg_a2t_sav = np.ones((setting.N_agent, self.N_task)) / 2

        self.msg_t2a = np.ones(self.N_task) / 2
        self.buff_a2t = np.ones((setting.buff_size, self.N_task)) / 2
        self.buff_t2a = np.ones((setting.buff_size, self.N_task)) / 2
        self.buff_id = 0

        self.decision = np.zeros(self.N_task)


        self.N_task_total = setting.N_task
        self.N_agent_total = setting.N_agent
        self.da2t = setting.dist_a2t
        self.dt2t = setting.dist_t2t
        self.task_time = setting.task_time
        self.base_dist = setting.base_dist

        self.finish_flag = False
        self.fin_cnt = 0

        self.total_finish_flag = False
        self.total_fin_cnt = 0

        self.refine_flag = False

        self.iter_cnt = 0

        self.id_a4t = np.zeros(self.N_task)

    def cal_score(self, route):
        route_time = np.zeros(len(route))
        route_reward = np.zeros(len(route))
        t = 0
        for i in range(len(route)):
            if i == 0:
                t += self.cost_a2t[route[i]] * self.base_dist / self.vel
            else:
                t += self.cost_t2t[route[i - 1], route[i]] * self.base_dist / self.vel
            route_reward[i] = self.task_reward[route[i]] * np.exp(-self.discount * t)
            t += self.task_time[route[i]]
            route_time[i] = t
        return sum(route_reward) - route_time[-1] / self.base_dist

    def two_opt(self):
        route = self.route
        ## time
        # route_time = self.route_time
        N_pck = self.route_length
        value_a2t_cost = self.value_a2t_cost
        value_t2t_cost = self.value_t2t_cost
        value_t2a_cost = self.value_t2a_cost
        if N_pck > 0:
            if self.discount > 0:
                score = self.cal_score(route)

                toggle = True
                while toggle:
                    toggle = False
                    for id_loc_1 in range(N_pck - 1):
                        for id_loc_2 in range(id_loc_1 + 1,
                                              N_pck):  # searching the half of the solution space: efficient when costs are symmetric or scheduling no returning path
                            route_tp = copy.deepcopy(route)
                            route_tp[id_loc_1:id_loc_2 + 1] = np.flip(route_tp[id_loc_1:id_loc_2 + 1])
                            score_tp = self.cal_score(route_tp)
                            if score_tp > score:  # just for symmetric costs
                                toggle = True
                                route[id_loc_1:id_loc_2 + 1] = np.flip(route[id_loc_1:id_loc_2 + 1])
                                score = self.cal_score(route)
                            # if id_loc_1 == 0:
                            #     value_old_route_1 = value_a2t_cost[route[id_loc_1]]
                            #     value_new_route_1 = value_a2t_cost[route[id_loc_2]]
                            # else:
                            #     value_old_route_1 = value_t2t_cost[route[id_loc_1 - 1], route[id_loc_1]]
                            #     value_new_route_1 = value_t2t_cost[route[id_loc_1 - 1], route[id_loc_2]]
                            # if id_loc_2 == N_pck - 1:
                            #     value_old_route_2 = value_t2a_cost[route[id_loc_2]]
                            #     value_new_route_2 = value_t2a_cost[route[id_loc_1]]
                            # else:
                            #     value_old_route_2 = value_t2t_cost[route[id_loc_2], route[id_loc_2 + 1]]
                            #     value_new_route_2 = value_t2t_cost[route[id_loc_1], route[id_loc_2 + 1]]
                            #
                            # if value_old_route_1 * value_old_route_2 < value_new_route_1 * value_new_route_2:  # just for symmetric costs
                            #     toggle = True
                            #     route[id_loc_1:id_loc_2 + 1] = np.flip(route[id_loc_1:id_loc_2 + 1])
                            #     # route_time[id_loc_1:id_loc_2 + 1] = np.flip(route_time[id_loc_1:id_loc_2 + 1])
                self.decision = np.zeros(self.N_task)
                if len(route) == 1:
                    route = [route[0]]
                    # route_time = [route_time[0]]
                    self.decision[route[0]] = 1
                else:
                    for i in range(len(route)):
                        route[i] = int(route[i])
                        # route_time[i] = int(route_time[i])
                        self.decision[route[i]] = 1
                self.route = route
                self.Cal_route_time()
            else:
                toggle = True
                while toggle:
                    toggle = False
                    for id_loc_1 in range(N_pck - 1):
                        for id_loc_2 in range(id_loc_1 + 1,
                                              N_pck):  # searching the half of the solution space: efficient when costs are symmetric or scheduling no returning path
                            # route_tp = copy.deepcopy(route)
                            # route_tp[id_loc_1:id_loc_2 + 1] = np.flip(route_tp[id_loc_1:id_loc_2 + 1])
                            # score_tp = self.cal_score(route_tp)
                            # if score_tp > score:  # just for symmetric costs
                            #     toggle = True
                            #     route[id_loc_1:id_loc_2 + 1] = np.flip(route[id_loc_1:id_loc_2 + 1])
                            #     score = self.cal_score(route)
                            if id_loc_1 == 0:
                                value_old_route_1 = value_a2t_cost[route[id_loc_1]]
                                value_new_route_1 = value_a2t_cost[route[id_loc_2]]
                            else:
                                value_old_route_1 = value_t2t_cost[route[id_loc_1 - 1], route[id_loc_1]]
                                value_new_route_1 = value_t2t_cost[route[id_loc_1 - 1], route[id_loc_2]]
                            if id_loc_2 == N_pck - 1:
                                value_old_route_2 = value_t2a_cost[route[id_loc_2]]
                                value_new_route_2 = value_t2a_cost[route[id_loc_1]]
                            else:
                                value_old_route_2 = value_t2t_cost[route[id_loc_2], route[id_loc_2 + 1]]
                                value_new_route_2 = value_t2t_cost[route[id_loc_1], route[id_loc_2 + 1]]
                            if value_old_route_1 * value_old_route_2 < value_new_route_1 * value_new_route_2:  # just for symmetric costs
                                toggle = True
                                route[id_loc_1:id_loc_2 + 1] = np.flip(route[id_loc_1:id_loc_2 + 1])
                                # route_time[id_loc_1:id_loc_2 + 1] = np.flip(route_time[id_loc_1:id_loc_2 + 1])
                self.decision = np.zeros(self.N_task)
                if len(route) == 1:
                    route = [route[0]]
                    # route_time = [route_time[0]]
                    self.decision[route[0]] = 1
                else:
                    for i in range(len(route)):
                        route[i] = int(route[i])
                        # route_time[i] = int(route_time[i])
                        self.decision[route[i]] = 1
                self.route = route
                self.Cal_route_time()

    def prune_prev_route(self, msg_ratio):
        if self.route_length > 0:
            self.prune_all_at_once(msg_ratio)

    def cal_val_remove_element(self, N_pck, route, route_time, mr, id_loc, route_reward):
        value_a2t = self.value_a2t
        value_t2t = self.value_t2t
        value_t2a = self.value_t2a
        if id_loc == 0:
            val_remove = mr * value_a2t[route[0]]
            if self.discount > 0:
                val_remove *= np.exp((route_reward[0] - self.task_reward[route[0]]) / self.max_task_reward)
            if N_pck > 1:
                val_remove *= value_t2t[route[0], route[1]] / value_a2t[route[1]]
                if self.discount > 0:
                    dt = (((self.cost_a2t[route[0]] +
                            self.cost_t2t[route[0], route[1]] -
                            self.cost_a2t[route[1]]) / self.vel * self.base_dist)
                            + self.task_time[id_loc])
                    rr = sum(route_reward[1:N_pck])
                    val_remove *= np.exp(rr * (-1 + np.exp(-self.discount * dt)) / self.max_task_reward)
        elif id_loc == N_pck - 1:
            # val_remove = (mr * value_t2t[route[-2], route[-1]])
            val_remove = (value_t2t[route[-2], route[-1]] *
                          value_t2a[route[-1]] /
                          value_t2a[route[-2]] * mr)
            if self.discount > 0:
                val_remove *= np.exp((route_reward[id_loc] - self.task_reward[route[id_loc]]) /
                                     self.max_task_reward)

        else:
            val_remove = (value_t2t[route[id_loc - 1], route[id_loc]] *
                                  value_t2t[route[id_loc], route[id_loc + 1]] /
                                  value_t2t[route[id_loc - 1], route[id_loc + 1]] * mr)
            if self.discount > 0:
                val_remove *= np.exp((route_reward[id_loc] - self.task_reward[route[id_loc]]) /
                                     self.max_task_reward)
                dt = ((self.cost_t2t[route[id_loc - 1], route[id_loc]] +
                       self.cost_t2t[route[id_loc], route[id_loc + 1]] -
                       self.cost_t2t[route[id_loc - 1], route[id_loc + 1]]) / self.vel * self.base_dist
                       + self.task_time[id_loc])
                rr = sum(route_reward[id_loc + 1:N_pck])
                val_remove *= np.exp(rr * (-1 + np.exp(-self.discount * dt)) / self.max_task_reward)



        return val_remove

    def cal_val_remove(self, N_pck, route, route_time, msg_ratio, route_reward):
        val_remove = np.zeros(N_pck)
        for id_loc in range(N_pck):
            id_t4a = route[id_loc]
            mr = msg_ratio[id_t4a]
            val_remove[id_loc] = self.cal_val_remove_element(N_pck, route, route_time, mr, id_loc, route_reward)
        return val_remove

    def cal_val_insert_element(self, N_pck, route, route_time, mr, id_loc, task, route_reward):
        value_a2t = self.value_a2t
        value_t2t = self.value_t2t
        value_t2a = self.value_t2a
        if id_loc == 0:
            val_insert = mr * value_a2t[task]
            if self.discount > 0:
                t0 = (self.cost_a2t[task] / self.vel * self.base_dist)
                val_insert *= np.exp((self.task_reward[task]*np.exp(-self.discount * t0) - self.task_reward[task])
                                     / self.max_task_reward)
            if N_pck > 0:
                val_insert *= value_t2t[task, route[0]] / value_a2t[route[0]]
                if self.discount > 0:
                    dt = ((self.cost_a2t[task] +
                           self.cost_t2t[task, route[0]] -
                           self.cost_a2t[route[0]]) / self.vel * self.base_dist) + self.task_time[task]
                    rr = sum(route_reward[0:N_pck])
                    val_insert *= np.exp(rr * (-1 + np.exp(-self.discount * dt)) / self.max_task_reward)

        elif id_loc == N_pck:
            # val_insert = (mr * value_t2t[route[-1], task])
            val_insert = (value_t2t[route[- 1], task] *
                          value_t2a[task] /
                          value_t2a[route[- 1]] * mr)
            if self.discount > 0:
                t0 = route_time[-1] + (self.cost_t2t[route[-1], task] / self.vel * self.base_dist)
                val_insert *= np.exp((self.task_reward[task]*np.exp(-self.discount * t0) - self.task_reward[task])
                                     / self.max_task_reward)
        else:
            val_insert = (value_t2t[route[id_loc - 1], task] *
                                  value_t2t[task, route[id_loc]] /
                                  value_t2t[route[id_loc - 1], route[id_loc]] * mr)
            if self.discount > 0:
                t0 = route_time[id_loc - 1] + (self.cost_t2t[route[id_loc - 1], task] / self.vel * self.base_dist)
                val_insert *= np.exp((self.task_reward[task]*np.exp(-self.discount * t0) - self.task_reward[task])
                                     / self.max_task_reward)

                dt = (((self.cost_t2t[route[id_loc - 1], task] +
                       self.cost_t2t[task, route[id_loc]] -
                       self.cost_t2t[route[id_loc - 1], route[id_loc]]) / self.vel * self.base_dist)
                      + self.task_time[task])
                rr = sum(route_reward[id_loc:N_pck])
                val_insert *= np.exp(rr * (-1 + np.exp(-self.discount * dt)) / self.max_task_reward)
        return val_insert

    def cal_val_insert(self, N_pck, route, route_time, msg_ratio, route_reward):
        val_insert = {}
        task_list = np.arange(0, self.N_task)
        if len(route) == 0:
            for task in task_list:
                val_insert[0, int(task)] = self.cal_val_insert_element(N_pck, route, route_time,
                                                                       msg_ratio[task], 0, task,
                                                                       route_reward)
        else:
            rem_id = self.decision == 0
            rem_task = task_list[rem_id]
            rem_msg_ratio = msg_ratio[rem_id]
            for task, mr in zip(rem_task, rem_msg_ratio):
                for id_loc in range(N_pck ):
                    val_insert[id_loc, int(task)] = self.cal_val_insert_element(N_pck, route, route_time,
                                                                                mr, id_loc, task,
                                                                                route_reward)
        return val_insert

    def prune_all_at_once(self, msg_ratio):
        route = self.route
        route_time = self.route_time
        N_pck = self.route_length
        if len(route) != N_pck:
            print("ERROR : route_length is not equal to N_pck")

        route_reward = np.zeros(N_pck)
        for id_loc in range(N_pck):
            route_reward[id_loc] = self.task_reward[route[id_loc]] * np.exp(
                - self.discount * (route_time[id_loc] - self.task_time[route[id_loc]]))

        loss_remove = self.cal_val_remove(N_pck, route, route_time, msg_ratio, route_reward)

        rem_route = []
        for i in range(len(route)):
            if loss_remove[i] <= 1:
                rem_route.append(route[i])

        for r in rem_route:
            route.remove(r)

        self.route = route
        self.Cal_route_time()
        self.route_length = len(route)
        for i in range(self.N_task):
            if i in route:
                self.decision[i] = 1
            else:
                self.decision[i] = 0

    def Cal_route_time(self):
        self.route_time = np.zeros(len(self.route))
        t = 0
        for i in range(len(self.route)):
            if i == 0:
                t += self.cost_a2t[self.route[i]] * self.base_dist / self.vel
            else:
                t += self.cost_t2t[self.route[i - 1], self.route[i]] * self.base_dist / self.vel
            t += self.task_time[self.route[i]]
            self.route_time[i] = t

    def insert_first(self, msg_ratio):
        route = self.route
        route_time = self.route_time
        N_pck = self.route_length

        route_reward = []

        value_insert = self.cal_val_insert(N_pck, route, route_time, msg_ratio, route_reward)
        (loc, task) = max(value_insert, key=value_insert.get)
        task = int(task)
        if value_insert[loc, task] > max(1, msg_ratio[task]):
            self.route = [int(task)]
            self.Cal_route_time()
            self.decision[task] = 1
            self.route_length = 1

    def insert_task(self, msg_ratio):
        task_list = np.arange(0, self.N_task)
        route = self.route
        route_time = self.route_time
        N_pck = self.route_length

        route_reward = np.zeros(N_pck)
        for id_loc in range(N_pck):
            route_reward[id_loc] = self.task_reward[route[id_loc]] * np.exp(
                - self.discount * (route_time[id_loc] - self.task_time[route[id_loc]]))

        value_insert = self.cal_val_insert(N_pck, route, route_time, msg_ratio, route_reward)
        (loc, ta) = max(value_insert, key=value_insert.get)
        if value_insert[loc, ta] > max(1, msg_ratio[ta]) :

            N_pck = N_pck + 1
            # print('a',self.route, value_insert[loc, ta], msg_ratio[ta])
            self.route.insert(loc, int(ta))
            # print('b',self.route)
            self.Cal_route_time()
            self.decision[ta] = 1
            self.route_length = N_pck

    def build_route(self, msg_ratio):
        # build_type = 0 : 그대로, 1 : 계승을 하지 않기, 2 : Pruning 만 하지 않기
        if self.build_type == 0: # 기존 그대로
            self.prune_prev_route(msg_ratio)
        elif self.build_type == 1: #
            self.route = []
            self.route_time = []

        if not self.route:
            self.decision = np.zeros(self.N_task)  # for safety
            self.route_length = 0  # for safety
            self.insert_first(msg_ratio)

        if self.route_length > 0:
            for id_t4a in range(self.route_length, min(self.N_task, self.cap)):
                length_old = self.route_length
                self.insert_task(msg_ratio)
                if length_old < self.route_length:
                    self.two_opt()
                    pass
                else:
                    break

    def compute_msg_a2t(self):
        route = self.route
        route_time = self.route_time
        N_pck = self.route_length

        # value_a2t = self.value_a2t
        # value_t2t = self.value_t2t

        msg_0 = np.ones(self.N_task)  # default is 1
        msg_1 = np.ones(self.N_task)  # default is 1

        route_reward = np.zeros(N_pck)
        for id_loc in range(N_pck):
            route_reward[id_loc] = self.task_reward[route[id_loc]] * np.exp(
                - self.discount * (route_time[id_loc] - self.task_time[route[id_loc]]))

        for id_t4a in range(self.N_task):
            if self.decision[id_t4a] == 1:  # msg_1 = 1
                id_loc = np.where(np.array(route) == id_t4a)[0][0]
                msg_1[id_t4a] = self.cal_val_remove_element(N_pck, route, route_time, 1, id_loc, route_reward)

            else:  # msg_0 = 1
                if N_pck > 0:
                    value_insert = np.zeros(N_pck + 1)
                    for id_loc in range(N_pck + 1):
                        value_insert[id_loc] = self.cal_val_insert_element(N_pck, route, route_time,
                                                                           1, id_loc, id_t4a, route_reward)
                    msg_1[id_t4a] = max(value_insert)
                else:
                    msg_1[id_t4a] = self.value_a2t[id_t4a]
        self.buff_a2t[self.buff_id, :] = msg_1 / (msg_0 + msg_1 + 1e-64)
        self.msg_a2t = np.mean(self.buff_a2t, 0)

    def compute_msg_t2a(self, board):
        id_a = self.id
        for id_t4a in range(self.N_task):
            id_t = self.task[id_t4a]
            if board[id_t].N_agent == 1:
                self.buff_t2a[self.buff_id, id_t4a] = 1
            else:
                msg_a2t = []
                for ii, id_a_other in enumerate(board[id_t].agent):
                    if id_a != id_a_other:
                        msg_a2t.append(board[id_t].msg[0, ii])
                # self.buff_t2a[self.buff_id, id_t4a] = 1 - max(msg_a2t)
                self.buff_t2a[self.buff_id, id_t4a] = 1 - max(msg_a2t)
        self.msg_t2a = np.mean(self.buff_t2a, 0)

    def update_board(self, new_board):

        N_pck = self.route_length
        if N_pck != len(self.route):
            print("ERROR in update board")
        for id_t4a in range(self.N_task):
            id_t = self.task[id_t4a]
            id_a4t = int(self.id_a4t[id_t4a])
            if id_a4t > -1:
                new_board[id_t].msg[0, id_a4t] = self.msg_a2t[id_t4a]
                new_board[id_t].msg[3, id_a4t] = self.msg_a2t[id_t4a]
                new_board[id_t].msg[1, id_a4t] = self.decision[id_t4a]
                if N_pck < self.cap:
                    new_board[id_t].msg[2, id_a4t] = 1
                else:
                    new_board[id_t].msg[2, id_a4t] = 0
                # print(self.id, self.finish_flag, 'aa')
                if self.finish_flag == True:
                    # print(self.id, "aa")
                    new_board[id_t].msg[4, id_a4t] = 1
                else:
                    new_board[id_t].msg[4, id_a4t] = 0
                # if the_agent.route_cnt > 2 and len(the_agent.route_prev) == the_agent.cap:
                #     if not id_t4a in the_agent.route_prev:
                #         new_board[id_t].msg[0, id_a4t] = max(1 - (1 - the_agent.msg_a2t[id_t4a]) * 100, 0.001)
                #         new_board[id_t].msg[3, id_a4t] = max(1 - (1 - the_agent.msg_a2t[id_t4a]) * 100, 0.001)

        return new_board

    def dec_BP_msg(self, board, new_board):
        id_a = self.id
        self.iter_cnt += 1

        if not self.finish_flag:
            self.msg_a2t_sav = np.zeros((board[0].N_agent, self.N_task))
            for j in range(self.N_task):
                for i in range(board[j].N_agent):
                    if i == id_a:
                        self.msg_a2t_sav[i, j] = self.msg_a2t[j]
                    else:
                        self.msg_a2t_sav[i, j] = board[j].msg[0, i]

            self.compute_msg_t2a(board)
            msg_t2a = self.msg_t2a
            msg_ratio = msg_t2a / (1 - msg_t2a + 1e-10)
            # print(self.iter_cnt,msg_t2a)
            self.build_route(msg_ratio)
            # print(self.id, self.route, len(self.route), self.route_length)
            self.compute_msg_a2t()
            new_board = self.update_board(new_board)
            buff_size = len(self.buff_a2t)
            self.buff_id = self.buff_id + 1
            if self.buff_id >= buff_size:
                self.buff_id = 0
        else:
            self.route_length = len(self.route)
            new_board = self.update_board(new_board)
            # print(self.iter_cnt, self.id, self.route, len(self.route), self.route_length)

        # if self.iter_cnt >= self.N_iter - 1:
        #     # print(self.N_iter)
        #     self.refinement(board)
                    
        return new_board

    def take_open_task(self, board, new_board):
        id_a = self.id
        # N_pck = self.route_length
        N_pck = len(self.route)

        if N_pck < self.cap:
            for id_t4a in range(self.N_task):
                id_t = self.task[id_t4a]
                id_a4t = int(self.id_a4t[id_t4a])
                if self.decision[id_t4a] == 0 and np.sum(board[id_t].msg[1, :]) == 0:
                    max_id = np.argmax(board[id_t].msg[0, :] * board[id_t].msg[2, :])
                    if board[id_t].agent[max_id] == id_a and board[id_t].msg[0, max_id] > 0.5:
                        route = self.route
                        route_time = self.route_time

                        if N_pck == 0:
                            value_insert = [self.value_a2t[id_t4a]]
                            max_loc = 0
                            route = [id_t4a]
                        else:
                            route_reward = np.zeros(N_pck)
                            for id_loc in range(N_pck):
                                route_reward[id_loc] = self.task_reward[route[id_loc]] * np.exp(
                                    - self.discount * (route_time[id_loc] - self.task_time[route[id_loc]]))

                            value_insert = np.zeros(N_pck + 1)
                            for id_loc in range(N_pck + 1):
                                value_insert[id_loc] = self.cal_val_insert_element(N_pck, route, route_time,
                                                                                   1, id_loc, id_t4a, route_reward)
                            max_loc = np.argmax(value_insert)
                            route = route[:max_loc] + [id_t4a] + route[max_loc:]

                        if value_insert[max_loc] > 1:
                            N_pck = N_pck + 1
                            self.decision[id_t4a] = 1
                            self.route = route
                            self.route_length = len(self.route)
                            self.Cal_route_time()
                            self.two_opt()

                            new_board[id_t].msg[1, id_a4t] = 1
                            if N_pck == self.cap:
                                new_board[id_t].msg[2, id_a4t] = 0
                                break

        return new_board

    def release_task(self, board, new_board):
        id_a = self.id
        route = self.route
        N_pck = self.route_length
        for id_t4a in range(self.N_task):
            id_t = self.task[id_t4a]
            id_a4t = int(self.id_a4t[id_t4a])
            id_other_a4t = []
            for i, a in enumerate(board[id_t].agent):
                if a != id_a:
                    id_other_a4t.append(True)
                else:
                    id_other_a4t.append(False)
            if self.decision[id_t4a] == 1 and sum(board[id_t].msg[1, id_other_a4t]) >= 1:
                sum_compete_higher = 0
                for i, a in enumerate(board[id_t].agent):
                    if id_other_a4t[i]:
                        if board[id_t].msg[1, i] == 1 and board[id_t].msg[0, i] >= board[id_t].msg[0, id_a4t]:
                            sum_compete_higher += 1
                if sum_compete_higher >= 1:
                    N_pck = N_pck - 1
                    route.remove(id_t4a)
                    self.decision[id_t4a] = 0
                    new_board[id_t].msg[1, id_a4t] = 0
                    if N_pck < self.cap:
                        new_board[id_t].msg[2, id_a4t] = 1
                    else:
                        new_board[id_t].msg[2, id_a4t] = 0
        # print("msg", the_agent.id, new_board[0].msg[2, :])
        self.route = route
        self.Cal_route_time()
        self.route_length = N_pck
        self.two_opt()
        return new_board

    def dec_BP_conf_res(self, board, new_board):
        id_a = self.id
        comp_msg = np.zeros((board[0].N_agent, self.N_task))
        for j in range(self.N_task):
            for i in range(board[j].N_agent):
                if i == id_a:
                    comp_msg[i, j] = abs(self.msg_a2t_sav[i, j] - self.msg_a2t[j])
                else:
                    comp_msg[i, j] = abs(self.msg_a2t_sav[i, j] - board[j].msg[0, i])
        # print(np.max(comp_msg))
        if np.max(comp_msg) < 1e-6 and self.iter_cnt > 1:
            self.finish_flag = True
        else:
            self.finish_flag = False

        if self.iter_cnt < self.N_iter:
            if not self.finish_flag:
                if self.conf_res_type == 0:
                    # print(self.id, self.route)
                    new_board = self.take_open_task(board, new_board)
                    new_board = self.release_task(board, new_board)
                    # print(self.id, self.route)
            else:
                # tp = np.zeros((board[0].N_agent, self.N_task))
                # for j in range(self.N_task):
                #     for i in range(board[j].N_agent):
                #         tp[i, j] = board[j].msg[4, i]
                #
                # if np.min(tp) == 1:
                #     # print(tp)
                #     self.total_fin_cnt += 1
                # else:
                #     self.total_fin_cnt = 0
                #
                # if self.total_fin_cnt > 1:
                #     self.total_finish_flag = True

                if self.refine_type == 0:
                    if self.finish_flag and not self.refine_flag:
                        self.refine_flag = True
                        self.refinement(board)

            # print("AA", self.route)
        elif self.refine_type == 0 and not self.refine_flag:
            self.refine_flag = True
            self.refinement(board)


        # if self.refine_type == 0:
        #     if self.iter_cnt >= self.N_iter and not self.refine_flag:
        #         print(self.iter_cnt, self.N_iter)
        #         self.refine_flag = True
        #         self.refinement(board)


        return new_board

    def refinement(self, board):
        # if self.id == 0:
        #     print(self.iter_cnt)
        N_agent = board[0].N_agent
        N_task = self.N_task
        msg_mat = np.zeros((N_agent, N_task))
        dec_mat = np.zeros((N_agent, N_task))

        # board_msg 0 : msg0 value, 1 : 할당 여부, 2: 할당 위치
        for j in range(N_task):
            if sum(board[j].msg[1, :]) > 1:
                ind = np.argmax(board[j].msg[0, :] * ( board[j].msg[1, :]))
                for k in range(N_agent):
                    if k == ind:
                        dec_mat[ind, j] = 1
            elif sum(board[j].msg[1, :]) == 1:
                dec_mat[board[j].msg[1, :] == 1, j] = 1

        for j in range(N_task):
            if j in self.route and dec_mat[self.id, j] == 0:
                self.route.remove(j)
                self.Cal_route_time()

        for i in range(N_agent):
            for j in range(N_task):
                if sum(dec_mat[:, j]) == 0:
                    msg_mat[i, j] = board[j].msg[0, i]
                else:
                    msg_mat[i, j] = 0

        # print(msg_mat)
        while True:
            ind = np.unravel_index(np.argmax(msg_mat, axis=None), msg_mat.shape)
            # print(msg_mat[ind[0], ind[1]])
            if msg_mat[ind[0], ind[1]] > 0.5:
                if sum(dec_mat[ind[0], :]) < self.cap:
                    dec_mat[ind[0], ind[1]] = 1
                    msg_mat[:, ind[1]] = 0
                    if ind[0] == self.id:
                        # print(ind[1])
                        self.route.append(ind[1])
                        self.Cal_route_time()
                        self.route_cnt += 1
                        self.route_length = len(self.route)
                        # print(self.route)
                        self.two_opt()
                        # print(self.route)
                else:
                    msg_mat[ind[0], ind[1]] = 0
            else:
                break

class Board:
    def __init__(self, id_t, agent):
        self.agent = []
        for ag in agent:
            if id_t in ag.task:
                self.agent.append(ag.id)
        self.N_agent = len(self.agent)
        self.msg = np.zeros((5, self.N_agent))
        self.msg[0, :] = np.ones(self.N_agent) / 2
        self.msg[2, :] = np.ones(self.N_agent)
        self.msg[3, :] = np.ones(self.N_agent) / 2

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
        self.msg_t2a = np.zeros((setting.N_iter, setting.N_agent, setting.N_task))

        self.record_time_arr = np.zeros(setting.N_task)

        self.total_t = 0
        self.total_r = 0
        self.total_n = 0

class AMP_CR:
    def __init__(self, prob, build_type=0, conf_res_type=0, refine_type=0, buff_size=2, discount=-1, N_iter=30, return_type=False):
        self.prob = prob
        self.return_type = return_type
        # build_type = 0 : 그대로, 1 : 계승을 하지 않기, 2 : Pruning 만 하지 않기
        # conf_res_type = 0 : 그대로, 1 : 안하기 --> 이 때 build 는 1이 되어야 함.
        # refine_type = 0 : 그대로, 1 : 안하기
        self.build_type = build_type
        self.conf_res_type = conf_res_type
        self.refine_type = refine_type
        self.buff_size = buff_size
        self.discount = discount
        self.N_iter = N_iter

    def solve(self):
        setting = Setting(self.prob, self.buff_size, self.N_iter)
        if self.discount > -1:
            setting.discount = self.discount

        self.task_reward = setting.task_reward
        self.agent_position = setting.agent_position
        self.task_position = setting.task_position
        self.task_time = setting.task_time

        agent = []
        for i in range(setting.N_agent):
            agent.append(Agent(i, setting, self.build_type, self.conf_res_type, self.refine_type, self.return_type))
        board = []
        for i in range(setting.N_task):
            board.append(Board(i, agent))

        for id_a in range(setting.N_agent):
            agent[id_a].id_a4t = np.zeros(agent[id_a].N_task)
            for id_t4a in range(agent[id_a].N_task):
                id_t = agent[id_a].task[id_t4a]
                if id_a in board[id_t].agent:
                    agent[id_a].id_a4t[id_t4a] = board[id_t].agent.index(id_a)
                else:
                    agent[id_a].id_a4t[id_t4a] = -1

        # run BP
        result = Result(setting)
        new_board = copy.deepcopy(board)
        for id_iter in range(setting.N_iter):
            # print("id_iter", id_iter)
            # message calculation
            for id_a in range(setting.N_agent):
                new_board = agent[id_a].dec_BP_msg(board, new_board)

                for id_t in range(setting.N_task):
                    result.msg_t2a[id_iter, id_a, id_t] = agent[id_a].msg_t2a[id_t]
                    if id_iter > 0:
                        if abs(result.msg_t2a[id_iter - 1, id_a, id_t] - result.msg_t2a[id_iter, id_a, id_t]) < 1e-6:
                            result.cvg_msg_rate[id_iter] += 1

            board = copy.deepcopy(new_board) #write( and read) the board
            # conflict resolution
            for id_a in range(setting.N_agent):
                new_board = agent[id_a].dec_BP_conf_res(board, new_board)

        total_t = 0
        total_r = 0
        total_n = 0
        for i, a in enumerate(agent):
            result.route[i] = a.route
            # print(a.route)
            r = 0
            t = 0
            nn = 0
            # print(i, a.route_time)
            for j in range(len(a.route)):
                nn += 1
                if self.discount > 0:
                    r += self.task_reward[a.route[j]] * np.exp(
                        -self.discount * (a.route_time[j] - self.task_time[a.route[j]]))
                    # print(a.route[j], a.route_time[j], self.task_reward[a.route[j]] * np.exp(-self.discount * a.route_time[j]))
                    # r += self.task_reward[a.route[j]]
                    # print(a.route[j], self.task_reward[a.route[j]] * np.exp(-self.discount * a.route_time[j]))
                else:
                    r += self.task_reward[a.route[j]]
                if j == 0:
                    t += LA.norm(self.task_position[a.route[j], :] - self.agent_position[i, :]) / a.vel
                else:
                    t += LA.norm(self.task_position[a.route[j], :] - self.task_position[a.route[j - 1], :]) / a.vel
                result.record_time_arr[a.route[j]] = t
                t += self.task_time[a.route[j]]

            total_t += t
            total_r += r
            total_n += nn
        result.total_t = total_t
        result.total_r = total_r
        result.total_n = total_n
        return result



