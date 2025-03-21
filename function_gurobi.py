import gurobipy as gp
from gurobipy import Model, GRB, quicksum
import numpy as np
import numpy.linalg as LA

class Setting:
    def __init__(self, prob, buff_size=2, N_iter=30):
        self.N_agent = prob.N_agent
        self.N_task = prob.N_task
        self.capacity = prob.capacity # 하나의 값
        self.max_depth = self.capacity

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

        self.Error = False

class Gurobi_solve:
    def __init__(self, prob):
        self.prob = prob

    def solve_sav(self):
        self.N_agent = self.prob.N_agent
        self.N_task = self.prob.N_task
        self.agent_position = self.prob.agent_position
        self.task_position = self.prob.task_position
        self.task_time = self.prob.task_time
        self.base_dist = self.prob.base_dist
        self.agent_vel = self.prob.agent_vel  # vector size : N_agent

        self.dist_a2t = np.zeros((self.N_agent, self.N_task))
        for i in range(self.N_agent):
            for j in range(self.N_task):
                self.dist_a2t[i, j] = np.sqrt((self.task_position[j, 0] - self.agent_position[i, 0]) ** 2
                                              + (self.task_position[j, 1] - self.agent_position[i, 1]) ** 2)

        self.R = np.ones(self.N_task) * 10
        self.dist_t2t = np.zeros((self.N_task, self.N_task))
        for i in range(self.N_task):
            for j in range(self.N_task):
                if i == j:
                    self.dist_t2t[i, j] = 1e9 * self.base_dist
                else:
                    self.dist_t2t[i, j] = np.sqrt((self.task_position[i, 0] - self.task_position[j, 0]) ** 2
                                              + (self.task_position[i, 1] - self.task_position[j, 1]) ** 2)


        self.d = np.zeros((self.N_agent + self.N_task, self.N_agent + self.N_task))
        for i in range(self.N_agent + self.N_task):
            for j in range(self.N_agent + self.N_task):
                if i < self.N_agent and j >= self.N_agent:
                    self.d[i, j] = self.dist_a2t[i, j - self.N_agent] / self.base_dist
                elif i >= self.N_agent and j < self.N_agent:
                    self.d[i, j] = 0
                elif i < self.N_agent and j < self.N_agent:
                    self.d[i, j] = 1e9
                else:
                    self.d[i, j] = self.dist_t2t[i - self.N_agent, j - self.N_agent] / self.base_dist


        self.T = list(range(self.N_agent, self.N_agent + self.N_task))
        self.A = list(range(self.N_agent))
        self.U = self.A + self.T

        mdl = gp.Model("VRP")
        Pairs = [(i, j, k) for i in self.A for j in self.U for k in self.U]
        x = mdl.addVars(Pairs, vtype=GRB.BINARY, name="x")
        mdl.modelSense = GRB.MINIMIZE

        mdl.setObjective(quicksum(self.d[j, k] / self.agent_vel[i] * x[i, j, k] for i, j, k in Pairs), GRB.MINIMIZE)
        u = mdl.addVars(self.T, vtype=GRB.CONTINUOUS, name="u")
        # Constraint
        mdl.addConstrs((quicksum(x[i, j, k] for k in self.T for j in self.A if not i == j) == 0) for i in self.A )
        mdl.addConstrs((quicksum(x[i, i, k] for k in self.T) <= 1) for i in self.A) # i, i 같을 때만 의미 있음 j in A 일때는
        mdl.addConstrs((quicksum(x[i, k, i] for k in self.T) <= 1) for i in self.A)  # i, i 같을 때만 의미 있음 j in A 일때는
        mdl.addConstrs((quicksum(x[i, j, k] for j in self.U for k in self.T) <= self.prob.capacity) for i in self.A) # cap
        mdl.addConstrs((quicksum(x[i, j, k] for i in self.A for j in self.U) == 1) for k in self.U) # 모든 task 는 할당된다.
        # mdl.addConstrs((quicksum(x[i, k, k] for i in self.A) == 0) for k in self.T)
        # mdl.addConstrs((quicksum(x[i, j, k] for i in self.A) == quicksum(x[i, k, j] for i in self.A)) for j in self.T for k in self.T)
        mdl.addConstrs(
            (quicksum(x[i, j, k] for j in self.U) - quicksum(x[i, k, j] for j in self.U)) == 0 for i in self.A for k in self.U)
        mdl.addConstrs(
            (u[j] - u[k] + len(self.T) * x[i, j, k] <= len(self.T) - 1) for i in self.A for j in self.T for k in self.T)

        # Solve
        mdl.Params.MIPGap = 0.00001  # Optimal 과 tolance
        mdl.Params.TimeLimit = 300  # seconds
        mdl.Params.LogToConsole = 0
        mdl.optimize()

        self.path_list = [[] for i in range(self.N_agent)]
        for i in range(self.N_agent):
            tp = [(j, k) for i1, j, k in x.keys() if x[i1, j, k].X > 0.5 if i1 == i]
            if tp:
                tp2 = [i for i, j in tp]
                ind = tp[tp2.index(i)][0]
                for k in range(len(tp2) - 1):
                    ind = tp[tp2.index(ind)][1]
                    self.path_list[i].append(ind - self.N_agent)

        setting = Setting(self.prob)
        result = Result(setting)

        total_t = 0
        total_r = 0
        for i, p in enumerate(self.path_list):
            result.route[i] = p
            r = 0
            t = 0
            for j in range(len(p)):
                r += 1
                if j == 0:
                    t += LA.norm(self.task_position[p[j], :] - self.agent_position[i, :]) / self.agent_vel[i]
                else:
                    t += LA.norm(self.task_position[p[j], :] - self.task_position[p[j - 1], :]) / self.agent_vel[i]
                result.record_time_arr[p[j]] = t
                t += self.task_time[p[j]]

            total_t += t
            total_r += r
        # print(total_t, total_r)
        result.total_t = total_t
        result.total_r = total_r
        return result

    def solve(self):
        self.N_agent = self.prob.N_agent
        self.N_task = self.prob.N_task
        self.agent_position = self.prob.agent_position
        self.task_position = self.prob.task_position
        self.task_time = self.prob.task_time
        self.base_dist = self.prob.base_dist
        self.agent_vel = self.prob.agent_vel  # vector size : N_agent

        self.dist_a2t = np.zeros((self.N_agent, self.N_task))
        for i in range(self.N_agent):
            for j in range(self.N_task):
                self.dist_a2t[i, j] = np.sqrt((self.task_position[j, 0] - self.agent_position[i, 0]) ** 2
                                              + (self.task_position[j, 1] - self.agent_position[i, 1]) ** 2)

        self.dist_t2t = np.zeros((self.N_task, self.N_task))
        for i in range(self.N_task):
            for j in range(self.N_task):
                if i == j:
                    self.dist_t2t[i, j] = 0 * self.base_dist
                else:
                    self.dist_t2t[i, j] = np.sqrt((self.task_position[i, 0] - self.task_position[j, 0]) ** 2
                                              + (self.task_position[i, 1] - self.task_position[j, 1]) ** 2)

        self.R = np.ones(self.N_agent + self.N_task)
        for i in range(self.N_agent + self.N_task):
            if i < self.N_agent:
                self.R[i] = 0

        # self.d = np.zeros((self.N_agent + self.N_task, self.N_agent + self.N_task))
        #
        # for j in range(self.N_agent + self.N_task):
        #     if i < self.N_agent and j >= self.N_agent:
        #         self.d[i, j] = self.dist_a2t[i, j - self.N_agent] / self.base_dist
        #     elif i >= self.N_agent and j < self.N_agent:
        #         self.d[i, j] = 0
        #     elif i < self.N_agent and j < self.N_agent:
        #         self.d[i, j] = 1e9
        #     else:
        #         self.d[i, j] = self.dist_t2t[i - self.N_agent, j - self.N_agent] / self.base_dist

        self.c = np.zeros((self.N_agent, self.N_agent + self.N_task, self.N_agent + self.N_task))
        for i in range(self.N_agent):
            for j in range(self.N_agent + self.N_task):
                for k in range(self.N_agent + self.N_task):
                    if j < self.N_agent and k < self.N_agent:
                        if i == j and k == j:
                            self.c[i, j, k] = 0
                        else:
                            self.c[i, j, k] = 1e9
                    elif j < self.N_agent and k >= self.N_agent:
                        if i == j:
                            self.c[i, j, k] = (self.dist_a2t[j, k - self.N_agent]  / self.agent_vel[i]
                                               + self.task_time[k - self.N_agent]) / self.base_dist
                        else:
                            self.c[i, j, k] = 1e9
                    elif j >= self.N_agent and k < self.N_agent:
                        if i == k:
                            self.c[i, j, k] = 0
                        else:
                            self.c[i, j, k] = 1e9
                    else:
                        self.c[i, j, k] = (self.dist_t2t[j - self.N_agent, k - self.N_agent] / self.agent_vel[i]
                                           + self.task_time[k - self.N_agent]) / self.base_dist


        self.T = list(range(self.N_agent, self.N_agent + self.N_task))
        self.A = list(range(self.N_agent))
        self.U = self.A + self.T

        mdl = gp.Model("VRP")
        Pairs = [(i, j, k) for i in self.A for j in self.U for k in self.U]
        x = mdl.addVars(Pairs, vtype=GRB.BINARY, name="x")
        mdl.modelSense = GRB.MINIMIZE

        mdl.setObjective(quicksum((-self.R[i] + self.c[i, j, k]) * x[i, j, k] for i, j, k in Pairs), GRB.MINIMIZE)
        u = mdl.addVars(self.T, vtype=GRB.CONTINUOUS, name="u")
        # Constraint
        # mdl.addConstrs((quicksum(x[i, j, k] for k in self.T for j in self.A if not i == j) == 0) for i in self.A)
        mdl.addConstrs((quicksum(x[i, j, k] for j in self.U for k in self.T) <= self.prob.capacity) for i in self.A) # cap
        mdl.addConstrs((quicksum(x[i, j, k] for i in self.A for j in self.U) == 1) for k in self.U) # 모든 task 는 할당된다.
        mdl.addConstrs(
            (quicksum(x[i, j, k] for j in self.U) - quicksum(x[i, k, j] for j in self.U)) == 0 for i in self.A for k in self.U)
        mdl.addConstrs(
            (u[j] - u[k] + len(self.T) * x[i, j, k] <= len(self.T) - 1) for i in self.A for j in self.T for k in self.T)

        # Solve
        mdl.Params.MIPGap = 0.00001  # Optimal 과 tolance
        mdl.Params.TimeLimit = 300  # seconds
        mdl.Params.LogToConsole = 0
        mdl.optimize()

        try:
            self.path_list = [[] for i in range(self.N_agent)]
            for i in range(self.N_agent):
                tp = [(j, k) for i1, j, k in x.keys() if x[i1, j, k].X > 0.5 if i1 == i]
                # print(tp)
                if tp:
                    tp2 = [i for i, j in tp]
                    ind = tp[tp2.index(i)][0]
                    for k in range(len(tp2) - 1):
                        ind = tp[tp2.index(ind)][1]
                        self.path_list[i].append(ind - self.N_agent)

            setting = Setting(self.prob)
            result = Result(setting)

            total_t = 0
            total_r = 0
            for i, p in enumerate(self.path_list):
                result.route[i] = p
                r = 0
                t = 0
                for j in range(len(p)):
                    r += self.prob.task_reward[p[j]]
                    if j == 0:
                        t += LA.norm(self.task_position[p[j], :] - self.agent_position[i, :]) / self.agent_vel[i]
                    else:
                        t += LA.norm(self.task_position[p[j], :] - self.task_position[p[j - 1], :]) / self.agent_vel[i]
                    result.record_time_arr[p[j]] = t
                    t += self.task_time[p[j]]

                total_t += t
                total_r += r
            # print(total_t, total_r)
            result.total_t = total_t
            result.total_r = total_r
            return result

        except:
            setting = Setting(self.prob)
            result = Result(setting)
            result.total_t = 0
            result.total_r = 0
            result.Error = True
            return result

# def function_gurobi(p):
#     self.C = list(range(self.depot_n, self.customer_n + self.depot_n))
#     self.D = list(range(self.depot_n))
#     self.U = self.D + self.C
#     num_depot = p.depot_n
#     positions = p.positions
#
#     mdl = gp.Model("VRP")
#     Pair = [(i, j) for i in U for j in U if i != j]
#     pairs = [(i, j, k, l) for i in U for j in U for k, l in K]
#
#     x = mdl.addVars(pairs, vtype=GRB.BINARY, name="x")
#     mdl.modelSense = GRB.MINIMIZE
#     u = mdl.addVars(C, vtype=GRB.CONTINUOUS, name="u")
#
#     # Objective Function
#     mdl.setObjective(quicksum(d[i, j] / p.Vel[k, l] * x[i, j, k, l] for i, j in Pair for k, l in K), GRB.MINIMIZE)
#
#     # Constraint
#     mdl.addConstr(quicksum(x[i, i, k, l] for i in U for k, l in K) == 0)
#     mdl.addConstrs((quicksum(x[i, j, k, l] for i in U) - quicksum(x[j, i, k, l] for i in U)) == 0 for j in U for k, l in K)
#     mdl.addConstrs(quicksum(x[i, j, k, l] for k, l in K for i in U if i != j) == 1 for j in C)
#     mdl.addConstrs(quicksum(x[k, j, k, l] for j in C) <= 1 for k, l in K)
#     mdl.addConstrs(quicksum(x[i, j, k, l] for j in C for i in D if not i == k) == 0 for k, l in K)
#     mdl.addConstrs(quicksum(x[i, j, k, l] * q[j] for i in U for j in C if j != i) <= Q[k, l] for k, l in K)
#     mdl.addConstrs((u[i] - u[j] + len(C) * x[i, j, k, l] <= len(C) - 1) for i in C for j in C if i != j for k, l in K)
#
#     # Solve
#     mdl.Params.MIPGap = 0.00001  # Optimal 과 tolance
#     mdl.Params.TimeLimit = 60  # seconds
#     mdl.Params.LogToConsole = 0
#     mdl.optimize()
#
#     # 각 차량의 경로를 저장할 딕셔너리
#     routes = {}
#     caps = {}
#     vehs = {}
#     route_len = np.zeros(p.depot_n)
#     for dep, num_v in enumerate(p.vehicle_n):
#         cap = [[]]
#         route = [[]]
#         veh = []
#         cnt = 0
#         for v in range(num_v):
#             tp = [(i, j) for i, j, k, l in x.keys() if k == dep and v == l and x[i, j, k, l].X > 0.5]
#             if tp:
#                 tp2 = [i for i, j in tp]
#                 ind = tp[0][0]
#                 for j in range(len(tp2)):
#                     route_len[dep] += euclidean_distance(positions[tp[j][0]], positions[tp[j][1]]) / p.Vel[dep, v]
#                     if ind in tp2:
#                         ind = tp[tp2.index(ind)][1]
#                         if ind < num_depot:
#                             route.append([])
#                             cap.append([])
#                             cnt += 1
#                         else:
#                             route[cnt].append(ind - num_depot)
#                             cap[cnt].append(p.q[ind])
#                 veh.append((v, p.Depots[dep].Vehicles[v].type_))
#         if not route[-1]:
#             route.remove(route[-1])
#             cap.remove(cap[-1])
#         routes[dep] = route
#         caps[dep] = cap
#         vehs[dep] = veh
#
#     result = Results(routes, caps, route_len, vehs, "GRB")
#     return result