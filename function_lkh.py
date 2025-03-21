import lkh
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

class LKH_solve:
    def __init__(self, prob):
        self.prob = prob

    def gen_prb_str(self, num_customer, num_depot, d, D, C, Vel, num):
        problem_str = "NAME : mVRP\n"
        problem_str += "TYPE : ATSP\n"
        problem_str += "DIMENSION : " + str((num_customer + 2) * num_depot) + "\n"
        problem_str += "EDGE_WEIGHT_TYPE : EXPLICIT\n"
        problem_str += "EDGE_WEIGHT_FORMAT : FULL_MATRIX\n"
        problem_str += "EDGE_WEIGHT_SECTION\n"
        dist_mat = np.zeros(((num_customer + 2) * num_depot, (num_customer + 2) * num_depot))

        M = 10000
        for i in range(num_depot):
            for j in range(num_depot):
                ## Agent / Task 간 연결 부분은 일단 다 M 으로 채우는 것
                for k in range(num_customer + 2):
                    for l in range(num_customer + 2):
                        if k >= num_customer or l >= num_customer:
                            dist_mat[i + k * num_depot, j + l * num_depot] = M * M
                            dist_mat[i + l * num_depot, j + k * num_depot] = M * M

                if i == j:
                    ## agent_start to Task 로 가는 edge agent_i 에서 Task_j_i 로 가는 것만 cost 가 있고 나머지는 M
                    for k in range(num_customer):
                        dist_mat[i + num_customer * num_depot, j + k * num_depot] = d[D[i], C[k]] / Vel[j] + M
                    ## agent_start 에서 agent_finish 로 가능 건 본인 node 만 가능
                    dist_mat[i + num_customer * num_depot, j + (num_customer + 1) * num_depot] = 0
                else:
                    ## agent_finish 에서 agent_start 로 가능 건 본인 node 아닌쪽만 가능
                    dist_mat[i + (num_customer + 1) * num_depot, j + num_customer * num_depot] = 0

                ## agent 가 2개 뿐일때는 diagonal 은 0 이고, 아닌쪽은 2--> 1 1-->2 로 항상 연결되므로 M으로 처리할 부분이 없음
                ## 아래는 동일 Task 간 edge 연결에 대한 것
                ## agent 가 많으면 1 --> 2, 2 --> 3 3--> 1 로 가는거만 허용 반대로 연결되는 건 불허
                if num_depot > 2:
                    if i + 1 == j or j + num_depot == i + 1:
                        pass
                    else:
                        for k in range(num_customer):
                            dist_mat[i + k * num_depot, j + k * num_depot] = M * M

                if i + 1 == j or i + 1 == j + num_depot:
                    # 서로 다른 태스크 간 연결 시에
                    for k in range(num_customer - 1):
                        for l in range(num_customer - k - 1):
                            dist_mat[i + k * num_depot, j + (l + k + 1) * num_depot] = d[C[k], C[l + k + 1]] / Vel[
                                j] + M
                            dist_mat[i + (l + k + 1) * num_depot, j + k * num_depot] = d[C[k], C[l + k + 1]] / Vel[
                                j] + M
                    # customer 에서 depot_finish 쪽으로 연결되는 것
                    for k in range(num_customer):
                        # dist_mat[i + k * num_depot, j + (num_customer + 1) * num_depot] = d[j, C[k]] / p.Vel[j, 0]
                        ## 복귀하는 것을 고려하지 않는다는 뜻
                        dist_mat[i + k * num_depot, j + (num_customer + 1) * num_depot] = 0

                else:
                    for k in range(num_customer - 1):
                        for l in range(num_customer - k - 1):
                            dist_mat[i + k * num_depot, j + (l + k + 1) * num_depot] = M * M
                            dist_mat[i + (l + k + 1) * num_depot, j + k * num_depot] = M * M

        for i in range((num_customer + 2) * num_depot):
            for j in range((num_customer + 2) * num_depot):
                if dist_mat[i, j] < M * M:
                    problem_str += '{:.5e}'.format(dist_mat[i, j] * 100) + " "
                else:
                    problem_str += str(num) + " "

            problem_str += "\n"

        return problem_str

    def solve(self):
        self.N_agent = self.prob.N_agent
        self.N_task = self.prob.N_task
        self.agent_position = self.prob.agent_position
        self.task_position = self.prob.task_position
        self.task_time = self.prob.task_time
        self.base_dist = self.prob.base_dist
        self.agent_vel = self.prob.agent_vel  # vector size : N_agent
        self.T = list(range(self.N_agent, self.N_agent + self.N_task))
        self.A = list(range(self.N_agent))
        self.U = self.A + self.T

        self.positions = {}
        for i in range(self.N_agent):
            self.positions[i] = self.agent_position[i] * 100  ## 꼭 np.array 여야 함

        for i in range(self.N_agent, self.N_task + self.N_agent):
            self.positions[i] = self.task_position[i - self.N_agent] * 100

        # 거리 정보
        self.d = {(i, j): LA.norm(self.positions[i] - self.positions[j]) for i in self.U for j in self.U if i != j}

        self.Vel = {(i): self.agent_vel[i] for i in self.A}

        customer_demand = []
        for i in range(self.N_task):
            customer_demand.append(1)

        solver_path = 'LKH-3.0.6/LKH'  ### 위치가 바뀌면 여기를 수정해야 함. 컴퓨터 별로 다를 수 있음

        if self.N_task < 6:
            run_num = 10
            trial_num = 1000
        elif self.N_task < 11:
            run_num = 20
            trial_num = 8000
        elif self.N_task < 21:
            run_num = 30
            trial_num = 10000
        elif self.N_task < 31:
            run_num = 70
            trial_num = 20000
        elif self.N_task < 41:
            run_num = 120
            trial_num = 35000
        else:
            run_num = 150
            trial_num = 40000

        try:
            problem_str = self.gen_prb_str(self.N_task, self.N_agent, self.d, self.A, self.T, self.Vel, 1.8e7)
            problem = lkh.LKHProblem.parse(problem_str)
            out = lkh.solve(solver_path, problem=problem, max_trials=trial_num, runs=run_num)
        except:
            try:
                print("2nd try")
                problem_str = self.gen_prb_str(self.N_task, self.N_agent, self.d, self.A, self.T, self.Vel, 1.6e7)
                problem = lkh.LKHProblem.parse(problem_str)
                out = lkh.solve(solver_path, problem=problem, max_trials=10000, runs=50)
            except:
                print("3rd try")
                problem_str = self.gen_prb_str(self.N_task, self.N_agent, self.d, self.A, self.T, self.Vel, 1.5e7)
                problem = lkh.LKHProblem.parse(problem_str)
                out = lkh.solve(solver_path, problem=problem, max_trials=10000, runs=50)
        # print(out)
        ind_set = []
        for i in range(self.N_agent):
            ind = self.N_task * self.N_agent + i + 1
            ind_set.append([out[0].index(ind), out[0].index(ind + self.N_agent)])
        path = []
        for i in range(self.N_agent):
            if ind_set[i][0] < ind_set[i][1]:
                path.append(out[0][ind_set[i][0]:ind_set[i][1] + 1])
            else:
                tp = out[0][ind_set[i][0]:]
                for j in range(ind_set[i][1] + 1):
                    tp.append(out[0][j])
                path.append(tp)

        routes = {}
        cnt = 0
        for p_i in path:
            path_ = []
            for t in p_i:
                if t <= (self.N_agent * self.N_task):
                    if t % self.N_agent == 0:
                        path_.append(int(t / self.N_agent) - 1)
            if path_:
                routes[cnt] = [path_]
            else:
                routes[cnt] = []
            # print(cnt, routes[cnt])
            cnt += 1


        setting = Setting(self.prob)
        result = Result(setting)

        total_t = 0
        total_r = 0
        for i in range(len(routes)):
            if routes[i]:
                result.route[i] = routes[i][0]
            else:
                result.route[i] = []
            r = 0
            t = 0
            for j in range(len(result.route[i])):
                r += self.prob.task_reward[routes[i][0][j]]
                if j == 0:
                    t += LA.norm(self.task_position[routes[i][0][j], :] - self.agent_position[i, :]) / self.agent_vel[i]
                else:
                    t += LA.norm(self.task_position[routes[i][0][j], :] - self.task_position[routes[i][0][j - 1], :]) / self.agent_vel[i]
                result.record_time_arr[routes[i][0][j]] = t
                t += self.task_time[routes[i][0][j]]

            total_t += t
            total_r += r
        # print(total_t, total_r)
        result.total_t = total_t
        result.total_r = total_r
        return result