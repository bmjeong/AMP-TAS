import numpy as np

# print(np.random.randint(0, 1, 10))

class Prob:
    def __init__(self, N_agent, N_task, seed, R, map=2000, var = 0, discount = 0.1):
        np.random.seed(seed)
        self.discount = discount
        self.N_agent = N_agent
        self.N_task = N_task
        self.capacity = int(np.ceil(self.N_task / self.N_agent * 1.3))
        # self.capacity = self.N_task
        # print("cap", self.capacity)
        # Agent Type, Task Type 관련된 것
        self.N_agent_type = 1
        self.N_task_type = 1
        self.feasibility = np.eye(1)

        self.agent_type = np.random.randint(0, self.N_agent_type, self.N_agent)
        # if R > 1:
        #     self.task_reward = np.random.randint(1, R, self.N_task)  # task reward: 1 ~ 2
        # else:
        #     self.task_reward = R * np.ones(self.N_task)


        self.task_reward = R * np.ones(self.N_task) + var * np.random.rand(self.N_task) - var/2
        # if R > 1:
        #     min_r = 1
        # else:
        #     min_r = 0.1
        # for i in range(len(self.task_reward)):
        #     if self.task_reward[i] < min_r:
        #         self.task_reward[i] = min_r
        self.task_type = np.random.randint(0, self.N_task_type, self.N_task)
        self.map_size = np.asarray([[0, map], [0, map]])
        self.agent_position = self.map_size[:, 0] + np.random.random((self.N_agent, 2)) * self.map_size[:, 1]
        self.task_position = self.map_size[:, 0] + np.random.random((self.N_task, 2)) * self.map_size[:, 1]

        # self.base_dist = np.sqrt((self.map_size[0, 1] - self.map_size[0, 0]) ** 2 +
        #                          (self.map_size[1, 1] - self.map_size[1, 0]) ** 2) / np.sqrt(self.N_task/self.N_agent)
        self.base_dist =500

        self.agent_vel = 1 * np.ones(self.N_agent)  # agent 의 속도 일괄 1
        self.task_time = 2 * np.ones(self.N_task) + 8 * np.random.rand(self.N_task)