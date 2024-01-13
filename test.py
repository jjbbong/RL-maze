import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import math
import time


# 定义agent
class QLearningAgent:
    def __init__(self, alpha, gamma, epsilon, actions, start):
        self.start = start  # 起始位置
        self.alpha = alpha  # 学习率
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon
        self.actions = actions  # 0:上, 1:下, 2:左, 3:右
        self.q_table = dict()  # Q表

    def choose_action(self, state):
        # Epsilon-greedy 选择行动
        if np.random.uniform(0, 1) < self.epsilon:
            action = np.random.choice(self.actions)
        else:
            q_values = self.q_table.get(state, np.zeros(len(self.actions)))
            max_q = np.max(q_values)
            # 会有最大值对应多个行动的情况
            actions_with_max_q = [action for action, q in enumerate(q_values) if q == max_q]
            action = np.random.choice(actions_with_max_q)
        return action

    def learn(self, state, action, reward, next_state):
        # 更新Q表
        q_values = self.q_table.get(state, np.zeros(len(self.actions)))
        q_values_next = self.q_table.get(next_state, np.zeros(len(self.actions)))
        values_action = q_values[action] + self.alpha * (reward + self.gamma * np.max(q_values_next) - q_values[action])
        q_values[action] = -np.inf if np.isnan(
            values_action) else values_action  # 由于reward的设置，会出现np.inf-np.inf的情况，因此要判断是否为np.nan
        self.q_table[state] = q_values

    def get_move(self):
        return pd.DataFrame(pd.DataFrame.from_dict(self.q_table, orient='index').idxmax(axis=1))


# 定义迷宫环境
class MazeEnvironment:
    def __init__(self, maze, end, start, maze_path, block_size):
        self.start = start
        self.maze_path = maze_path
        self.maze = maze
        self.block_size = block_size
        self.shape = maze.shape
        self.end = end
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上,下,左,右

    def is_out_boundary(self, state):
        # 检查是否超过边界
        return state[0] * (state[0] - self.shape[0] + 1) > 0 or state[1] * (state[1] - self.shape[1] + 1) > 0

    def is_wall(self, state):
        # 检查是否撞墙
        return self.maze[state[0]][state[1]] == 1

    def get_reward(self, next_state):
        # 回报函数
        if next_state == self.end:
            return 1000  # 到达终点时有高回报
        elif self.is_out_boundary(next_state):
            return -np.inf  # 超过边界的惩罚
        elif self.is_wall(next_state):
            return -np.inf  # 撞墙的惩罚
        else:
            return -1  # 每走一步的惩罚，来得到最短路径

    def step(self, state, action):
        # 行动的反馈
        action = self.actions[action]

        next_state = (state[0] + action[0], state[1] + action[1])

        reward = self.get_reward(next_state)

        if self.is_out_boundary(next_state):  # 如果超出边界或者撞墙，agent不动
            next_state = state
        elif self.is_wall(next_state):
            next_state = state

        return next_state, reward

    def get_path(self, move):
        path = []
        path.append(self.start)
        state = self.start
        while True:
            action = move.loc[[state]][0][0]
            action = self.actions[action]
            state = (state[0] + action[0], state[1] + action[1])
            path.append(state)
            if state == self.end:
                break
        return path

    def draw_path_on_maze(self, move):
        path = self.get_path(move)
        maze_img = cv2.imread(self.maze_path)

        # 将起始点与终点坐标转换成像素坐标
        start_px = (self.start[1] * self.block_size + 2, self.start[0] * self.block_size)
        end_px = (self.end[1] * self.block_size + 2, self.end[0] * self.block_size)

        # 将path的坐标转换为像素坐标
        path_px = [(x[1] * self.block_size, x[0] * self.block_size) for x in path]

        # 画出起始点与终点
        cv2.circle(maze_img, start_px, 5, (0, 255, 0), -1)  # 绿色代表起始点
        cv2.circle(maze_img, end_px, 5, (255, 0, 0), -1)  # 蓝色代表终点

        # 画路径
        for i in range(len(path_px) - 1):
            cv2.line(maze_img, path_px[i], path_px[i + 1], (0, 0, 255), 5)  # 红色代表路径

        # Show the image with the path
        plt.imshow(cv2.cvtColor(maze_img, cv2.COLOR_BGR2RGB))
        plt.title('Path in Maze')
        plt.axis('off')
        plt.show()


def get_end(maze_path):
    clicks = []

    def click_event(event, x, y, flags, param):
        # 如果发生了左键点击
        if event == cv2.EVENT_LBUTTONDOWN:
            clicks.append((x, y))
            cv2.circle(maze, (x, y), 3, (255, 0, 0), -1)  # 在点击位置画一个小圆圈
            cv2.destroyAllWindows()
            cv2.imshow("maze", maze)
            cv2.waitKey(1000)
            cv2.destroyAllWindows()

    maze = cv2.imread(maze_path)
    cv2.imshow('maze', maze)

    cv2.setMouseCallback('maze', click_event)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    end_px = clicks[-1]
    end_x = math.floor(end_px[1] / 8)
    end_y = math.floor((end_px[0] - 2) / 8)
    return (end_x, end_y)


if __name__ == '__main__':
    '''
    探究影响因素
    1.学习率   目前最优：0.54
    2.折扣因子  目前最优：0.95
    3.epsilon   目前最优：0.2
    4.迷宫是否封住出口  目前最优：密封
    '''
    # 初始化参数与数据
    maze_path = 'maze.jpg'
    maze_matrix_closed = pd.read_csv('maze_matrix_closed.csv', header=None).values
    maze_matrix_open = pd.read_csv('maze_matrix_open.csv', header=None).values
    maze = [maze_matrix_closed, maze_matrix_open]
    info = pd.read_csv('info.csv')

    block_size = info['block_size'][0]
    start = (info['start_x'][0], info['start_y'][0])
    # end = (31, 63)
    end = (7, 69)

    lr = [0.25, 0.54]
    discount = 0.96
    epsilon = [0.5, 0.2]
    actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上,下,左,右

    for j in range(2):
        T1 = time.time()
        agent = QLearningAgent(alpha=0.53, gamma=0.95, epsilon=0.2, actions=range(len(actions)), start=start)
        env = MazeEnvironment(maze[j], end, start, maze_path, block_size)
        for i in range(1000):
            state = start
            while True:
                action = agent.choose_action(state)
                next_state, reward = env.step(state, action)
                agent.learn(state, action, reward, next_state)
                state = next_state
                if next_state == end:
                    break
        # move = agent.get_move()
        # env.draw_path_on_maze(move)
        T2 = time.time()
        if j == 0:
            print('迷宫封口时用时{}毫秒'.format((T2 - T1) * 1000))
        else:
            print('迷宫没封口时用时{}毫秒'.format((T2 - T1) * 1000))


