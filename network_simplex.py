# 单纯形法求解最优传输问题
import numpy as np
np.set_printoptions(precision=4, suppress=True, threshold=np.inf)
from copy import deepcopy
# 网络单纯形法
class network_simplex():
    def __init__(self, cost_matrix, supply, demand, num_iters=1):
        self.cost_matrix = cost_matrix
        self.supply = supply
        self.demand = demand
        self.num_iters = num_iters
        self.n = len(supply) # 工厂数目
        self.m = len(demand) # 仓库数目
        
    def north_west_corner(self, supply, demand):   
       # s: suply, d: demand
       s = deepcopy(supply)
       d = deepcopy(demand)
       
       # 初始化基本可行解, x为传输方案，每行之和等于供给量，每列之和等于需求量
       x = np.zeros([len(s), len(d)])
       i, j = 0, 0
       while i < len(s) and j < len(d):
           if s[i] > d[j]:
               x[i, j] = d[j]
               s[i] -= d[j]
               j += 1
           else:
               x[i, j] = s[i]
               d[j] -= s[i]
               i += 1
       return x
    
    def get_dual(self, edge, start_node):
        # 计算对偶变量f和g
        f = np.zeros([self.n])
        g = np.zeros([self.m])
        f_flag = np.zeros([self.n]) # 标记f是否已经计算
        g_flag = np.zeros([self.m])
        f_near_dict = [] # 记录f的邻接节点
        g_near_dict = []
        
        f[start_node] = 0
        f_flag[start_node] = 1
        
        for j in range(self.m):
            if edge[start_node, j] == 1:
                f_near_dict.append([start_node, j])
                
        while len(f_near_dict)!=0 or len(g_near_dict)!=0:
            if len(f_near_dict)!=0: # 遍历f的邻接节点,计算g
                for element in f_near_dict:
                    start_node = element[0]
                    item = element[1]
                    g_flag[item] = 1
                    g[item] = self.cost_matrix[start_node, item] - f[start_node]
                    # 更新g的邻接节点
                    for i in range(self.n):
                        if f_flag[i] == 0 and edge[i, item] == 1:
                            g_near_dict.append([item, i])
                f_near_dict = []
                
            else:
                for element in g_near_dict:
                    start_node = element[0]
                    item = element[1]
                    f_flag[item] = 1
                    f[item] = self.cost_matrix[item, start_node] - g[start_node]
                    # 更新f的邻接节点
                    for j in range(self.m):
                        if g_flag[j] == 0 and edge[item, j] == 1:
                            f_near_dict.append([item, j])
                g_near_dict = []
        return f, g 
    
    # 深度优先搜索
    def dfs(self, edge, edge_pass, cycle, cycle_ans):
        if cycle[0] == cycle[-1] and len(cycle) %2 == 1 and len(cycle) > 1:
            cycle_ans.append(deepcopy(cycle))
            return
        if len(cycle) >= self.n + self.m: # 遍历完所有节点
            return
        start_node = cycle[-1]
        if len(cycle) % 2 == 1: # 遍历节点g
            for j in range(self.m):
                if edge[start_node, j] == 1 and edge_pass[start_node, j] == 0:
                    cycle.append(j)
                    edge_pass[start_node, j] = 1
                    self.dfs(edge, edge_pass, cycle, cycle_ans)
                    cycle.pop()
                    edge_pass[start_node, j] = 0
        else: # 遍历节点f
            for i in range(self.n):
                if edge[i, start_node] == 1 and edge_pass[i, start_node] == 0:
                    cycle.append(i)
                    edge_pass[i, start_node] = 1
                    self.dfs(edge, edge_pass, cycle, cycle_ans)
                    cycle.pop()
                    edge_pass[i, start_node] = 0
        return
    
    def find_cycle(self, edge, start_node):
        # 从f节点出发寻找含有start_node的环
        cycle = []
        cycle.append(start_node)
        cycle_ans = [] # 记录所有环，注意最后一个元素等于第一个元素
        edge_pass = np.zeros([self.n, self.m]) # 记录边是否被遍历
        self.dfs(edge, edge_pass, cycle, cycle_ans)
        return cycle_ans
   
    def solve(self):
        # 西北角法初始化传输方案
        P = self.north_west_corner(self.supply, self.demand)
        print(f"初始传输方案：{P}")
        edge = np.zeros([self.n, self.m])
        edge_update = True
        while True: 
            # 更新边
            if edge_update:
                for i in range(self.n):
                    for j in range(self.m):
                        if P[i, j] > 0:
                            edge[i, j] = 1
                        else:
                            edge[i, j] = 0
            # 找到计算对偶变量的起始节点
            start_node = 0
            flag_start = False  
            for i in range(self.n):
                for j in range(self.m):
                    if edge[i, j] == 1:
                        start_node = i
                        flag_start = True
                        break
                if flag_start:
                    break
                
            # 计算对偶变量
            f, g = self.get_dual(edge, start_node)
            # 找到第一个违反对偶约束条件的节点
            violate_pair_list = []
            violate_num = 0
            for i in range(self.n):
                for j in range(self.m):
                    if (int)(f[i] + g[j])>self.cost_matrix[i, j]:
                        violate_pair_list.append([i, j])
                        violate_num += 1
            if violate_num == 0:
                print(f"最优传输方案：{P}")
                break
            
            updata_flag = False
            violate_pair = violate_pair_list[0]
            # 加入两个violate节点
            edge[violate_pair[0], violate_pair[1]] = 1
            # 寻找含有violate节点的环
            violate_cycle = self.find_cycle(edge, violate_pair[0])
            if len(violate_cycle) == 0:
                edge_update = False # 直接更新对偶变量
            else:
                # 更新边
                edge_update = True
                for cycle in violate_cycle:
                    if violate_pair[0] != cycle[0] or violate_pair[1] != cycle[1]: # 确保violate节点在环中第一和第二的位置
                        continue
                    # 计算环中最小的传输量，更新传输方案
                    min_value = 1e9
                    for i in range(len(cycle)-1):
                        if i % 2 == 1:
                            value = P[cycle[i+1], cycle[i]]
                            if value < min_value:
                                min_value = value
                    if min_value == 0:
                        continue
                    # 更新传输方案
                    for i in range(len(cycle)-1):
                        if i % 2 == 0:
                            P[cycle[i], cycle[i+1]] += min_value
                        else:
                            P[cycle[i+1], cycle[i]] -= min_value
                    print(f"更新传输方案：{P}")
                    break
        return P
            
            
cost_matrix = np.array([[14, 5, 21, 23, 2, 26], 
                        [10, 1, 7, 35, 8, 9], 
                        [16, 13, 27, 6, 11, 12],
                        [15, 20, 18, 36, 19, 17],
                        [22, 28, 3, 24, 25, 4],
                        [29, 30, 31, 32, 33, 34]])
supply = [20, 40, 60, 50, 30, 10]
demand = [22, 34, 41, 27, 17, 69]

ns = network_simplex(cost_matrix, supply, demand)
# 测试环搜索算法
# edge = np.array([[1, 0, 1, 0, 1],
#                  [1, 1, 0, 0, 0],
#                  [0, 1, 1, 0, 0],
#                  [1, 0, 0, 1, 0],
#                  [0, 0, 0, 1, 1]])
# start_node = 0
# cycle = ns.find_cycle(edge, start_node)
# assert len(cycle) == 4
# 计算最优传输方案
P_optimal = ns.solve()
