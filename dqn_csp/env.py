import random

import torch

from core.parser import parse


class Environment:
    def __init__(self, pth, device='cpu', var_id_dim=4, var_id_ub=0.01):
        self.var_dom, self.functions = parse(pth)
        self.var_id_encoding = dict()
        self.device = torch.device(device)
        for name, _ in self.var_dom:
            self.var_id_encoding[name] = torch.rand(var_id_dim, device=self.device, dtype=torch.float32) * var_id_ub
        var_dom = {x[0]: x[1] for x in self.var_dom}
        self.var_dom = var_dom
        self.ordering = []
        self.cur_var_start_idx = -1
        self.partial_assignment = dict()
        self.cur_involved_functions = []

    def reset(self, shuffle=True):
        self.ordering = list(self.var_dom.keys())
        self.partial_assignment.clear()
        if shuffle:
            random.shuffle(self.ordering)

    def observe(self):
        admissible_vars, admissible_functions = set(), []
        self.cur_involved_functions.clear()
        for func, var1, var2 in self.functions:
            if var1 not in self.partial_assignment or var2 not in self.partial_assignment:
                admissible_vars.update([var2, var1])
                admissible_functions.append((func, var1, var2))
            if var1 == self.ordering[0] and var2 in self.partial_assignment:
                self.cur_involved_functions.append((func, var1, var2))
            elif var2 == self.ordering[0] and var1 in self.partial_assignment:
                self.cur_involved_functions.append((func, var1, var2))
        x = []
        var_start_index = dict()
        scatter_index = []
        idx = 0
        for var in admissible_vars:
            var_start_index[var] = len(x)
            dom = 1 if var in self.partial_assignment else self.var_dom[var]
            for _ in range(dom):
                x.append(self.var_id_encoding[var])
                scatter_index.append(idx)
            idx += 1
        scatter_index = torch.tensor(scatter_index, dtype=torch.long, device=self.device)
        x = torch.stack(x, dim=0)
        edge_index = [[], []]
        src, dest = edge_index
        edge_attr = []
        for func, var1, var2 in admissible_functions:
            if var1 in self.partial_assignment:
                dom1 = [self.partial_assignment[var1]]
            else:
                dom1 = [k for k in range(self.var_dom[var1])]
            if var2 in self.partial_assignment:
                dom2 = [self.partial_assignment[var2]]
            else:
                dom2 = [k for k in range(self.var_dom[var2])]
            for i in dom1:
                for j in dom2:
                    edge_attr.append(func[i][j])
                    edge_attr.append(func[i][j])
                    src.append(var_start_index[var1] + i)
                    dest.append(var_start_index[var2] + j)
                    dest.append(var_start_index[var1] + i)
                    src.append(var_start_index[var2] + j)
        edge_index = torch.tensor(edge_index, dtype=torch.long, device=self.device)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32, device=self.device)

        cur_var = self.ordering[0]
        self.cur_var_start_idx = var_start_index[cur_var]
        action_space = [self.cur_var_start_idx + i for i in range(self.var_dom[cur_var])]
        return x, edge_index, edge_attr, scatter_index, action_space

    def act(self, action, scale=1.0):
        action -= self.cur_var_start_idx
        cur_var = self.ordering.pop(0)
        reward = 0
        for func, var1, var2 in self.cur_involved_functions:
            if cur_var == var1:
                reward += func[action][self.partial_assignment[var2]]
            else:
                reward += func[self.partial_assignment[var1]][action]
        return -reward / scale, len(self.ordering) == 0
