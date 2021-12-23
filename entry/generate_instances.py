import random

from core.problem import Problem


if __name__ == '__main__':
    nb_instances = 20
    pth = '../problems/valid'
    for i in range(nb_instances):
        nb_agents = random.randint(40, 60)
        p1 = random.random() * 0.25
        p1 = max(p1, .1)
        p2 = random.random() * 0.4 + 0.3
        dom_size = random.randint(3, 15)
        p = Problem()
        p.random_binary(nb_agents, dom_size, p1, p2)
        p.save(f'{pth}/{i}.xml')