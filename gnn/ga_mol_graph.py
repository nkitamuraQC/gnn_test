#個体の各遺伝子を決めるために使用
import random
#DEAPの中にある必要なモジュールをインポート
from deap import base
from deap import creator
from deap import tools
from deap import algorithms


import numpy as np
import pandas as pd
import networkx as nx
import copy
import matplotlib.pyplot as plt
from pyscf import fci
import math

class Graph_QC:
    def __init__(self, index, norb):
        self.index = index
        self.norb = norb

        self.t = 1
        self.U = 10
        self.V = 5

    def decimal2graph(self):
        binary = bin(self.index)
        print(binary)
        data = binary[2:]
        adj_vec = []
        for d in range(self.norb**2):
            if d < len(data):
                adj_vec.append(int(data[d]))
            else:
                adj_vec.append(0)
        self.adj_mat = np.array(adj_vec).reshape((self.norb, self.norb))
        tril = np.tril(self.adj_mat, k=-1)
        self.adj_mat = tril + tril.T
        self.G = nx.from_pandas_adjacency(pd.DataFrame(self.adj_mat), create_using=nx.DiGraph)
        return
    
    def show(self, i):
        G = self.G
        print( G.nodes(data=True) )
        # 隣接情報（重みの属性）を確認
        print( G.adj )

        pos = nx.spring_layout(G)

        other_param={
            "node_color":"red",
            "with_labels":True
        }

        plt.figure(figsize=(6, 6))
        nx.draw_networkx_edges(G, pos)
        nx.draw_networkx_nodes(G, pos)
        plt.axis('off')
        plt.savefig(f"graph_{i}.png")

        return
    
    def get_int1e(self):
        adj_mat = copy.deepcopy(self.adj_mat)
        return adj_mat * self.t
    
    def get_int2e(self):
        adj_mat = copy.deepcopy(self.adj_mat)
        nodes = adj_mat.shape[0]

        adj_mat *= self.V

        for n in range(nodes):
            if adj_mat[n, n] == 0:
                adj_mat[n, n] += self.U

        int2e = np.zeros((nodes, nodes, nodes, nodes))
        for n in range(nodes):
            for m in range(nodes):
                int2e[n,n,m,m] = adj_mat[n, m]
        return int2e
    
def list2bin(indivisual):
    bin_ = "0b"
    for s in indivisual:
        bin_ += str(s)
    return bin_


class CalcModel:
    def __init__(self, natom, nelec):
        self.natom = natom
        self.nelec = nelec
        
    def calc(self, individuals):
        save = []
        for individual in individuals:
            if individual < 0:
                individual *= -1
            norb = self.natom # 2
            individual = math.floor(individual)
            graph = Graph_QC(individual, norb)
            graph.decimal2graph()
            int1e = graph.get_int1e()
            int2e = graph.get_int2e()
            self.Solver = fci.direct_spin0.FCISolver()
            self.e, self.c = self.Solver.kernel(int1e, int2e, norb, self.nelec)
            val = self.calc_observable()
            save.append(val)

        return (np.max(save),)
    
    def calc_observable(self):
        return 0
    



class Search_Mol:
    def __init__(self, obfunc, natom, NGEN=5, POP=80, CXPB=0.9, MUTPB=0.1):
        self.obfunc = obfunc
        self.natom = natom
        self.gene_max = 2**(natom**2)
        self.mu = [0.0 for i in range(16)]
        self.sigma = [20.0 for i in range(16)]
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("individual", list, fitness=creator.FitnessMin)
        self.toolbox = base.Toolbox()
        self.toolbox.register("attribute", random.randint, 1, self.gene_max)
        self.toolbox.register("individual", tools.initRepeat, creator.individual, self.toolbox.attribute, natom**2)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("select", tools.selTournament, tournsize=POP)
        self.toolbox.register("mate", tools.cxBlend,alpha=0.2)
        self.toolbox.register("mutate", tools.mutGaussian, mu=self.mu, sigma=self.sigma, indpb=0.2)
        self.toolbox.register("evaluate", self.obfunc)
        random.seed(64)
        self.NGEN = NGEN
        self.POP = POP
        self.CXPB = CXPB
        self.MUTPB = MUTPB
        self.pop = self.toolbox.population(n=self.POP)
        
    def opt(self):
        for individuals in self.pop:
            #for individual in individuals:
            individuals.fitness.values = self.toolbox.evaluate(individuals)
        hof = tools.ParetoFront()
        algorithms.eaSimple(self.pop, self.toolbox, cxpb=self.CXPB, mutpb=self.MUTPB, ngen=self.NGEN, halloffame=hof)
        best_ind = tools.selBest(self.pop, 1)[0]
        return
    
if __name__ == "__main__":
    nelec = 2
    natom = 4
    cm = CalcModel(natom, nelec)
    ob = cm.calc
    sm = Search_Mol(ob, natom)
    sm.opt()
