import numpy as np
import torch
import copy
import math

def get_node_degree(adj, i, l):
    return np.sum(adj[l, i, :]) + 1

def get_neighbors(adj, i, l):
    N_i = adj[l, i, :]
    ret = []
    for j in range(N_i.shape[0]):
        if N_i[j] == 1:
            ret.append(j)
    ret.append(i)
    return ret

class MessagePassing(torch.nn.Module):
    def __init__(self):
        super(MessagePassing, self).__init__()
        self.W = None

        self.adj = None
        self.feature = None

    def get_feature(self):
        self.L = self.feature.shape[1]
        self.nelem = self.feature.shape[2]
        return
    
    def read_adj(self, i):
        mol_name = f"./tmp/mol_adj_{i}.txt"
        with open(mol_name, "r") as f:
            lines = f.readlines()
            a, b, c, _ = lines[-1].split()
            a = int(a) + 1
            b = int(b) + 1
            c = int(c) + 1
            self.adj = np.zeros((a, b, c))
            for line in lines:
                l, i, j, val = line.split()
                l, i, j, val = int(l), int(i), int(j), float(val)
                self.adj[l, i, j] = val
        return
    
    def read_feature(self, i):
        feature_name = f"./feature/feature_{i}.txt"
        with open(feature_name, "r") as f:
            lines = f.readlines()
            a, b, c, _ = lines[-1].split()
            a = int(a) + 1
            b = int(b) + 1
            c = int(c) + 1
            self.feature = np.zeros((a, b, c))
            for line in lines:
                i, j, l, val = line.split()
                i, j, l, val = int(i), int(j), int(l), float(val)
                self.feature[i, j, l] = val
        feature = copy.deepcopy(self.feature)
        return feature
    
    def init_params(self):
        self.get_feature()
        self.nodes = self.adj.shape[1]
        size = self.nelem
        self.W = np.random.rand(size, size)
        self.W = torch.from_numpy(self.W)
        return
    
    def forward(self, feature):
        self.get_feature()
        nodes = self.adj.shape[1]
        ret = torch.zeros_like(feature)
        for i in range(nodes):
            for l in range(self.L):
                js = get_neighbors(self.adj, i, l)
                d_i = get_node_degree(self.adj, i, l)
                for j in js:
                    d_j = get_node_degree(self.adj, j, l)
                    ret[j, l, :] = torch.matmul(self.W, feature[i, l, :]) / math.sqrt(d_i * d_j)
        return ret
    
    def output_feature(self, fname, txt=""):
        with open(fname, "w") as f:
            for i in range(self.feature.shape[0]):
                for j in range(self.feature.shape[1]):
                    for k in range(self.feature.shape[2]):
                        val = self.feature[i, j, k]
                        txt += f"{i} {j} {k} {val}\n"
            f.write(txt)
        return

if __name__ == "__main__":
    mp = MessagePassing()
    feature = mp.read_feature(0)
    mp.read_adj(0)
    mp.init_params()
    feature = torch.from_numpy(feature)
    
    mp.output_feature("feature.txt")
    ret = mp.forward(feature)
