import numpy as np
import pandas as pd
import os

def parse_csv():
    df = pd.read_csv("eSOL.csv")

    sol = df["measured log solubility in mols per litre"]
    smiles = df["smiles"]

    nrow = len(df)

    sol_list = []
    xyz = []

    os.system("mkdir xyz")
    #os.chdir("./xyz")
    
    for i in range(nrow):
        sol_list.append(sol.iloc[i])
        sm = smiles.iloc[i]
        print("sm =", sm)
        os.system(f"echo '{sm}' | obabel -i smi -o xyz -O ./xyz/out_{i}.xyz --gen3D")
        
        xyz_ = []
        with open(f"./xyz/out_{i}.xyz", "r") as f:
            lines = f.readlines()
            for i in range(len(lines)):
                if i < 2:
                    continue
                elem, X, Y, Z = lines[i].split()
                X, Y, Z = float(X), float(Y), float(Z)
                xyz_.append([elem, np.array([X, Y, Z])])
        xyz.append(xyz_)

    print("sol_list =", sol_list)
    #print("xyz =", xyz)
    return sol_list, xyz

class mol_graph:
    def __init__(self, xyz):
        self.xyz = xyz # [["C", xyz], ["C", xyz]]
        self.x_min = 10
        self.x_max = 100
        self.L = 10

        self.save_file = ""
        self.log_file = ""

    def seperate(self):
        diff = (self.x_max - self.x_min) / (self.L)
        intervals = [self.x_min + diff * i for i in range(self.L+1)]
        self.intervals = intervals
        return intervals

    def get_natoms(self):
        return len(self.xyz)

    def get_adjMat(self):
        intervals = self.seperate()
        n = self.get_natoms()
        adj = np.zeros((self.L, n, n), dtype=int)

        for l in range(self.L):
            for i in range(len(self.xyz)):
                xyz1 = self.xyz[i][1]
                for j in range(len(self.xyz)):
                    xyz2 = self.xyz[j][1]
                    dist = np.linalg.norm(xyz1-xyz2)
                    if dist > intervals[l] and dist < intervals[l+1]:
                        adj[l,i,j] = 1
        return adj

    def save(self, adj, filename):
        np.save(filename, adj)
        return
    
    def output(self, adj, filename, txt=""):
        with open(filename, "w") as f:
            for l in range(adj.shape[0]):
                for i in range(adj.shape[1]):
                    for j in range(adj.shape[2]):
                        val = adj[l,i,j]
                        txt += f"{l} {i} {j} {val}\n"
            f.write(txt)
        return
    
if __name__ == "__main__":
    os.system("mkdir tmp")
    sol_s, xyz_s = parse_csv()
    sol_name = "./tmp/sol_{i}.txt"
    mol_name = "./tmp/mol_adj_{i}.txt"
    for i, xyz in enumerate(xyz_s):
        print("xyz =", xyz)
        mol = mol_graph(xyz)
        adj = mol.get_adjMat()
        mol.output(adj, mol_name.format(i=i))




