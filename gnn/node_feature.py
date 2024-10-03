import numpy as np
import json, os


def parse_xyz(n_mols=800):
    all = []
    for i in range(n_mols):
        mol = []
        with open(f"./xyz/out_{i}.xyz", "r") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if i > 1:
                    elem, x, y, z = line.split()
                    x, y, z = float(x), float(y), float(z)
                    mol.append([elem, np.array([x, y, z])])
        all.append(mol)
    return all


class NodeFeature:
    def __init__(self, xyz, i):
        os.system("mkdir feature")
        self.xyz = xyz
        self.x_min = 0.2
        self.x_max = 10
        self.L = 10
        self.nelems = 20
        self.natoms = 400
        self.log = f"./feature/info_{i}.log"
        self.feature_log = f"./feature/feature_{i}.txt"
        self.feature_file = f"./feature/feature_{i}.pt"

    def seperate(self):
        diff = (self.x_max - self.x_min) / (self.L)
        intervals = [self.x_min + diff * i for i in range(self.L + 1)]
        self.intervals = intervals
        return intervals

    def get_nelems(self):
        self.save = {
            "C": 0,
            "H": 1,
            "O": 2,
            "N": 3,
            "P": 4,
            "Cl": 5,
            "F": 6,
            "Br": 7,
            "S": 8,
            "Si": 9,
            "I": 10,
        }
        return

    def get_node_feature(self):
        intervals = self.seperate()
        # n = self.get_nelems()
        # natoms = len(self.xyz)
        feature = np.zeros((self.natoms, self.L, self.nelems), dtype=int)
        for l in range(self.L):
            a = intervals[l]
            b = intervals[l + 1]
            for i in range(len(self.xyz)):
                xyz1 = self.xyz[i][1]
                for j in range(len(self.xyz)):
                    xyz2 = self.xyz[j][1]
                    elem = self.xyz[j][0]
                    dist = np.linalg.norm(xyz1 - xyz2)
                    if a <= dist and dist < b:
                        idx = self.save[elem]
                        feature[i, l, idx] += 1
        self.feature = feature
        np.save(self.feature_file, feature)
        return feature

    def output_log(self):
        dic = {}
        dic["intervals"] = self.intervals
        dic["elems"] = self.save
        with open(self.log, "w") as f:
            json.dump(dic, f, indent=4)
        return

    def output_feature(self, txt=""):
        with open(self.feature_log, "w") as f:
            for i in range(self.feature.shape[0]):
                for j in range(self.feature.shape[1]):
                    for k in range(self.feature.shape[2]):
                        val = self.feature[i, j, k]
                        txt += f"{i} {j} {k} {val}\n"
            f.write(txt)
        return


if __name__ == "__main__":
    xyz_s = parse_xyz()
    for i, xyz in enumerate(xyz_s):
        node = NodeFeature(xyz, i)
        node.get_nelems()
        feature = node.get_node_feature()
        node.output_feature()
