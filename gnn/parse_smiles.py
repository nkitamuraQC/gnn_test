import numpy as np


class ParseSmiles:
    def __init__(self, smiles):
        self.parser = Parser(smiles)
        self.parser.parse()
        self.smiles = smiles
        self.natoms = len(self.parser.atoms_elems)
        self.atoms_elems = self.parser.atoms_elems
        self.side_chain = self.parser.side_chain
        self.ring = self.parser.ring
        self.graph_matrix = np.zeros((self.natoms, self.natoms))

    def get_chain(self):
        for i in range(len(self.atoms_elems) - 1):
            if self.side_chain[i] == self.side_chain[i + 1]:
                self.graph_matrix[i, i + 1] = 1
                self.graph_matrix[i + 1, i] = 1
            if self.side_chain[i] + 1 == self.side_chain[i + 1]:
                self.graph_matrix[i, i + 1] = 1
                self.graph_matrix[i + 1, i] = 1

        for i in range(len(self.atoms_elems) - 1):
            level = self.side_chain[i]
            add_bond_flag = False
            for j in range(i + 1, len(self.atoms_elems)):
                if self.side_chain[j] == level:
                    if self.side_chain[j - 1] > level:
                        pass
                    else:
                        break
                if self.side_chain[j] > level:
                    add_bond_flag = True
                elif add_bond_flag:
                    self.graph_matrix[i, j] = 1
                    self.graph_matrix[j, i] = 1
                    break

        print("graph_matrix@chain =\n", self.graph_matrix)
        return

    def get_ring(self):
        for i in range(len(self.atoms_elems)):
            for j in range(len(self.atoms_elems)):
                if self.ring[i] == self.ring[j] and self.ring[i] != 0 and i != j:
                    self.graph_matrix[i, j] = 1
                    self.graph_matrix[j, i] = 1
        print("graph_matrix@ring =\n", self.graph_matrix)
        return

    def get_graph(self):
        self.get_chain()
        self.get_ring()
        return

    def show(self):
        return


class Parser:
    def __init__(self, smiles):
        self.smiles = smiles
        self.target_elems = ["c"]
        self.atoms_elems = []  # natoms
        self.side_chain = []  # natoms
        self.ring = []  # natoms

    def parse_atoms_elems(self):

        for smi in self.smiles:
            if smi in self.target_elems:
                self.atoms_elems.append(smi)

        return

    def parse_sidechain(self):

        level = 0

        for smi in self.smiles:
            if smi in self.target_elems:
                self.side_chain.append(level)
            if smi == "(":
                level += 1
            if smi == ")":
                level -= 1

        return

    def parse_ring(self):
        atm_idx = -1
        ring = [0 for i in range(len(self.atoms_elems))]
        for smi in self.smiles:
            if smi in self.target_elems:
                atm_idx += 1
            try:
                a = int(smi)
            except ValueError:
                continue
            else:
                ring[atm_idx] = a
        self.ring = ring
        return

    def parse(self):
        self.parse_atoms_elems()
        self.parse_sidechain()
        self.parse_ring()

        print("atoms_elems =", self.atoms_elems)
        print("side_chain =", self.side_chain)
        print("ring =", self.ring)
        return


if __name__ == "__main__":
    smiles = "c1ccc(c)c1"
    smi = ParseSmiles(smiles)
    smi.get_graph()
    print(smi.graph_matrix)
