from message_passing import message_passing
from gnn import GNN
import gnn
import torch

class Mol_GDL(torch.nn.Module):
    def __init__(self):
        super(Mol_GDL, self).__init__()
        self.mp = message_passing()
        self.feature = self.mp.read_feature(0)
        self.mp.read_adj(0)
        self.mp.init_params()
        self.gnn = GNN()

    def forward(self, features):
        features = self.mp.forward(features)
        features = self.gnn.forward(features)
        return features


    def dense(self, features):
        return
