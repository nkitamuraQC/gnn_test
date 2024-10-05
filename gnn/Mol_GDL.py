from message_passing import MessagePassing
from gnnclass import GNN
import torch


class Mol_GDL(torch.nn.Module):
    def __init__(self):
        super(Mol_GDL, self).__init__()
        self.mp = MessagePassing()
        self.feature = self.mp.read_feature(0)
        self.mp.read_adj(0)
        self.mp.init_params()
        self.gnn = GNN()

    def forward(self, features):
        features = self.mp.forward(features)
        features = self.gnn.forward(features)
        return features
