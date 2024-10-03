from torch_geometric import nn
from torch.nn import Linear, ReLU
import torch

# self.L = 10
# self.nelems = 20
# self.natoms = 100

mlp_input = 20  # nelems
mlp_output = 1  # target

in_feature = 200  #
out_feature = 200


class GNN(torch.nn.Module):
    def __init__(self):
        super(GNN, self).__init__()
        self.pooling = nn.pool.global_max_pool
        self.linear = Linear(in_feature, out_feature)
        self.ReLU = ReLU()
        self.linear2 = Linear(mlp_input, mlp_input)
        self.linear3 = Linear(mlp_input, mlp_input)
        self.linear4 = Linear(mlp_input, mlp_output)
        self.readout = nn.pool.global_max_pool

    def forward(self, feature):
        node = feature.shape[0]
        L = feature.shape[1]
        nelem = feature.shape[2]
        feature = feature.reshape((node, -1))
        # print("feature_size1 =", feature.size())
        batch = torch.tensor([0 for i in range(node)])
        feature = self.pooling(feature, batch)
        # print("feature_size2 =", feature.size())
        feature = feature.to(torch.float32)
        feature = feature.reshape((1, -1))
        # print("feature_size3 =", feature.size())
        feature = self.linear(feature)
        # print("feature_size4 =", feature.size())
        feature = self.ReLU(feature)
        # print("feature_size5 =", feature.size())
        batch = torch.tensor([0 for i in range(L)])
        feature = feature.reshape((L, -1))
        # print("feature_size6 =", feature.size())
        feature = self.readout(feature, batch)
        # print("feature_size7 =", feature.size())
        feature = self.linear2(feature)
        feature = self.ReLU(feature)
        feature = self.linear3(feature)
        feature = self.linear4(feature)
        # print("feature_size8 =", feature.size())
        return feature[0]

    def output_vector(self):
        node = feature.shape[0]
        L = feature.shape[1]
        nelem = feature.shape[2]
        feature = feature.reshape((node, -1))
        # print("feature_size1 =", feature.size())
        batch = torch.tensor([0 for i in range(node)])
        feature = self.pooling(feature, batch)
        # print("feature_size2 =", feature.size())
        feature = feature.to(torch.float32)
        feature = feature.reshape((1, -1))
        # print("feature_size3 =", feature.size())
        feature = self.linear(feature)
        # print("feature_size4 =", feature.size())
        feature = self.ReLU(feature)
        # print("feature_size5 =", feature.size())
        batch = torch.tensor([0 for i in range(L)])
        feature = feature.reshape((L, -1))
        # print("feature_size6 =", feature.size())
        feature = self.readout(feature, batch)
        # print("feature_size7 =", feature.size())
        feature = self.linear2(feature)
        feature = self.ReLU(feature)
        feature = self.linear3(feature)
        return feature
