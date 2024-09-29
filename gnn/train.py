import os.path as osp
import numpy as np
import torch, os
from torch_geometric.data import Dataset, download_url
from torch_geometric.data import Data, InMemoryDataset
from Mol_GDL import Mol_GDL
from message_passing import message_passing
import pandas as pd

device = "cpu"

import torch
from torch_geometric.data import InMemoryDataset, download_url
        

all = 800

def parse_csv():
    df = pd.read_csv("eSOL.csv")

    sol = df["measured log solubility in mols per litre"]

    nrow = len(df)

    sol_list = []

    os.system("mkdir xyz")
    #os.chdir("./xyz")
    
    for i in range(nrow):
        sol_list.append(sol.iloc[i])
    return sol_list



def train(epoch, n_train):
    model.train()
    loss_all = 0
    mp = message_passing()
    ys = parse_csv()

    mae_loss = torch.nn.L1Loss()
    mse_loss = torch.nn.MSELoss()
    for i in range(n_train):
        features = torch.from_numpy(mp.read_feature(i))
        #features = features.to(torch.float32)
        data = features.to(device)
        optimizer.zero_grad()
        output = model.forward(data)
        #print(output)
        #print(ys[i])
        loss = mae_loss(output, torch.tensor(ys[i]))
        #loss = mse_loss(output, torch.tensor(ys[i]))
        loss.backward()
        loss_all += loss.item()
        optimizer.step()
    return loss_all / n_train


@torch.no_grad()
def test(epoch, n_train):
    model.eval()
    loss_all = 0
    mae_loss = torch.nn.L1Loss()
    mp = message_passing()
    ys = parse_csv()
    for i in range(n_train, all):
        features = torch.from_numpy(mp.read_feature(i))
        #features = features.to(torch.float32)
        #data = features.to(device)
        output = model(features)
        print("output =", output)
        print("target =", ys[i])
        loss = mae_loss(output, torch.tensor(ys[i]))
        loss_all += loss.item()
    return loss_all / (all - n_train)


model = Mol_GDL()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
max_epoch = 1001
#max_epoch = 3

best_val_loss = test_loss = 0
best_model = None
loss_hist = []
val_hist = []
test_hist = []
for epoch in range(1, max_epoch):
    train_loss = train(epoch, 600)
    val_loss = test(epoch, 600)
    #test_loss = test(test_loader)
    loss_hist.append(train_loss)
    #val_hist.append(val_loss)
    #test_hist.append(test_loss)
    if best_model is None or val_loss > best_val_loss:
        best_val_loss = test_loss
        best_model = model
        torch.save(best_model, "./model.pth")
    if val_loss > best_val_loss:
        best_val_loss = val_loss
    print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')


