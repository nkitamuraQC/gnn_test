# gnn_test
gnn_test is an experimental code for predicting molecular properties with a graph neural network.

## Features
- molecular geometric deep learning (for the ESOL dataset)
  - see https://arxiv.org/abs/2306.15065
  - see also https://pubs.acs.org/doi/10.1021/ci034243x

## Usages

```shell
cd gnn
python mol_graph.py
python node_feature.py
python train.py
```

## Installation

```shell
conda create -n gnn_test python=3.9
conda activate gnn_test
git clone https://github.com/nkitamuraQC/gnn_test.git
pip install -r requirements.txt
```
