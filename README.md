# collab-link-prediction

Doing link prediction 

## The dataset: Collaboration network


## Graph topological features
This baseline model tests the ability of features intrinsic to the graph structure to model the evolution of the graph. In the collaboration network, these features intuitively make sense as predictors for new collaborations. For example, if two authors share coauthors, it is likely they operate in the same subdomain of science and such may be working on similar topics. In [[1]](#1), the authors provide evidence of the predictive ability of a variety of graph topological features also used in this project:

- Resource Allocation Index
- Jaccard Coefficient
- Adamic Adar Index
- Preferential Attachment
- Common Neighbor Centrality

These are implemented using NetworkX [[2]](#2) from the [link prediction](https://networkx.org/documentation/stable/reference/algorithms/link_prediction.html) module.

### Usage

Feature generation script usage:
```bash
python -m lynks.topo_features \
    --config_path=./configs/experiments.toml \
    --out_dir=./data/processed
```

Train clf

```bash
python -m lynks.train_topo_clf \
    --config_path=./configs/experiments.toml \
    --data_dir=./data/processed \
    --result_dir=./data/results \
    --n_proc=4
```


## References
<a id="1">[1]</a> 
David Liben-Nowell and Jon Kleinberg. 2003. The link prediction problem for social networks. In Proceedings of the twelfth international conference on Information and knowledge management (CIKM '03). Association for Computing Machinery, New York, NY, USA, 556–559. DOI:https://doi.org/10.1145/956863.956972

<a id="2">[2]</a> 
Aric A. Hagberg, Daniel A. Schult and Pieter J. Swart, “Exploring network structure, dynamics, and function using NetworkX”, in Proceedings of the 7th Python in Science Conference (SciPy2008), Gäel Varoquaux, Travis Vaught, and Jarrod Millman (Eds), (Pasadena, CA USA), pp. 11–15, Aug 2008