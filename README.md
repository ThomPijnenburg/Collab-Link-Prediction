# collab-link-prediction

Knowledge graphs are dynamic data structures, for example due to scientific progress or due to user interactions in social networks. With Link Prediction we attempt to build ML models that can surface missing edges in the graph.

## The dataset: collaboration network
Here we are looking at the collaboration graph `ogbl-collab` from the OGB  library [[1]](#1). This is an undirected graph representing a collaboration network between authors. Each node represents an author, and each edge a collaboration. The nodes come with a 128-dim feature vector representation of the averaging of word embeddings of the author's published papers. Edges come with the attributes year and weight reflecting the number of coauthored papers in that year. This graph is a multigraph since there can be more than one edge between two authors if they collaborate in more than one year.

## Graph topological features
This baseline model tests the ability of features intrinsic to the graph structure to model the evolution of the graph. In the collaboration network, these features intuitively make sense as predictors for new collaborations. For example, if two authors share coauthors, it is likely they operate in the same subdomain of science and such may be working on similar topics. In [[2]](#2), the authors provide evidence of the predictive ability of a variety of graph topological features also used in this project:

- Resource Allocation Index
- Jaccard Coefficient
- Adamic Adar Index
- Preferential Attachment
- Common Neighbor Centrality

These are implemented using NetworkX [[3]](#3) from the [link prediction](https://networkx.org/documentation/stable/reference/algorithms/link_prediction.html) module.

Feature generation script usage:
```bash
poetry run python -m lynks.topo_features \
    --config_path=./configs/experiments.toml \
    --out_dir=./data/processed
```

Train a binary classification model using the generated data

```bash
poetry run python -m lynks.train_topo_clf \
    --config_path=./configs/experiments.toml \
    --data_dir=./data/processed \
    --result_dir=./data/results \
    --n_proc=4
```

### Todo

- [ ] rewrite lp model evaluation to use ranking
- [ ] add MLP classifier

## Graph embedding models
Knowledge Graph Embedding (KGE) models attempt to learn low dimensional numerical representations for the entities and sometimes relations in the graph. These representations are useful for many tasks in information retrieval, property prediction or classification and link prediction.

### Todo
- [ ] pykeen wrapper to load train/valid/test triples
- [ ] train script for KGE model
- [ ] hpo
- [ ] add notebook to run training on google collab


## References
<a id="1">[1]</a> 
Hu, Weihua, et al. ‘Open Graph Benchmark: Datasets for Machine Learning on Graphs’. ArXiv:2005.00687 [Cs, Stat], Feb. 2021. arXiv.org, http://arxiv.org/abs/2005.00687.

<a id="2">[2]</a> 
David Liben-Nowell and Jon Kleinberg. 2003. The link prediction problem for social networks. In Proceedings of the twelfth international conference on Information and knowledge management (CIKM '03). Association for Computing Machinery, New York, NY, USA, 556–559. DOI:https://doi.org/10.1145/956863.956972

<a id="3">[3]</a> 
Aric A. Hagberg, Daniel A. Schult and Pieter J. Swart, “Exploring network structure, dynamics, and function using NetworkX”, in Proceedings of the 7th Python in Science Conference (SciPy2008), Gäel Varoquaux, Travis Vaught, and Jarrod Millman (Eds), (Pasadena, CA USA), pp. 11–15, Aug 2008