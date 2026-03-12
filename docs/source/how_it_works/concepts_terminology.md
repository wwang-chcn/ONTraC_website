# Concepts & Terminology

## Parameters for GNN model used in ONTraC

### patience

The training process will be terminated if the model does not improve after the number of epochs set by this parameter.

### min-delta

The model will be considered improved if the total loss decreases by the proportion set by this parameter.

### hidden-feats

The number of niche features after GCN processing (step 2).

### modularity-loss-weight

The modularity loss controls the spatial smoothness of niche clusters.

### purity-loss-weight

The purity loss controls the purity of cell type composition within each niche cluster.

### regularization-loss-weight

The regularization loss controls the balance of niche cluster sizes. The higher this weight is set, the more equal the size of each niche cluster.

### beta

The beta value of the softmax function used in generating niche cluster assignment matrix.
