# Literature Review

    Paper 3: "node2vec: Scalable Feature Learning for Networks" (Grover et al.)

## Previous Approaches

- Hand-engineering domain-specific features based on expert knowlegde: this method doesn't generalize across different prediction tasks.
- Learning feature representation by solving an optimization problem: this method results in good accuracy with a cost of high training time complexity due to a big number of parameters.
- Classical appraoch based on linear and non-linear dimension reduction techniques such as **Principle Component Analysis, Multi-Dimensional Scaling** are expensive for large real-world networks, and resulting in latent represenations that give poor performance on various prediction tasks over networks.

## Proposed Method: `node2vec`

- A semi-supervise algorithm for scalable feature learning in network.
- This approach returns feature representations that maximize the likelihood of preserving network neighborhoods of nodes in a d-dimensional feature space.
- It uses 2nd order random walk approach to generate (sample) network neighborhoods for nodes.
- `node2vec` can learn representations that organize nodes based on their network roles and/or communities they belong to.
- Can also generate representation of edgdes.
- The experiment for this method is based on 2 prediction tasks:
  1. Multi-label classification task
  2. Link prediction task
