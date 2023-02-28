# Literature Review

    Paper 1: "graph2vec: Learning Distributed Representation of Graphs" (Narayanan et al.)

## Prevoius approaches:

- **Graph Kernels and handicraft features**:

  - Graph kernels evaluate smililarity (kernel value) between a pair graphs G and G' by recursively decomposing them into substructures.
  - Can be used for clustering and classification.
  - Exhibit **2 limitation** 😒:
    1. Many of the the kernel methods do not provide explicit graph embeddings --> Unusable with graph data! 👎
    2. The substructures (walks, paths, etc) by these kernel are determined manually with specific well defined functions that helps extract substructures from graphs.
       - This could be fixed by replacing the handicraft features with ones that are learnt automatically from data

- **Learning substructure embeddings**:

  - Embeddings of nodes, paths, and subgraphs
  - Is incapable of learning the representation of the whole graph --> cannot be used in graph classification. 😒
  - Could try to learn the representation of the whole graph via trivia extensions like averaging or max pooling over the substructure embeddings, but this will leads to suboptimal results.

- **Learning task-specific graph embeddings**:
  - Is a supervised approach, with proven good results
  - Exhibit **2 limitations** that reduce their usability:
    1. It requires a large amount of labeled data
    2. The representation thus learnt are specific to one particular ML task and cannot be transfered to other tasks.

✅ To solve the above approaches' limitations, we need a completey unsupervised approach that can capture the generic characteristic of the entire graphs in the form of their embeddings.

## Proposed Method

- **Neural embedding framework named `graph2vec`**:
  - Inspired by document embedding models that exploit the way how words/word sequences compose documents to learn their embeddings.
  - In `graph2vec`, an entire graph are viewed as a document and the rooted subgraphs around every node in the graph as words that compose the document.
  - It provides key advantages like:
    - **Unsupervised representaion learning**: no need labelled data
    - **Task-agnostic embeddings**: Does not leverage on any task-specific information, the embeddings are generic that can be used across all tasks that use graph.
    - **Data-driven embeddings**: Unlike graph kernels, `graph2vec` learns graph embeddings from a large corpus of graph data.
    - **Capture structure equivalence**: Preserve _structural equivalence_, hence this ensures `graph2vec`'s representation learining process yields similar embeddings for structurally similar graphs.

## Notations

Given a set of graphs $\mathbb{G} = \{G_1,G_2,\dots\}$ and a positive integer $\delta$ (i.e., expected embedding size). The goal is to learn $\delta\text{-dimensional distributed representations}$ for every graph $G_i \in \mathbb{G}$.
The matrix of representations of all graphs is denoted as $\Phi \in \mathbb{R}^{|\mathbb{G}|\text{ x }\delta}$

- Let $G = (N,E,\lambda)$, represent a graph, where <br/>
  $N$ is a set of nodes and $E \subseteq (N \times N)$ be a set of edges. Graph $G$ is labeled if there exists a function $\lambda$ such that $\lambda : N \rightarrow l$ which assigns a unique label from alphabet $l$ to every node $n \in N$ otherwise $G$ is considered as unlabelled.

- Edges node can also be labeled with a function $\eta : E \rightarrow e$
- Given $G = (N,E,\lambda)$ and $sg = (N_{sg},E_{sg}, \lambda_{sg})$, where $sg$ is a subgraphs of $G$ iff there exists an injective mapping $\mu : N_{sg} \rightarrow N$ such that $(n_1,n_2) \in E_{sg}$ iff $((\mu(n_1),\mu(n_2))) \in E.$
- In this paper context, they specificly refer to a specific subgraphs called **rooted subgraphs**.
- $d$ (number of edges) is the degree around node $n \in N$

## Skipgram word & Document Embedding models

### Skipgram models for word embeddings

- `word2vec` works based on the rationale that _the words appearing in similar contexts tend to have similar meanings and hence have similar vector representaions_.
- To learn a target's representation, this model exploits the notion of context, where a context is defined as a fixed number of words surrounding the target word.
- Given a sequence of words {${w_1,w_2,\dots w_t, \dots, w_T}$}, $w_t$ is the target word that we need to learn its representation. The length of the context window is $c$.
- The objective is to maximize the following likelihood:

$$
\begin{align}
\sum_{t=1}^T log  Pr\left(w_{t-c},\dots, w_{t+c}|w_t\right)
\end{align}
$$

$\text{where } w_{t-c},\dots,w_{t+c} \text{ are the context of the target word } w_t.$
$Pr(w_{t-c},\dots,w_{t+c})$ is computed as:

$$
\begin{align}
\prod_{-c \leq j \leq c, j \neq 0} Pr(w_{t+j}|w_t)
\end{align}
$$

- $Pr(w_{t+j}|w_t)$ is defined as:

$$
\begin{align}
\frac{exp(\vec{w_t} \cdot \vec{w'_{t+j}})}{\sum_{w \in V} exp(\vec{w_t} \cdot \vec{w})}
\end{align}
$$

where $\vec{w}$ and $\vec{w'}$ are the input and output vectors of word $w$ and $V$ is the vocabulary of all the words.

### Negative Sampling

- If a word $w$ appears in the context of another word $w'$, then the vector embedding of $w$ is closer to that of $w'$ compared to any other randomly chosen word from the vocabulary.

## Neural document embedding models

- This model words by considering a word $w_j \in \text{sequence of word } c(d_i)$ to be context of the document $d_i$ and tries to maximize the following log likelihood:

$$
\begin{align}
\sum_{t=1}^T log  Pr\left(w_{j}|d_i\right)
\end{align}
$$

where, the probability $Pr(w_i|d)$ is defined as:

$$
\begin{align}
\frac{exp(\vec{d} \cdot \vec{w'_{j}})}{\sum_{w \in V} exp(\vec{d} \cdot \vec{w})}
\end{align}
$$

## Methods: Learning Graph Representation

- 2 reasons that make rooted subgraphs more easily controlled for learning graph embeddings: 👍

  1. Higher order substructure: It offers richer representation of composition of the graph.
  2. Non-linear substructure: Compared to linear sub-structures such as walks and paths, rooted subgraphs capture the inherent non-linearity in the graphs better.
     - A Walk is a sequence of vertices and edges of a graph i.e. if we traverse a graph then we get a walk. (Closed when starting and ending vertices are identical)
     - A Path is an open walk in which no edge is repeated.

### Algorithm Overview

- To train the skipgram model, we need to extract rooted subgraphs and assign a unique label for all the rooted subgraphs in the vocabulary. After, we deploy Weisfeiler-Lehman (WL) relabeling strategy.

### `graph2vec`: Algorithm

![graph2vec algorithm](/image/graph2vec_algo.png "graph2vec algorithm")

- This algorithm consist of 2 main components:

  1. Procedure to generate _rooted subgraph_ around every node given a graph
  2. Procedure to learn the embedding of the graphs

- The steps are:
  1. Randomly initialize embeddings for all graphs (line 2)
  2. Extract _rooted subgraphs_ in each graphs (line 8)
  3. Iterative learn the corresponding graph's embedding

### Extracting Rooted Subgraph

- WL relabeling process was used to extract rooted subgraphs

![getWLsubGraph algorithm](/image/getWLsubGraph_algo.png "getWLsubGraph algorithm")
