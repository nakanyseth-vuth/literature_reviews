# Literature Review

    Paper 4: "Evaluating Word Embedding Models: Methods and Experimental Results" (Wang et al.)

## Introduction

- There are 2 types of evaluators:

  1. Intrinsic evaluator: test the quality of a representation independent of specific NLP tasks.
  2. Extrinsic evaluator: use word embeddings as input features to a downstream task and measure changes in performance metrics specific to that task.

- 6 embedding models were evaluated:
  1. Skip-gram negative sampling (SGNS)
  2. CBOW
  3. GloVe
  4. FastText
  5. ngram2vec
  6. Dict2vec

## Desired properties of Embedding models and Evaluators

### Embedding models

- **Non-conflation**: Different local contexts around a word should give rise to specific properties of the word e.g., the plural or singular form, the tenses, etc. Embedding models should be able to discern differences in the contexts and encode these details into a meaningful representation in the word subspace.
- **Robustness Against Lexical Ambiguity**: All senses of a word should be represented. Model should be able to discern the sense of a word from its context and find appropriate embedding.
- **Demonstration of Multifacetedness**: The facet, phonetic, morphological, syntactic, and other properties, of a word should contribute to its final representation. For example, a representation of a word should change when its tense is changed or a prefix is added.
- **Reliability**: Even if a model creates different representations from the same dataset because of random initialization, the performance of various representations should score consistently.
- **Good Geometry**: The geometry of an embedding space should have good spread. Word models should overcome the difficulty of arising from inconsistent frequency of word usage and derive some meaning from word frequency.

### Evaluator

- **Good testing data**: testing data should be varied with a good spread in the span of word space. Frequently and rarely occurring words should be included in the evaluation.
- **Comprehensiveness**: Ideally, an evaluator should test for many properties of a word embedding model.
- **High correlation**: The score of a word model in an _intrinsic evaluation_ task should correlate well with the performance of the ones in downstream NLP tasks (extrinsic)
- **Efficiency**: Evaluator should be computational efficient. It should be simple yet able to predict the downstream performance of a model.
- **Statistical Significant**: The performance of different word embedding models wrt to an evaluator should have enough statistical significance, or enough variance between score distribution, to be differentiated.

## Intrinsic Evaluators

They measure syntactic or semantic relationships between word directly.

### 1. Word Similarity

- The word similarity evaluator correlates the distance between word vectors and human perceived semantic similarity. The way distributional semantic models simulate similarity is still ambiguous.
- One commonly used evaluator is the cosine similarity:

$$
  \begin{align}
    cos(w_x, w_y) = \frac{w_x \cdot w_y}{\lVert w_x \rVert \lVert w_y \rVert}
  \end{align}
$$

- where $w_x$ and $w_y$ are 2 word vectors and $\lVert w_x \rVert$ and $\lVert w_y \rVert$ are the $\mathbb{l_2}$ norm ($l_2$ → calculates the distance of the vector coordinate from the origin of the vector space)
- This on the other hand has several problems. This test is aimed at finding distributional similarity among pairs of words, but this is often conflated with morphological relations (play, playground) and simple collocations (social media, machine learning). Similarity maybe confused with relatedness.
- Example: _car_ and _train_ are two similar words, while _car_ and _road_ are two related words. The correlation between score from intrinsic test and other extrinsic test could be low in some cases.
- This might not be comprehensive.

### 2. Word Analogy

- When given a pair of words $a$ and $a^*$ and a third word $b$, the analogy relationship between $a$ and $a^*$ can be used to find the corresponding word $b$ to $b^*$. One example could be:
- write : writing :: read : <u>reading</u>.
- The _3CosAdd_ method solves for $b^*$ using the following equation:

$$
  \begin{align}
    b^* = \underset{b^{'}}{\operatorname{argmax}}(cos(b^{'}, a^* - a + b))
  \end{align}
$$

- Thus, high cosine similarity means that vectors share a similar director. However, it is important to note that _3CosAdd_ method normalizes vector lengths using cosine similarity. **Since cosine is inverse to the angle, high cosine similarity (close to 1) means the vectors share a very similar direction. Note that this metric normalizes (and thus ignore) the vector's lengths, unlike Euclidean distance between them.**
- Alternatively, there is the _3CosMul_ method, which defined as:

$$
  \begin{align}
    b^* = \underset{b^{'}}{\operatorname{argmax}} \frac{cos(b^{'},b)cos(b^{'}, a^*)}{cos(b^{'}, a)+\epsilon}
  \end{align}
$$

- where $\epsilon = 0.001$ us used to prevent division by zero.
- With _3CosMul_, small difference are enlarged while large ones are suppressed.
- Many models score under 30% on analogy tests, suggesting that not all relations can be identified in this way.
- It is mentioned that the analogy test is most successful when all the three source vectors are relatively close to the target vector. Accuracy of this test decreases as their distance increases.
- Another seemingly counter-intuitive finding is that words with denser neighborhoods yields higher accuracy.

- The analogy test is subjective. The dataset on which current models are trained does not encode our human sense and reasoning. Thus, given a word pair, the vector space model may find a different relationship from what human may find
