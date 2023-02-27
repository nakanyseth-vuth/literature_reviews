# Literature Review

    Paper 2: "What are the biases in my word embedding?" (Swinger et al.)

## Overview

_Unsupervised Bias Enumeration (UBE)_: discovering biases automatically from an unlabeled data representation. The algorithm take a list of target tokens (eg. names) and a word embedding and ouputs a number of _Word Embedding Association Tests (WEATs)_ that capture biases present in the data.

- WEAT is a fairness metric that quantifies the relationship between 2 sets of target words (sets of words intended to denote social groups) and 2 sets of attribute words (eg. attitude, characteristic, trait, occupantional field...)
- WEAT inspired by Implicit Embedding Association (IAT), which is widely used for measuring human biases
- An IAT $\mathcal{T} = (X_1,A_1,X_2,A_2)$ compares two sets of _target tokens_ $X_1$ and $X_2$ such as female vs. male names, and a pair of opposing attribute tokens $A_1$ and $A_2$ such as workplace vs. family-themed words.
- The inputs of WEAT are a sets of tokens $\mathcal{T}$ predefined by researchers
-
