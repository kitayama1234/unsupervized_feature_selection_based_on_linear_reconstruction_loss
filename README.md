# Unsupervized Feature Selection
Unofficial implementation of the **unsupervised feature selection** algorithm proposed by Ono in March 2020 [1].

## Installation
1. Just add the `fs_ono2020.py` to your directory.
2. Import `fs_ono2020.FeatureSelector` class.

## Usage of `fs_ono2020.FeatureSelector`

### Tutorial
See `demo.ipynb`.

## Algorithm
From the abstruct in the paper:
> In this study, we consider an objective function defined as the reconstruction loss of a linear autoencoder, and this is formulated as a discrete optimization problem that selects the element that minimizes it. Also, we propose a method to solve this problem by sequentially replacing elements chosen so that the objective function becomes smaller.

## Reference
[1] Nobutaka Ono, "Dimension reduction without multiplication in machine learning,"
IEICE Tech. Rep., vol.119, no.440, SIP2019-106, pp.21-26, March 2020. (in Japanese).
URL: https://www.ieice.org/ken/paper/20200302U1Xv/eng/
