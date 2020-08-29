# Unsupervized Feature Selection
Unofficial implementation of the **unsupervised feature selection** algorithm proposed by Ono in March 2020 [1].

## Installation
1. Just add the `fs_ono2020.py` to your directory.
2. Import `fs_ono2020.FeatureSelector` class.

## Usage of FeatureSelector
See `demo.ipynb`.

#### parameters

>- ```n_features```
>> *int*  
>> Number of features to be selected.
>
>-```random_state```
>> *int (optional, default: None)*  
>> Specify integer if you want reproducible output.
>
>- ```logging```
>> *bool (optional, default: False)*  
>> Specify True if you want to see the progress of instance fitting.
>
>- ```loop_limit```
>> *positive int (optional, default: numpy.inf)*  
>> Specify positive integer if you want the optimization loop to end in the middle.

#### methods

>- ```fit(X)```
>> Fit the `FeatureSelector` instance with your numpy.ndarray dataset `X`.
>
>- ```fit_transform(X)```
>> Fit the `FeatureSelector` instance with your numpy.ndarray dataset `X`, and return new dataset `X_selected` with the selected features.
>
>- ```transform(X)```
>> Return new dataset `X_selected` with the selected features.
>
>- ```reconstruct(X_selected)```
>> Return the reconstructed dataset `X_reconstructed` from the tranceformed dataset `X_selected`.

#### attributes

>- ```selected```
>> The list of selected feature indices
>
>- ```deselected```
>> The list of deselected feature indices
>
>- ```original_dim``` 
>> The original dimension of the input vectors


## Algorithm
From the abstruct in the paper:
> In this study, we consider an objective function defined as the reconstruction loss of a linear autoencoder, and this is formulated as a discrete optimization problem that selects the element that minimizes it. Also, we propose a method to solve this problem by sequentially replacing elements chosen so that the objective function becomes smaller.

## Reference
[1] Nobutaka Ono, "Dimension reduction without multiplication in machine learning,"
IEICE Tech. Rep., vol.119, no.440, SIP2019-106, pp.21-26, March 2020. (in Japanese).
URL: https://www.ieice.org/ken/paper/20200302U1Xv/eng/
