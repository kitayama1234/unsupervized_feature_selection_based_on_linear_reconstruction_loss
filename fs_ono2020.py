r"""
Author: Masaki Kitayama


Reference:

Nobutaka Ono,
"Dimension reduction without multiplication in machine learning,"
IEICE Tech. Rep., vol.119, no.440, SIP2019-106, pp.21-26, March 2020. (in Japanese).
URL:https://www.ieice.org/ken/paper/20200302U1Xv/eng/

"""

import numpy as np
from sklearn.linear_model import Ridge
from typing import Union


class FeatureSelector:
    def __init__(self, n_features: int,
                 random_state: Union[int, None] = None,
                 logging: bool = False,
                 loop_limit=np.inf):
        # parameters
        self.n_features   = n_features    # (int) Number of features to be selected
        self.random_state = random_state  # [optional] (int) Specify this if you want reproducible output.
        self.logging      = logging       # [optional] (bool) Specify this if you want to see the progress of instance fitting.
        self.loop_limit   = loop_limit    # [optional] (positive int) Specify this if you want the optimization loop to end in the middle.

        # attributes
        self.is_fitted     = False
        self.original_dim  = None
        self.selected      = None      # list of selected feature indices
        self.deselected    = None      # list of deselected feature indices
        self.reconstructor = Ridge(alpha=0., fit_intercept=True, random_state=random_state)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        if self.logging:
            print("Starting to fit Feature Selector.")

        _check_input(X)
        self.original_dim = X.shape[1]

        # Initial covariance matrix
        initial_cov_matrix = self._cov_matrix(X)

        # Initialize the list of selected/deselected feature indices randomly.
        np.random.seed(self.random_state)
        all_features = np.random.permutation(np.arange(0, self.original_dim))
        self.selected = all_features[:self.n_features]
        self.deselected = all_features[self.n_features:]

        # Initialize the objective function.
        obj_func = ObjectiveFunction(self.original_dim, self.n_features)
        obj_func.update(initial_cov_matrix, self.selected, self.deselected)
        obj_value = obj_func.get_obj_value()
        if self.logging:
            print("Objective function initialized: %f" % obj_value)

        # flags for the optimization loop
        is_swapped = np.zeros(self.n_features, dtype=np.int)
        is_swapped[:] = 1
        obj_diff = np.zeros(self.original_dim - self.n_features)

        # optimization loop
        loop_cnt = 0
        num_swapped = is_swapped.sum()
        while (num_swapped > 0) and (loop_cnt < self.loop_limit):
            loop_cnt += 1
            if self.logging:
                print("starting loop %d ... " % loop_cnt, end="")

            is_swapped[:] = 0
            for idx_selected in range(self.n_features):

                obj_diff[:] = 0.
                for idx_deselected in range(self.original_dim - self.n_features):
                    # In this loop, we evaluate all idx_deselected and swap operations for the current idx_selected.
                    # Calculate the differences of the objective function in the cases where the features are swapped.
                    obj_diff[idx_deselected] = obj_func.get_obj_diff(idx_selected, idx_deselected)

                # Adopt and execute the swap operation with  maximum obj_diff in the above loop.
                obj_diff_max = obj_diff.max()
                if obj_diff_max > 0:
                    obj_value = obj_value + obj_diff_max
                    self._swap(idx_selected, obj_diff.argmax())
                    obj_func.update(initial_cov_matrix, self.selected, self.deselected)
                    is_swapped[idx_selected] = 1

            num_swapped = is_swapped.sum()
            if self.logging:
                print("completed. (%d features swapped, obj func: %f)" % (num_swapped, obj_value))

        # select features
        X_selected = X[:, self.selected]

        # A regression model for reconstruction is trained by datasets (least squares method).
        self.reconstructor.fit(X_selected, X)

        self.is_fitted = True
        if self.logging:
            print("Feature Selector fitted.")

        return X_selected

    def fit(self, X: np.ndarray) -> None:
        _ = self.fit_transform(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise InstanceNotFittedError
        _check_input(X, self.original_dim)
        return X[:, self.selected]

    def reconstruct(self, X_selected: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise InstanceNotFittedError
        _check_input(X_selected, self.n_features)
        X_reconstructed = self.reconstructor.predict(X_selected)
        return X_reconstructed

    def _swap(self, idx_selected, idx_deselected):
        selected_temp = self.selected[idx_selected]
        self.selected[idx_selected] = self.deselected[idx_deselected]
        self.deselected[idx_deselected] = selected_temp

    def _cov_matrix(self, X):
        X_centered = X - X.mean(axis=0)
        cov_matrix = X_centered.transpose() @ X_centered
        # Add a small unit matrix to ensure numerical stability.
        cov_matrix += np.eye(self.original_dim) * abs(np.linalg.eig(cov_matrix)[0][0]) * 1e-6
        return cov_matrix


class ObjectiveFunction:
    def __init__(self, original_dim, n_features):
        self.n_features = n_features
        self.diff_dim = original_dim - n_features

        self.indices = np.zeros(original_dim, dtype=np.int)
        self.cov_matrix = np.zeros([original_dim, original_dim])
        self.mat_A = np.zeros([self.n_features, self.n_features])
        self.mat_B = np.zeros([self.n_features, self.diff_dim])
        self.mat_W = np.zeros([self.n_features, self.n_features])
        self.mat_F = np.zeros([self.n_features, 2])
        self.mat_H = np.zeros([self.n_features, 2])
        self.mat_L = np.zeros([self.diff_dim, 2])
        self.mat_X = np.array([[0., 1.], [1., 0.]])   # Eq.(28)
        self.mat_G = np.zeros([2, 2])
        self.mat_R = np.zeros([self.n_features, 2])
        self.mat_S = np.zeros([self.diff_dim, 2])
        self.mat_T = np.zeros([self.n_features, 2])
        self.mat_U = np.zeros([self.n_features, 2])
        self.temp = np.zeros([2, 2])

    def update(self, initial_cov_matrix, selected, deselected):
        self.indices[:self.n_features] = selected
        self.indices[self.n_features:] = deselected
        self.cov_matrix[:, :] = initial_cov_matrix[:, self.indices][self.indices, :]
        self.mat_A[:, :] = initial_cov_matrix[:, selected][selected, :]
        self.mat_B[:, :] = initial_cov_matrix[:, deselected][selected, :]
        self.mat_W[:, :] = np.linalg.inv(self.mat_A)

    def get_obj_value(self):
        return np.trace(self.mat_A) + np.trace(self.mat_B.transpose() @ self.mat_W @ self.mat_B)

    def get_obj_diff(self, idx_selected, idx_deselected):
        idx_i = idx_selected
        idx_j = idx_deselected + self.n_features

        # Eq.(22),(25)
        self.mat_F[:, :] = 0.
        self.mat_F[idx_i, 0] = 1.
        self.mat_F[:, 1] = self.cov_matrix[:self.n_features, idx_j] - self.cov_matrix[:self.n_features, idx_i]
        self.mat_F[idx_i, 1] = (self.cov_matrix[idx_j, idx_j] - self.cov_matrix[idx_i, idx_i]) / 2.

        # Eq.(23),(26)
        self.mat_H[:, :] = 0.
        self.mat_H[idx_i, 0] = 1.
        self.mat_H[:, 1] = self.cov_matrix[:self.n_features, idx_i] - self.cov_matrix[:self.n_features, idx_j]
        self.mat_H[idx_i, 1] = -(self.cov_matrix[idx_j, idx_j] - self.cov_matrix[idx_i, idx_i]) / 2.

        # Eq.(24),(27)
        self.mat_L[:, :] = 0.
        self.mat_L[idx_deselected, 1] = 1.
        self.mat_L[:, 0] = self.cov_matrix[self.n_features:, idx_j] - self.cov_matrix[self.n_features:, idx_i]
        self.mat_L[idx_deselected, 0] = (self.cov_matrix[idx_j, idx_j] - self.cov_matrix[idx_i, idx_i]) / 2.

        # Eq.(32)
        self.mat_G[:, :] = -(self.mat_X + self.mat_F.transpose() @ self.mat_W @ self.mat_F)

        # Eq.(35)~(38)
        self.mat_R[:, :] = self.mat_W @ self.mat_F
        self.mat_S[:, :] = self.mat_B.transpose() @ self.mat_R
        self.mat_T[:, :] = self.mat_B @ self.mat_L
        self.mat_U[:, :] = self.mat_W @ self.mat_H

        # Eq.(39)
        self.temp[:, :] = 2 * (self.mat_T.transpose() @ self.mat_R) + \
                          (self.mat_L.transpose() @ self.mat_L) @ (self.mat_H.transpose() @ self.mat_R)
        obj_diff = self.cov_matrix[idx_j, idx_j] - self.cov_matrix[idx_i, idx_i] + \
                   np.trace(
                       2 * self.mat_T.transpose() @
                       self.mat_U + (self.mat_H.transpose() @ self.mat_U) @
                       (self.mat_L.transpose() @ self.mat_L)
                   ) + \
                   np.trace(
                       np.linalg.inv(self.mat_G) @
                       (
                               self.mat_S.transpose() @ self.mat_S +
                               (self.mat_F.transpose() @ self.mat_U) @ self.temp
                       )
                   )

        return obj_diff


def _check_input(X, correct_vec_dim=None):
    if type(X) is not np.ndarray:
        raise NotNumpyNdarrayError
    array_shape = X.shape
    array_dim = len(array_shape)
    if array_dim != 2:
        raise WrongArrayDimError(array_dim)
    vec_dim = array_shape[1]
    if correct_vec_dim is not None and vec_dim != correct_vec_dim:
        raise WrongVectorDimError(vec_dim, correct_vec_dim)


class NotNumpyNdarrayError(ValueError):
    def __init__(self):
        super(NotNumpyNdarrayError, self).__init__("Input must be numpy.ndarray")


class WrongArrayDimError(ValueError):
    def __init__(self, array_dim):
        message = "Input array dimension is wrong (found %d, must be 2)." % array_dim
        super(WrongArrayDimError, self).__init__(message)


class WrongVectorDimError(ValueError):
    def __init__(self, vec_dim, correct_vec_dim):
        message = "Input vector dimension is wrong (found %d, must be %d)." % (vec_dim, correct_vec_dim)
        super(WrongVectorDimError, self).__init__(message)


class InstanceNotFittedError(Exception):
    def __init__(self):
        message = "This Feature Selector instance is not fitted yet. " \
                  "Call 'fit' or 'fit_transform' with your dataset 'X'."
        super(InstanceNotFittedError, self).__init__(message)




