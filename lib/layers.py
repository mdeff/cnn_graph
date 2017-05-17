from . import graph

import numpy as np
import scipy.sparse
import tensorflow as tf


class Layer:
    pass


class Fourier(Layer):
    """Graph convolutional layers that filter in Fourier."""

    def __init__(self, Fout, K):
        self.Fout = Fout
        self.K = K

    def __call__(self, x, L):
        assert K == L.shape[0]  # artificial but useful to compute number of parameters
        N, M, Fin = x.get_shape()
        N, M, Fin = int(N), int(M), int(Fin)
        # Fourier basis
        _, U = graph.fourier(L)
        U = tf.constant(U.T, dtype=tf.float32)
        # Weights
        W = self._weight_variable([M, self.Fout, Fin], regularization=False)
        return self._filter_in_fourier(x, L, self.Fout, self.K, U, W)

    def _filter_in_fourier(self, x, L, Fout, K, U, W):
        # TODO: N x F x M would avoid the permutations
        N, M, Fin = x.get_shape()
        N, M, Fin = int(N), int(M), int(Fin)
        x = tf.transpose(x, perm=[1, 2, 0])  # M x Fin x N
        # Transform to Fourier domain
        x = tf.reshape(x, [M, Fin*N])  # M x Fin*N
        x = tf.matmul(U, x)  # M x Fin*N
        x = tf.reshape(x, [M, Fin, N])  # M x Fin x N
        # Filter
        x = tf.matmul(W, x)  # for each feature
        x = tf.transpose(x)  # N x Fout x M
        x = tf.reshape(x, [N*Fout, M])  # N*Fout x M
        # Transform back to graph domain
        x = tf.matmul(x, U)  # N*Fout x M
        x = tf.reshape(x, [N, Fout, M])  # N x Fout x M
        return tf.transpose(x, perm=[0, 2, 1])  # N x M x Fout


class Spline(Fourier):

    def __call__(self, x, L):
        N, M, Fin = x.get_shape()
        N, M, Fin = int(N), int(M), int(Fin)
        # Fourier basis
        lamb, U = graph.fourier(L)
        U = tf.constant(U.T, dtype=tf.float32)  # M x M
        # Spline basis
        B = self._bspline_basis(self.K, lamb, degree=3)  # M x K
        # B = _bspline_basis(K, len(lamb), degree=3)  # M x K
        B = tf.constant(B, dtype=tf.float32)
        # Weights
        W = self._weight_variable([self.K, self.Fout*Fin], regularization=False)
        W = tf.matmul(B, W)  # M x Fout*Fin
        W = tf.reshape(W, [M, self.Fout, Fin])
        return self._filter_in_fourier(x, L, self.Fout, self.K, U, W)

    def _bspline_basis(self, K, x, degree=3):
        """
        Return the B-spline basis.

        K: number of control points.
        x: evaluation points
           or number of evenly distributed evaluation points.
        degree: degree of the spline. Cubic spline by default.
        """
        if np.isscalar(x):
            x = np.linspace(0, 1, x)

        # Evenly distributed knot vectors.
        kv1 = x.min() * np.ones(degree)
        kv2 = np.linspace(x.min(), x.max(), K-degree+1)
        kv3 = x.max() * np.ones(degree)
        kv = np.concatenate((kv1, kv2, kv3))

        # Cox - DeBoor recursive function to compute one spline over x.
        def cox_deboor(k, d):
            # Test for end conditions, the rectangular degree zero spline.
            if (d == 0):
                return ((x - kv[k] >= 0) & (x - kv[k + 1] < 0)).astype(int)

            denom1 = kv[k + d] - kv[k]
            term1 = 0
            if denom1 > 0:
                term1 = ((x - kv[k]) / denom1) * cox_deboor(k, d - 1)

            denom2 = kv[k + d + 1] - kv[k + 1]
            term2 = 0
            if denom2 > 0:
                term2 = ((-(x - kv[k + d + 1]) / denom2) * cox_deboor(k + 1, d - 1))

            return term1 + term2

        # Compute basis for each point
        basis = np.column_stack([cox_deboor(k, degree) for k in range(K)])
        basis[-1, -1] = 1
        return basis


class Chebyshev(Layer):

    def __init__(self, Fout, K):
        self.Fout = Fout
        self.K = K


class Chebyshev2(Chebyshev):

    def __call__(self, x, L):
        """
        Filtering with Chebyshev interpolation
        Implementation: numpy.

        Data: x of size N x M x F
            N: number of signals
            M: number of vertices
            F: number of features per signal per vertex
        """
        N, M, Fin = x.get_shape()
        N, M, Fin = int(N), int(M), int(Fin)
        # Rescale Laplacian. Copy to not modify the shared L.
        L = scipy.sparse.csr_matrix(L)
        L = graph.rescale_L(L, lmax=2)
        # Transform to Chebyshev basis
        x = tf.transpose(x, perm=[1, 2, 0])  # M x Fin x N
        x = tf.reshape(x, [M, Fin*N])  # M x Fin*N
        def chebyshev(x):
            return graph.chebyshev(L, x, self.K)
        x = tf.py_func(chebyshev, [x], [tf.float32])[0]  # K x M x Fin*N
        x = tf.reshape(x, [self.K, M, Fin, N])  # K x M x Fin x N
        x = tf.transpose(x, perm=[3, 1, 2, 0])  # N x M x Fin x K
        x = tf.reshape(x, [N*M, Fin*self.K])  # N*M x Fin*K
        # Filter: Fin*Fout filters of order K, i.e. one filterbank per feature.
        W = self._weight_variable([Fin*K, self.Fout], regularization=False)
        x = tf.matmul(x, W)  # N*M x Fout
        return tf.reshape(x, [N, M, self.Fout])  # N x M x Fout


def Chebyshev5(Chebyshev):

    def __call__(self, x, L):
        N, M, Fin = x.get_shape()
        N, M, Fin = int(N), int(M), int(Fin)
        # Rescale Laplacian and store as a TF sparse tensor. Copy to not modify the shared L.
        L = scipy.sparse.csr_matrix(L)
        L = graph.rescale_L(L, lmax=2)
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        L = tf.SparseTensor(indices, L.data, L.shape)
        L = tf.sparse_reorder(L)
        # Transform to Chebyshev basis
        x0 = tf.transpose(x, perm=[1, 2, 0])  # M x Fin x N
        x0 = tf.reshape(x0, [M, Fin*N])  # M x Fin*N
        x = tf.expand_dims(x0, 0)  # 1 x M x Fin*N
        def concat(x, x_):
            x_ = tf.expand_dims(x_, 0)  # 1 x M x Fin*N
            return tf.concat([x, x_], axis=0)  # K x M x Fin*N
        if self.K > 1:
            x1 = tf.sparse_tensor_dense_matmul(L, x0)
            x = concat(x, x1)
        for k in range(2, self.K):
            x2 = 2 * tf.sparse_tensor_dense_matmul(L, x1) - x0  # M x Fin*N
            x = concat(x, x2)
            x0, x1 = x1, x2
        x = tf.reshape(x, [self.K, M, Fin, N])  # K x M x Fin x N
        x = tf.transpose(x, perm=[3, 1, 2, 0])  # N x M x Fin x K
        x = tf.reshape(x, [N*M, Fin*self.K])  # N*M x Fin*K
        # Filter: Fin*Fout filters of order K, i.e. one filterbank per feature pair.
        W = self._weight_variable([Fin*self.K, self.Fout], regularization=False)
        x = tf.matmul(x, W)  # N*M x Fout
        return tf.reshape(x, [N, M, self.Fout])  # N x M x Fout


class Bias(Layer):
    pass


class Bias1Relu(Bias):
    """Bias and ReLU. One bias per filter."""
    def __call__(self, x):
        N, M, F = x.get_shape()
        b = self._bias_variable([1, 1, int(F)], regularization=False)
        return tf.nn.relu(x + b)


class Bias2Relu(Bias):
    """Bias and ReLU. One bias per vertex per filter."""
    def __call__(self, x):
        N, M, F = x.get_shape()
        b = self._bias_variable([1, int(M), int(F)], regularization=False)
        return tf.nn.relu(x + b)


class Pooling(Layer):
    def __init__(self, p):
        self.p = p


class MaxPooling(Pooling):
    def __call__(self, x):
        """Max pooling of size p. Should be a power of 2."""
        if self.p > 1:
            x = tf.expand_dims(x, 3)  # N x M x F x 1
            x = tf.nn.max_pool(x, ksize=[1,self.p,1,1], strides=[1,self.p,1,1], padding='SAME')
            #tf.maximum
            return tf.squeeze(x, [3])  # N x M/p x F
        else:
            return x


class AvgPooling(Pooling):
    def __call__(self, x):
        """Average pooling of size p. Should be a power of 2."""
        if self.p > 1:
            x = tf.expand_dims(x, 3)  # N x M x F x 1
            x = tf.nn.avg_pool(x, ksize=[1,self.p,1,1], strides=[1,self.p,1,1], padding='SAME')
            return tf.squeeze(x, [3])  # N x M/p x F
        else:
            return x


class Dense(Layer):

    def __init__(self, Mout, relu=True):
        self.Mout = Mout
        self.relu = relu

    def __call__(self, x):
        """Fully connected layer with Mout features."""
        N, Min = x.get_shape()
        W = self._weight_variable([int(Min), self.Mout], regularization=True)
        b = self._bias_variable([self.Mout], regularization=True)
        x = tf.matmul(x, W) + b
        return tf.nn.relu(x) if self.relu else x
