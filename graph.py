import numpy as np
import scipy.sparse, scipy.sparse.linalg, scipy.spatial.distance

dtype = np.float32

def grid(m):
    """Return the embedding of a grid graph."""
    M = m**2
    x = np.linspace(0,1,m, dtype=dtype)
    y = np.linspace(0,1,m, dtype=dtype)
    xx, yy = np.meshgrid(x, y)
    z = np.empty((M,2), dtype)
    z[:,0] = xx.reshape(M)
    z[:,1] = yy.reshape(M)
    return z

#class graph(object):
    #self.L

def adjacency(z, k=4):
    """Return the adjacency matrix of a kNN graph."""
    M = z.shape[0]

    # Compute pairwise distances.
    d = scipy.spatial.distance.pdist(z, 'euclidean')
    d = scipy.spatial.distance.squareform(d)
    d = d.astype(dtype)

    # k-NN graph.
    idx = np.argsort(d)[:,1:k+1]
    d.sort()
    d = d[:,1:k+1]

    # Weights.
    sigma2 = np.mean(d[:,-1])**2
    d = np.exp(- d**2 / sigma2)

    # Weight matrix.
    I = np.arange(0, M).repeat(k)
    J = idx.reshape(M*k)
    V = d.reshape(M*k)
    W = scipy.sparse.coo_matrix((V, (I, J)), shape=(M, M))

    # No self-connections.
    W.setdiag(0)

    # Non-directed graph.
    bigger = W.T > W
    W = W - W.multiply(bigger) + W.T.multiply(bigger)
    assert np.abs(W - W.T).mean() < 1e-10

    # CSR sparse matrix format for efficient multiplications.
    W = W.tocsr()
    W.eliminate_zeros()

    print("{} > {} edges".format(W.nnz, M*k))
    return W

def laplacian(W, normalized=True):
    """Return the Laplacian of the weigth matrix."""

    # Degree matrix.
    d = W.sum(axis=0)

    # Laplacian matrix.
    if not normalized:
        D = scipy.sparse.diags(d.A.squeeze(), 0)
        L = D - W
    else:
        d = 1 / np.sqrt(d)
        D = scipy.sparse.diags(d.A.squeeze(), 0)
        I = scipy.sparse.identity(d.size, dtype=D.dtype)
        L = I - D * W * D

    assert np.abs(L - L.T).mean() < 1e-10
    return L

def lmax(L, normalized=True):
    """Upper-bound on the spectrum."""
    if normalized:
        return 2
    else:
        return scipy.sparse.linalg.eigsh(
                L, k=1, which='LM', return_eigenvectors=False)[0]

def fourier(L, algo='eigh', k=1):
    """Return the Fourier basis, i.e. the EVD of the Laplacian."""

    def sort(lamb, U):
        idx = lamb.argsort()
        return lamb[idx], U[:,idx]

    if algo is 'eig':
        lamb, U = np.linalg.eig(L.toarray())
        lamb, U = sort(lamb, U)
    elif algo is 'eigh':
        lamb, U = np.linalg.eigh(L.toarray())
    elif algo is 'eigs':
        lamb, U = scipy.sparse.linalg.eigs(L, k=k, which='SM')
        lamb, U = sort(lamb, U)
    elif algo is 'eigsh':
        lamb, U = scipy.sparse.linalg.eigsh(L, k=k, which='SM')

    return lamb, U

def lanczos(L, X, K):
    """
    Given the graph Laplacian and a data matrix, return a data matrix which can
    be multiplied by the filter coefficients to filter X using the Lanczos
    polynomial approximation.
    """
    M, N = X.shape

    def basis(L, X, K):
        """
        Lanczos algorithm which computes the orthogonal matrix V and the
        tri-diagonal matrix H.
        """
        a = np.empty((K, N), dtype)
        b = np.zeros((K, N), dtype)
        V = np.empty((K, M, N), dtype)
        V[0,...] = X / np.linalg.norm(X, axis=0)
        for k in range(K-1):
            W = L.dot(V[k,...])
            a[k,:] = np.sum(W * V[k,...], axis=0)
            W = W - a[k,:] * V[k,...] - (b[k,:] * V[k-1,...] if k>0 else 0)
            b[k+1,:] = np.linalg.norm(W, axis=0)
            V[k+1,...] = W / b[k+1,:]
        a[K-1,:] = np.sum(L.dot(V[K-1,...]) * V[K-1,...], axis=0)
        return V, a, b

    def diag_H(a, b, K):
        """Diagonalize the tri-diagonal H matrix."""
        H = np.zeros((K*K, N), dtype)
        H[:K**2:K+1, :] = a
        H[1:(K-1)*K:K+1, :] = b[1:,:]
        H.shape = (K, K, N)
        Q = np.linalg.eigh(H.T, UPLO='L')[1]
        Q = np.swapaxes(Q,1,2).T
        return Q

    V, a, b = basis(L, X, K)
    Q = diag_H(a, b, K)
    Xt = np.empty((K, M, N), dtype)
    for n in range(N):
        Xt[...,n] = Q[...,n].T @ V[...,n]
    Xt *= Q[0,:,np.newaxis,:]
    Xt *= np.linalg.norm(X, axis=0)
    return Xt
#    return Xt, Q[0,...]

def rescale_L(L, lmax=2):
    """Rescale the Laplacian eigenvalues in [-1,1]."""
    M, M = L.shape
    I = scipy.sparse.identity(M, format='csr', dtype=dtype)
    L /= lmax * 2
    L -= I
    return L

def chebyshev(L, X, K):
    """Return T_k X where T_k are the Chebyshev polynomials of order up to K.
    Complexity is O(KMN)."""
    M, N = X.shape

#    L = rescale_L(L, lmax)
    # Xt = T @ X: MxM @ MxN.
    Xt = np.empty((K, M, N), dtype)
    # Xt_0 = T_0 X = I X = X.
    Xt[0,...] = X
    # Xt_1 = T_1 X = L X.
    if K > 1:
        Xt[1,...] = L.dot(X)
    # Xt_k = 2 L Xt_k-1 - Xt_k-2.
    for k in range(2, K):
        Xt[k,...] = 2 * L.dot(Xt[k-1,...]) - Xt[k-2,...]
    return Xt
