import numpy as np
import scipy.sparse, scipy.sparse.linalg, scipy.spatial.distance

def grid(m):
    """Return the embedding of a grid graph."""
    M = m**2
    x = np.linspace(0,1,m)
    y = np.linspace(0,1,m)
    xx, yy = np.meshgrid(x, y)
    z = np.empty((M,2))
    z[:,0] = xx.reshape(M)
    z[:,1] = yy.reshape(M)
    return z

def adjacency(z, k=4):
    """Return the adjacency matrix of a kNN graph."""
    M = z.shape[0]

    # Compute pairwise distances.
    d = scipy.spatial.distance.pdist(z, 'euclidean')
    d = scipy.spatial.distance.squareform(d)

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

    # Upper-bound on the spectrum.
    if normalized:
        lmax = 2
    else:
        lmax = scipy.sparse.linalg.eigsh(
                L, k=1, which='LM', return_eigenvectors=False)[0]

    assert np.abs(L - L.T).mean() < 1e-10
    return L, lmax

def fourier(L):
    """Return the Fourier basis, i.e. the EVD of the Laplacian."""

    def sort(lamb, U):
        idx = lamb.argsort()
        return lamb[idx], U[:,idx]

    lamb, U = np.linalg.eig(L.toarray())
    lamb, U = sort(lamb, U)
    return lamb, U
