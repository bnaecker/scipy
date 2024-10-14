"""Generalized singular value decomposition"""

import numpy as np

# Local imports.
from ._misc import LinAlgError
from .lapack import get_lapack_funcs, _compute_lwork


def gsvd(a, b, compute_u=True, compute_v=True, compute_x=True, full_matrices=True):
    """Generalized singular value decomposition (GSVD) of two matrices.

    Factors a pair of matrices ``a`` and ``b`` into unitary matrices ``U``, ``V``,
    and ``X``, and two 1-dimensional arrays ``c`` and ``s``, such that:

    .. math::

        a == U @ C @ X.H

        b == V @ S @ X.H

    The matrices ``U`` and ``V`` contain the left generalized singular vectors
    of ``a`` and ``b``, respectively. The matrix ``X`` contains the right
    generalized singular vectors of both ``a`` and ``b``.

    Parameters
    ----------
    a : (M, N) array_like
        First array to decompose.
    b : (P, N) array_like
        Second array to decompose.
    compute_u, compute_v, compute_x : bool, optional
        If ``True``, return the corresponding array, which contain the left
        generalized singular vectors of ``a`` and ``b`` and the right generalized
        singular vectors of both, respectively.
    full_matrices : bool, optional
        If ``True``, return all arrays as "full-sized", 2-dimensional arrays
        that can be directly multiplied together using the equations in the
        above summary. If ``False``, then the matrices will all be limited to a
        number of columns equal to the effective numerical rank of ``(A.H,
        B.H).H``.

        **IMPORTANT** Please note that this argument has different semantics
        than that of the same name in ``scip.linalg.svd``. The full matrix is
        always *computed* (e.g., an M-by-M matrix ``U``), but only the first
        ``r`` rows are *returned*. In other words, there is no "economy
        decomposition", as the underlying LAPACK routine always computes the
        full square matrices. This can lead to large memory consumption if the
        input matrices ``a`` or ``b`` have many rows. Use the arguments
        ``compute_*`` to avoid computing the corresponding matrices entirely, if
        they're not needed.

    Returns
    -------
    U : ndarray
        Unitary array with the left generalized singular vectors of ``a``. Only
        returned if ``compute_u`` is ``True``. If ``full_matrices`` is ``True``,
        this will have shape ``(M, M)``. If ``full_matrices`` is ``False``, this
        may have fewer than ``M`` columns.
    V : ndarray
        Unitary array with the left generalized singular vectors of ``b``. Only
        returned if ``compute_v`` is ``True``. If ``full_matrices`` is ``True``,
        this will have shape ``(P, P)``. If ``full_matrices`` is ``False``, this
        may have fewer than ``P`` columns.
    X : ndarray
        Unitary array with the right generalized singular vectors of both ``a``
        and ``b``. Only returned if ``compute_x`` is ``True``. If
        ``full_matrices`` is ``True``, this will have shape ``(N, N)``. If
        ``full_matrices`` is ``False``, this may have fewer than ``N`` columns.
    c, s : ndarray
        Arrays containing parts of the generalized singular value pairs of ``a``
        and ``b``. If ``full_matrices`` is ``True``, then these will have shape
        ``(N, N)``. If ``full_matrices`` is ``False``, then these will have
        shape ``(r,)``, where ``r`` is the effective numerical rank of ``(A.H,
        B.H).H``. In the case of full-rank inputs ``a`` and ``b``, ``r`` will
        generally be equal to ``N``, the number of columns in both ``a`` and
        ``b``. The generalized singular values can be computed as ``c / s``.

    Raises
    ------
    LinAlgError
        If the GSVD routine does not converge or on an internal error.
    ValueError
        If the input arrays have more than 2 dimensions, do not have the same
        number of columns, or if either array has zero elements.

    Examples
    --------
    >>> import numpy as np
    >>> import scipy as sp
    >>> a = np.random.rand(3, 3)
    >>> b = np.random.rand(3, 3)
    >>> u, v, x, c, s = sp.linalg.gsvd(a, b)
    >>> np.allclose(u.dot(np.diag(c)).dot(x.T), a)
    True
    >>> np.allclose(v.dot(np.diag(s)).dot(x.T), b)
    True

    For complex matrices, the reconstruction is almost identical, except that
    the conjugate-transpose is used:

    >>> import numpy as np
    >>> import scipy as sp
    >>> a = np.random.rand(3, 3) + 1j * np.random.rand(3, 3)
    >>> b = np.random.rand(3, 3) + 1j * np.random.rand(3, 3)
    >>> u, v, x, c, s = sp.linalg.gsvd(a, b)
    >>> np.allclose(u.dot(np.diag(c)).dot(x.conj().T), a)
    True
    >>> np.allclose(v.dot(np.diag(s)).dot(x.conj().T), b)
    True

    The GSVD applies to any pair of matrices with the same number of columns.
    The default value of `full_matrices=True` ensures that they can always be
    multiplied together to reconstruct the inputs. But if that is set to false,
    the outputs may have incompatible shapes.

    >>> import numpy as np
    >>> import scipy as sp
    >>> a = np.random.rand(4, 3)
    >>> b = np.random.rand(5, 3)
    >>> u, v, x, c, s = sp.linalg.gsvd(a, b, full_matrices=False)
    >>> u.shape
    (4, 3)
    >>> v.shape
    (5, 3)
    >>> x.shape
    (3, 3)
    >>> c.shape
    (3,)
    >>> s.shape
    (3,)

    Notes
    -----

    The GSVD is a generalization of the common SVD to a pair of matrices. It can
    be used to derive several other kinds of decompositions, including the SVD
    and cosine-sine (CS) decomposition. It is also useful in many kinds of
    regularization, optimization, and dimensionality reduction problems.

    .. versionadded:: 1.15.0

    References
    ----------
    [1] https://en.wikipedia.org/wiki/Generalized_singular_value_decomposition

    [2] Golub, G., and C.F. Van Loan, 2013, Matrix Computations, 4th Ed.

    """
    # Sanity check arguments, and copy input arrays as LAPACK overwrites them
    # with the results on completion.
    if a.size == 0 or b.size == 0:
        raise ValueError("Input arrays must not be empty")
    if a.ndim > 2 or b.ndim > 2:
        raise ValueError("Input arrays must be no more than 2D")
    dtype = np.common_type(a, b)
    Ac = np.array(a, copy=True, dtype=dtype, order="C", ndmin=2)
    Bc = np.array(b, copy=True, dtype=dtype, order="C", ndmin=2)
    m, n = Ac.shape
    p = Bc.shape[0]
    if Bc.shape[1] != n:
        raise ValueError("Input arrays must have the same number of columns")

    # Call LAPACK driver with optimal work size
    ggsvd, ggsvd_lwork = get_lapack_funcs(("ggsvd3", "ggsvd3_lwork"), dtype=dtype)
    lwork = _compute_lwork(ggsvd_lwork, Ac, Bc)
    K, L, Ac, Bc, alpha, beta, U, V, Q, iwork, info = ggsvd(
        Ac, Bc, lwork, compute_u, compute_v, compute_x
    )
    if info != 0:
        raise LinAlgError(f"ggsvd LAPACK routine failed, info = {info}")

    # Compute the returned matrix X if needed
    rank = K + L
    if compute_x:
        R_full = np.eye(n, dtype=dtype)
        R_full[n - rank :, n - rank :] = _extract_embedded_r(Ac, Bc, K, L).conj().T
        X = Q.dot(R_full)

    # Sort by decreasing `alpha`, which will ensure that the generalized
    # singular values in `c / s` are descreasing, matching the convention of the
    # traditional SVD routine.
    if m - rank >= 0:
        ix = np.argsort(alpha[K:rank])[::-1]
        if compute_u:
            U[:, K:rank] = U[:, K:rank][:, ix]
        if compute_v:
            V[:, :L] = V[:, :L][:, ix]
        if compute_x:
            X[:, -L:] = X[:, -L:][:, ix]
        alpha[K:rank] = alpha[K:rank][ix]
        beta[K:rank] = beta[K:rank][ix]
    else:
        ix = np.argsort(alpha[K:m])[::-1]
        if compute_u:
            U[:, K:] = U[:, K:][:, ix]
        if compute_v:
            V[:, : m - K] = V[:, : m - K][:, ix]
        if compute_x:
            X[:, n - L : n + m - rank] = X[:, n - L : n + m - rank][:, ix]
        alpha[K:m] = alpha[K:m][ix]
        beta[K:m] = beta[K:m][ix]

    # Move singular vectors to the diagonal where possible. This can't be done in
    # all cases, but it's covenient and matches the semantics of the
    # corresponding MATLAB routine (though the ordering of the singular values
    # is reversed).
    if n - rank > 0 and compute_x:
        X = np.roll(X, rank - n, axis=1)
    if K > 0 and p >= rank and compute_v:
        V = np.roll(V, K, axis=1)

    # Possibly expand alpha / beta into full arrays, or just return them as
    # compressed 1D arrays.
    if full_matrices:
        c = np.zeros((m, rank), dtype=dtype)
        s = np.zeros((p, rank), dtype=dtype)
        if m - rank >= 0:
            np.fill_diagonal(c[:K, :K], 1)
            np.fill_diagonal(c[K:, K:], alpha[K:rank])
            np.fill_diagonal(s[:L, K:], beta[K:rank])
        else:
            np.fill_diagonal(c[:K, :K], 1)
            np.fill_diagonal(c[K:m, K:m], alpha[K:m])
            np.fill_diagonal(s[: m - K, K:m], beta[K:m])
            np.fill_diagonal(s[m - K : L, m:], 1)
    else:
        if compute_u and m > rank:
            U = U[:, :rank]
        if compute_v and p > rank:
            V = V[:, :rank]
        if compute_x and n > rank:
            X = X[:, :rank]
        c = alpha[:rank]
        s = beta[:rank]

    outputs = tuple(
        (arr for arr, ret in zip((U, V, X), (compute_u, compute_v, compute_x)) if ret)
    )
    return outputs + (c, s)


def _extract_embedded_r(Ac, Bc, K, L):
    """Extract the matrix R embedded in arrays returned from GSVD LAPACK
    drivers.

    See https://netlib.org/lapack/explore-html/d1/d27/group__ggsvd3_ga2ea1a2c8351a881c0d1571c4f7da33fc.html#ga2ea1a2c8351a881c0d1571c4f7da33fc
    for a description of how R is embedded in these arrays, depending on their
    shapes and the effective rank of ``np.vstack((Ac.T, Bc.T)).T``.
    """
    m, n = Ac.shape
    rank = K + L
    if m - rank >= 0:
        return Ac[:rank, n - rank :]
    R = np.zeros((rank, rank), dtype=Ac.dtype)
    R[:m, :] = Ac[:, n - rank :]
    R[m:, m:] = Bc[m - K : L, n + m - rank :]
    return R
