# -*- coding: utf-8 -*-
"""
Author Esa Ollila

If you use this codes then please cite at least one of the following: 
      [1] Esa Ollila, "Matching pursuit covariance learning", EUSIPCO-2024, 
      Lyon, August 26-30, 2024. 
      |2] Esa Ollila, "Sparse signal recovery and source localization via 
      covariance learning," arXiv:2401.13975 [stat.ME], Jan 2024. 
      https://arxiv.org/abs/2401.13975
      
"""

import numpy as np
from scipy.linalg import solve 

def CLOMP(A, Y, K):
    """
    Estimating the (sparse) signal support using the greedy covariance 
    learning method (CL-OMP) described in [1,2].
    
    INPUT: 
        A       - Dictionary of size N x M
        Y       - Matrix of N x L (L is the number of measurement vectors)
        K       - number of non-zero sources

    OUTPUT:
        Ilocs  -  support of non-zeros signal powers (K-vector)
        gam0   -  Kx1 vector containing non-zero source signal powers
        sigc   -  noise variance estimate (positive scalar)

    REFERENCE:
        
        |1] Esa Ollila, "Sparse signal recovery and source localization via 
        covariance learning," arXiv:2401.13975 [stat.ME], Jan 2024. 
        https://arxiv.org/abs/2401.13975
        
        [2] Esa Ollila, "Matching pursuit covariance learning", EUSIPCO-2024, 
        Lyon, August 26-30, 2024. 
        

    Author: Esa Ollila, Aalto University
    """
    N, M = A.shape  # Number of sensors and dictionary entries
    _, L = Y.shape  # Number of snapshots in the data covariance

    # Assert the validity of K
    assert isinstance(K, int) and K <= M and K > 0, "'K' must be a positive integer smaller than M"

    RY = (1 / L) * Y @ Y.conj().T  # Data covariance matrix

    # Initialize
    sigc = np.real(np.trace(RY) / N)  # Noise estimate when gamma = 0
    SigmaYinv = (1 / sigc) * np.eye(N)
    SigmaY = sigc * np.eye(N)

    Ilocs = np.zeros(K, dtype=int)

    for k in range(K):
        # 1. Go through basis vectors
        B = SigmaYinv @ A
        gamma_num = np.maximum(0, np.real(np.sum(np.conj(B) * ((RY - SigmaY) @ B), axis=0)))
        gamma_denum = np.maximum(0, np.real(np.sum(np.conj(A) * B, axis=0)))
        gamma_denum[gamma_denum <= 1e-12] = 1e-12  # Avoid division by zero
        gamma = gamma_num / (gamma_denum ** 2)
        tmp = gamma * gamma_denum
        fk = np.log(1 + tmp) - tmp

        # 2. Update the index set
        indx = np.argmin(fk)
        if k > 0 and len(set(Ilocs[:k]) - {indx}) != k:
            tmp = list(set(range(M)) - set(Ilocs[:k]))
            indx = tmp[np.argmin(fk[tmp])]

        Ilocs[k] = indx

        # 3. Update gamma0 and noise power sigma^2
        Am = A[:, Ilocs[:k + 1]]
        Am_plus = np.linalg.pinv(Am)
        P_N = np.eye(N) - Am @ Am_plus
        al = np.real(np.trace(P_N @ RY))
        sigc = al / (N - k - 1)
        if k == 0:
            gam0 = gamma[Ilocs[0]].flatten() * (N / (N - 1))
        else:
            gam0 = np.real(np.diag(Am_plus @ (RY - sigc * np.eye(N)) @ Am_plus.conj().T))
            if not np.all(gam0 > 0):
                Gamhat, gam0, sigc = SML_MLE(Am, RY, al)

        # 4. Update the covariance matrix
        SigmaY = sigc * np.eye(N) + Am @ np.diag(gam0) @ Am.conj().T
        SigmaYinv = solve(SigmaY, np.eye(N),assume_a='pos')  

    # Sort Ilocs and gam0 by index
    sort_idx = np.argsort(Ilocs)
    Ilocs = Ilocs[sort_idx]
    gam0 = gam0[sort_idx]

    return Ilocs, gam0, sigc


def SML_MLE(A, RY, al):
    """
    If not all powers are positive then find the signal powers
    using the method described in Bresler (1988)
    
    Parameters:
    A -- (N, K) matrix of selected dictionary atoms
    RY -- (N, N) data covariance matrix
    al -- scaled noise estimate ( = trace( (I-A*pinv(A))*RY) )
    
    Returns:
    Gamhat -- estimated covariance matrix of the signal
    gamhat -- signal power estimates
    sigmahat -- estimated noise power
    
    Author: Esa Ollila, Aalto University 
    """
    N, K = A.shape
    KtoZero = np.arange(K, -1, -1)

    # 1. Compute the Cholesky factor:
    L = np.linalg.cholesky(A.conj().T @ A)
    Linv = solve(L, np.eye(K)) 

    # 2. Compute X and its eigenvalues
    X = Linv @ (A.conj().T @ RY @ A) @ Linv.conj().T
    Phi, G = np.linalg.eigh(X)
    Phi = np.real(Phi)
    ind = np.argsort(Phi)
    Phi = Phi[ind]
    G = G[:, ind]

    # 3. Evaluate alpha = trace(P_A^orth R_Y)
    # al = real(trace(P_n*RY));

    # 4. Compute sigma^2-s
    sigmasq = (al + np.cumsum(np.concatenate(([0], Phi)))) / (N - KtoZero)
    ind1 = sigmasq >= np.concatenate(([0], Phi))
    ind2 = sigmasq <= np.concatenate((Phi, [np.inf]))
    indvec = ind1 & ind2
    ii = np.where(indvec)[0][0]

    # sigma^2 estimate
    k = KtoZero[ii]
    sigmahat = sigmasq[ii]
    Phi = Phi[::-1]
    G = G[:, ::-1]
    psi = Phi - sigmahat
    psi[k:] = 0
    Gamhat = Linv.conj().T @ (G @ np.diag(psi) @ G.conj().T) @ Linv
    gamhat = np.real(np.diag(Gamhat))

    return Gamhat, gamhat, sigmahat

def SOMP(Y, A, K):
    """
    Simultaneous Orthogonal Matching Pursuit algorithm (SOMP).
    
    Parameters:
    Y -- (N, L) matrix of observations
    A -- (N, M) dictionary matrix
    K -- sparsity level
    
    Returns:
    X -- (K, L) sparse signal estimate
    S -- (K,) indices of selected atoms
    
    Author: Esa Ollila, Aalto University
    """
    _, L = Y.shape
    X = np.zeros((K, L), dtype=complex)
    R = Y.copy()
    S = np.zeros(K, dtype=int)

    for k in range(K):
        E = np.sum(np.abs(A.T.conj() @ R), axis=1)
        pos = np.argmax(E)
        S[k] = pos
        V = A[:, S[:k + 1]]
        X[:k + 1, :] = np.linalg.pinv(V) @ Y
        R = Y - V @ X[:k + 1, :]

    return X, S
