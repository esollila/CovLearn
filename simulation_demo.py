#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The code reproduces the simulation study in the reference: 
    [1] Esa Ollila, "Matching pursuit covariance learning", EUSIPCO-2024, 
    Lyon, August 26-30, 2024. 
    
Simulation is SNR w.r.t. N (dimension)

@author: Esa Ollila
"""

import numpy as np
import matplotlib.pyplot as plt
from CLOMPbase import CLOMP, SOMP
import time

def GaussianMM(m, n):
    """
    Generates a Gaussian measurement matrix of size m x n.
    The columns are normalized to unit norm.
    """
    A = (1/2) * (np.random.randn(m, n) + 1j * np.random.randn(m, n))
    len = np.sqrt(np.real(np.sum(A*A.conj(),axis=0)))
    A = A @ np.diag(1.0 / len)  # Normalize to unit-norm columns
    return A

# Simulation parameters
M = 2**8  # 256 (number of atoms)
K = 2**2  # 4 (sparsity level)
L = 2**4  # 16 (number of snapshots)
dims = [2**3, 2**4, 2**5, 2**6]  # [8, 16, 32, 64]

SNR = np.arange(1, 11)
nSNR = len(SNR)
sigmas = np.zeros((nSNR, K))

sigmas[:, 0] = 10.0**(SNR / 10.0)
sigmas[:, 1] = 10.0**((SNR - 1) / 10.0)
sigmas[:, 2] = 10.0**((SNR - 2) / 10.0)
sigmas[:, 3] = 10.0**((SNR - 4) / 10.0)

nmethods = 2
cputime = np.zeros((len(dims), nSNR, nmethods))
recmat = np.zeros((len(dims), nSNR, nmethods))
mdmat = np.zeros((len(dims), nSNR, nmethods))

NRSIM = 250 #% increase this: in the paper this was 2000

for ii, N in enumerate(dims):
    print(f'\n*** N = {N}, M = {M}, L = {L}, K = {K} ****')

    for jj, snr in enumerate(SNR):
        sig = sigmas[jj, :]
        Lam = np.diag(np.sqrt(sig))

        print(f'\nStarting iterations for sig = {sig[0]:.2f}, SNR = {snr:.2f}',end="")
        np.random.seed(0)

        for it in range(NRSIM):
            # Generate sparse source S, E, A, Y
            S = Lam @ (np.random.randn(K, L) + 1j * np.random.randn(K, L)) / np.sqrt(2)           
            A = GaussianMM(N, M)
            E = (1 / np.sqrt(2)) * (np.random.randn(N, L) + 1j * np.random.randn(N, L))
            loc = np.sort(np.random.choice(M, K, replace=False))
            Y = A[:, loc] @ S + E

            # Method 1: CLOMP (proposed)
            start_time = time.time()
            Scl, _, _ = CLOMP(A, Y, K)
            cputime[ii, jj, 0] += time.time() - start_time
            mdmat[ii, jj, 0] += len(set(Scl) - set(loc))
            if set(Scl) == set(loc):
                recmat[ii, jj, 0] += 1

            # Method 2: SOMP
            start_time = time.time()
            _, Somp = SOMP(Y, A, K)
            cputime[ii, jj, 1] += time.time() - start_time
            mdmat[ii, jj, 1] += len(set(Somp) - set(loc))
            if set(Somp) == set(loc):
                recmat[ii, jj, 1] += 1
            
            if np.mod(it,25) == 0:
                print('.',end="")
    

recmat /= NRSIM
mdmat  /= (NRSIM*K)
cputime /= NRSIM
#%% Plotting the results
x = SNR

plt.figure(figsize=(16, 4))
plt.clf()

for i in range(4):
    plt.subplot(1, 4, i + 1)
    plt.plot(x, recmat[i, :, 0], 'ro-', label='CL-OMP', linewidth=0.8, markersize=12)
    plt.plot(x, recmat[i, :, 1], 'bx-.', label='SOMP', linewidth=0.8, markersize=12)
    plt.legend(fontsize=18)
    plt.ylabel('Exact recovery')
    plt.xlabel('SNR')
    plt.grid(True)
    plt.ylim([0, 1])

plt.show()

#%%
plt.figure(figsize=(16, 4))
plt.clf()

for i in range(4):
    plt.subplot(1, 4, i + 1)
    plt.plot(x, mdmat[i, :, 0], 'ro-', label='CL-OMP', linewidth=0.8, markersize=12)
    plt.plot(x, mdmat[i, :, 1], 'bx-.', label='SOMP', linewidth=0.8, markersize=12)
    plt.legend(fontsize=18)
    plt.ylabel('Probability of mis-detection')
    plt.xlabel('SNR')
    plt.grid(True)
    plt.ylim([0, 1])

plt.show()