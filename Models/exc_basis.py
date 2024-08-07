import numpy as np

"""
This file transforms the Hamiltonian to the exciton basis to facilitate the construction of the Redfield tensor

"""

# ====== getting the unitary transformation matrix and the energy gaps for Redfield ======
def get_u_Omega_uv(hams, NStates):   

    # get the eigen energies and eigen states of Hs and sort ascendingly
    EigVals, EigVecs = np.linalg.eig(hams)
    sortinds = np.argsort(EigVals)
    U = EigVecs[:,sortinds]
    En = EigVals[sortinds]

    # record the energy differences, i.e., present omega_ab
    w_uv = np.zeros((NStates, NStates), dtype = complex)
    for i in range(NStates):
        for j in range (NStates):
            w_uv[i, j] = En[i] - En[j]

    return U, En, w_uv
