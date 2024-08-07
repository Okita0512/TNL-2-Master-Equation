import numpy as np
from exc_basis import get_u_Omega_uv

"""
Two-level atom interacting with a collection of cavity modes, featured by photon echo.

Refs: 
[1] Phys. Rev. A 99, 063819 (2019)
[2] J. Chem. Phys. 151, 244113 (2019)
[3] J. Phys. Chem. Lett. 12, 3163-3170 (2021)
[4] Phys. Rev. A 101, 033831 (2020)

"""
def coth(x):
    return (1 + np.exp(-2 * x)) / (1 - np.exp(-2 * x))

def bathParam(L, ndof):     # for bath descritization

    wn = np.zeros((ndof), dtype = float)
    cn = np.zeros((ndof), dtype = float)
    for i in range(1, ndof + 1):
        wn[i - 1] = i * 137.036 * np.pi / L
        cn[i - 1] = wn[i - 1] * np.sqrt(8 * np.pi / L) * np.sin(i * 0.5 * np.pi)

    return cn, wn

def get_Hs():

    Hs = np.zeros((2,2))

    Hs[0, 0] = - 0.6738
    Hs[1, 1] = - 0.2798

    return Hs

def get_Qs(NStates):

    Qs = np.zeros((NStates, NStates), dtype = complex)
    Qs[0, 1] = 1.0
    Qs[1, 0] = Qs[0, 1]

    return -1.034 * Qs

def get_rho0(NStates):

    rho0 = np.zeros((NStates, NStates), dtype = complex)
    rho0[1, 1] = 1.0 + 0.0j

    return rho0

class parameters():

    # Model parameters
    NStates = 2      # number of states
    L = 236200
    ndof = 400
    β = np.inf

# ====================================================================================================

    # propagation
#    dt = 0.0025      # time step
    dt = 0.1
    t = 2100
    NSteps = int(t / dt)
    L_mem = NSteps          # full memory length
    nskip = 100

    # produce the Hamiltonian, initial RDM
    hams = get_Hs()
    rho0 = get_rho0(NStates)

    # bath and system-bath coupling parameters
    nmod = 1
    coeff = np.zeros((nmod, ndof), dtype = complex)
    C_ab = np.zeros((nmod, NSteps), dtype = complex)    # bare-bath TCFs
    for n in range(nmod):
        coeff[n, :], ω  = bathParam(L, ndof)
        for i in range(NSteps):
            C_ab[n, i] = np.sum((coeff[n, :]**2 / (2 * ω)) * (coth(β * ω / 2) * np.cos(ω * i * dt) - 1.0j * np.sin(ω * i * dt)))

    qmds = np.zeros((nmod, NStates, NStates), dtype = complex)
    qmds[0, :, :] = get_Qs(NStates)

    # featured for Redfield
    U, En, w_uv = get_u_Omega_uv(hams, NStates)
    qmds_ad = np.zeros((nmod, NStates, NStates), dtype = complex)
    for i in range(nmod):           
        qmds_ad[i, :, :] = U.T.conjugate() @ qmds[i, :, :] @ U

    # memory kernel construction
    K = np.zeros((L_mem, NStates, NStates, NStates, NStates), dtype = complex)
    for i in range(L_mem):
        tn = i * dt
        for n in range(nmod):
            TCF = C_ab[n, i]
            Sn = qmds_ad[n, :, :]
            K[i, :, :, :, :] += TCF * np.kron(Sn @ np.diag(np.exp(- 1.0j * En * tn)) @ Sn, np.diag(np.exp(1.0j * En * tn))).reshape(NStates, NStates, NStates, NStates)
            K[i, :, :, :, :] += TCF.conjugate() * np.kron(np.diag(np.exp(- 1.0j * En * tn)), Sn @ np.diag(np.exp(1.0j * En * tn)) @ Sn).reshape(NStates, NStates, NStates, NStates)
            K[i, :, :, :, :] -= TCF * np.kron(np.diag(np.exp(- 1.0j * En * tn)) @ Sn, np.diag(np.exp(1.0j * En * tn)) @ Sn).reshape(NStates, NStates, NStates, NStates)
            K[i, :, :, :, :] -= TCF.conjugate() * np.kron(Sn @ np.diag(np.exp(- 1.0j * En * tn)), Sn @ np.diag(np.exp(1.0j * En * tn))).reshape(NStates, NStates, NStates, NStates)

# plot the memory kernel
import matplotlib.pyplot as plt
K = parameters.K
NSteps = parameters.NSteps
plt.plot(K[:, 0, 0, 0, 0], label = r'$K_{0000}$')
plt.plot(K[:, 0, 0, 1, 1], label = r'$K_{0011}$')
plt.plot(K[:, 1, 1, 0, 0], label = r'$K_{1100}$')
plt.plot(K[:, 1, 1, 1, 1], label = r'$K_{1111}$')
plt.hlines([0.0], 0, NSteps, linestyles = ['--'], colors = ['black'])

plt.xlabel('time')
plt.title('Memory kernels')

plt.legend(frameon = False)
plt.show()

