import numpy as np
from exc_basis import get_u_Omega_uv

def coth(x):
    return (1 + np.exp(-2 * x)) / (1 - np.exp(-2 * x))

def bathParam(ωc, alpha, ndof):     # for bath descritization

    c = np.zeros(( ndof ))
    ω = np.zeros(( ndof ))
    for d in range(ndof):

        ω[d] =  - ωc * np.log(1 - (d + 1)/(ndof + 1))
        c[d] =  np.sqrt(alpha * ωc / (ndof + 1)) * ω[d]

    return c, ω

def get_Hs(ε, Δ):

    Hs = np.zeros((2,2))

    Hs[0, 0] = ε
    Hs[1, 1] = - ε
    Hs[0, 1] = Hs[1, 0] = Δ

    return Hs

def get_Qs(NStates):

    Qs = np.zeros((NStates, NStates), dtype = complex)
    Qs[0, 0] = 1.0
    Qs[1, 1] = - 1.0

    return Qs

def get_rho0(NStates):

    rho0 = np.zeros((NStates, NStates), dtype = complex)
    rho0[0, 0] = 1.0 + 0.0j

    return rho0

class parameters():

    # Model parameters
    NStates = 2      # number of states
    
# ======== Models are taken from [J. Chem. Phys. 151, 024105 (2019)] with original numbering ========

    ε, Δ, β, ωc, alpha, ndof, t = 0, 1, 0.25, 5, 0.02, 1000, 12     # Spin Boson Model 3
#    ε, Δ, β, ωc, alpha, ndof, t = 0, 1, 0.25, 1, 0.1, 1000, 12      # Spin Boson Model 4
#    ε, Δ, β, ωc, alpha, ndof, t = 0, 1, 0.25, 0.25, 0.4, 1000, 12   # Spin Boson Model 5
#    ε, Δ, β, ωc, alpha, ndof, t = 1, 1, 0.25, 1, 0.4, 1000, 12      # Spin Boson Model 6
#    ε, Δ, β, ωc, alpha, ndof, t = 1, 1, 5, 2, 0.4, 1000, 10         # Spin Boson Model 7
#    ε, Δ, β, ωc, alpha, ndof, t = 1, 1, 5, 2.5, 0.1, 1000, 15      # Spin Boson Model 8
#    ε, Δ, β, ωc, alpha, ndof, t = 1, 1, 5, 2.5, 0.2, 1000, 15       # Spin Boson Model 9
#    ε, Δ, β, ωc, alpha, ndof, t = 1, 1, 5, 2.5, 0.4, 1000, 15       # Spin Boson Model 10
#    ε, Δ, β, ωc, alpha, ndof, t = 1, 1, 10, 2.5, 0.2, 1000, 15      # Spin Boson Model 11

# ====================================================================================================

    # propagation
    dt = 0.0025             # integration time step
    NSteps = int(t / dt)    # number of steps
#    L_mem = 100             # finite memory length approximation
    L_mem = NSteps          # full memory length
    nskip = 10              # interval for data recording

    # produce the Hamiltonian, initial RDM
    hams = get_Hs(ε, Δ)
    rho0 = get_rho0(NStates)

    nmod = 1         # number of dissipation modes
    C_ab = np.zeros((nmod, NSteps), dtype = complex)    # bare-bath TCFs
    coeff = np.zeros((nmod, ndof), dtype = complex)
    for n in range(nmod):
        coeff[n, :], ω  = bathParam(ωc, alpha, ndof)
        for i in range(NSteps):
            C_ab[n, i] = 4 * 0.25 * np.sum((coeff[n, :]**2 / (2 * ω)) * (coth(β * ω / 2) * np.cos(ω * i * dt) - 1.0j * np.sin(ω * i * dt)))
    
    qmds = np.zeros((nmod, NStates, NStates), dtype = complex)
    qmds[0, :, :] = get_Qs(NStates)

    # featured for TNL-2
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

# ====================================================================================================
#                                   plot the memory kernel
# ====================================================================================================
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