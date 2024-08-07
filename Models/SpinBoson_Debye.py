import numpy as np
from exc_basis import get_u_Omega_uv

"""
Actually, the Redfield equation faces problems when discretizing the Debye bath, due to the ill behavior of the TCF
around t -> 0 (which is logarithmic divergent). As a result, the calculated coherence might face numerical issues. 
The continuous form of the bath spectral density might help? 

"""
def coth(x):
    return (1 + np.exp(-2 * x)) / (1 - np.exp(-2 * x))

def bathParam(λ, ωc, ndof):     # for bath descritization

    c = np.zeros(( ndof ))
    ω = np.zeros(( ndof ))
    for d in range(ndof):
        
        ω[d] =  ωc * np.tan(np.pi * (1 - (d + 1)/(ndof + 1)) / 2)
        c[d] =  np.sqrt(2 * λ / (ndof + 1)) * ω[d]

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

#    ε, Δ, β, ωc, λ, ndof, t = 0, 1, 0.5, 0.25, 0.025, 10000, 40  # Model 12
#    ε, Δ, β, ωc, λ, ndof, t = 0, 1, 0.5, 0.25, 0.25, 10000, 16   # Model 13
#    ε, Δ, β, ωc, λ, ndof, t = 0, 1, 5, 0.25, 0.25, 10000, 30     # Model 14, see also Fig.2a of [J. Chem. Phys. 143, 194108 (2015)]
#    ε, Δ, β, ωc, λ, ndof, t = 0, 1, 0.5, 5, 0.25, 10000, 12      # Model 15
#    ε, Δ, β, ωc, λ, ndof, t = 0, 1, 50, 5, 0.25, 10000, 30       # Model 16
#    ε, Δ, β, ωc, λ, ndof, t = 1, 1, 0.5, 0.25, 0.25, 10000, 30   # Model 17, see also Fig.2b of [J. Chem. Phys. 143, 194108 (2015)]
#    ε, Δ, β, ωc, λ, ndof, t = 1, 1, 0.5, 5, 0.25, 10000, 12      # Model 18
    ε, Δ, β, ωc, λ, ndof, t = 1, 1, 50, 5, 0.25, 10000, 30      # Model 19

# ====================================================================================================

    # propagation
    dt = 0.0025             # integration time step
    NSteps = int(t / dt)    # number of steps
#    L_mem = 1000             # finite memory length approximation
    L_mem = NSteps          # full memory length
    nskip = 10              # interval for data recording

    # produce the Hamiltonian, initial RDM
    hams = get_Hs(ε, Δ)
    rho0 = get_rho0(NStates)

    # bath and system-bath coupling parameters
    nmod = 1         # number of dissipation modes
    coeff = np.zeros((nmod, ndof), dtype = complex)     # system-bath coupling coefficients
    C_ab = np.zeros((nmod, NSteps), dtype = complex)    # bare-bath TCFs
    for n in range(nmod):
        coeff[n, :], ω  = bathParam(λ, ωc, ndof)        # maybe we shouldn't discretize.... very hard to converge. Use the continuous form. 
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