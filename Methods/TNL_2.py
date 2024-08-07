import numpy as np

# ====== Redfield tensor plus the coherent term ======
def func(tn, rho_ad, par): 

    En = par.En
    dt = par.dt
    istep = int(tn / dt)
    L_mem = par.L_mem
    L0 = L_mem if (istep >= L_mem) else istep
    K = par.K

    # Simplified RK4 for Volterra integro-differential equation, a more rigorous one shall be found at [Numer. Math. 40, 119-135 (1982)]
    Y1 = rho_ad[istep, :, :]
    k1 = - 1.0j * (np.diag(En) @ Y1 - Y1 @ np.diag(En))
    for j in range(L0):
        k1 -= np.einsum('ijkl, kj', K[j, :, :, :, :], rho_ad[istep - j, :, :]) * dt
    Y2 = Y1 + 0.5 * dt * k1
    k2 = - 1.0j * (np.diag(En) @ Y2 - Y2 @ np.diag(En))
    for j in range(L0):
        k2 -= np.einsum('ijkl, kj', K[j, :, :, :, :], rho_ad[istep - j, :, :]) * dt
    Y3 = Y2 + 0.5 * dt * k2
    k3 = - 1.0j * (np.diag(En) @ Y3 - Y3 @ np.diag(En))
    for j in range(L0):
        k3 -= np.einsum('ijkl, kj', K[j, :, :, :, :], rho_ad[istep - j, :, :]) * dt
    Y4 = Y3 + 1.0 * dt * k3
    k4 = - 1.0j * (np.diag(En) @ Y4 - Y4 @ np.diag(En))
    for j in range(L0):
        k4 -= np.einsum('ijkl, kj', K[j, :, :, :, :], rho_ad[istep - j, :, :]) * dt

    K_rho = (k1 + 2 * k2 + 2 * k3 + k4) * (dt / 6)

    # Need to further elaborate on the actual RK4 version of the memory part... 
    # if secular approximation is on, then K only takes certain matrix elements? 
    
    return K_rho
