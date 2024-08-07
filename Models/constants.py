import numpy as np

"""
This file define the global variables, which can be called by:

    from constants import *

"""

# ==============================================================================================
#                                       Declaration
# ==============================================================================================

declaration = """
############################################################################
      2nd Order Perturbative Non-Markovian Redfield Master Equations
############################################################################
Authors: 
Wenxiang Ying, wying3@ur.rochester.edu

Last updated on 08/06/2024
############################################################################
"""

# ==============================================================================================
#                                   unit conversion to a.u.
# ==============================================================================================
"""
    definition: 

    e (electron charge)                 = 1
    me (electron mass)                  = 1
    hbar (reduced Planck's constant)    = 1
    kB (Boltzmann constant)             = 1
    kq (electrostatic constant)         = 1

"""

fs2au = 41.341374575751         # 1 fs       = 41.341374575751 a.u.         (time in femtosecond)
cm2au = 4.55633e-6              # 1 cm^-1    = 4.55633e-6 a.u.              (frequency in wavenumber)
THz2cm = 33.356                 # 1 THz      = 33.356 cm^-1                 (frequency in THz)
eV2au = 0.036749405469679       # 1 eV       = 0.036749405469679 a.u.       (electron volt)
K2au = 0.00000316678            # 1 K        = 0.00000316678 a.u.           (temperature in K)
Db2au = 0.393430                # 1 Debye    = 0.393430 a.u.                (dipole in Debye)
mm2au = 18897261.257078         # 1 mm       = 18897261.257078 a.u.         (length in mm)
c2au = 137.0359895              # c          = 137.036 a.u.                 (speed of light)
eps02au = 1 / (4 * np.pi)       # eps0       = 1 / (4 pi) a.u.              (vacuum permittivity)
kg2au = 1.097769e30             # 1 kg       = 1.097769e30 a.u.             (mass in kg)
kcal2au = 1.5936e-03            # 1 kcal/mol = 1.5936e-3 a.u.               (energy in kcal/mol)
J2au = 2.2937104e17             # 1 Joule    = 2.2937104e17 a.u.            (energy in Joule)

# e-p coupling unit to a.u. [J / (\sqrt(kg) * m)]
ep2au = J2au / (np.sqrt(kg2au) * 1000 * mm2au)
