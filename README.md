# Second Order Time Convolution Quantum Master Equation (TNL-2) for Open Quantum System Dynamics

First released on 08/06/2024

Author:

Wenxiang Ying, wying3@ur.rochester.edu

This is a second order time convolution quantum master equation code (or 2nd order time-nonlocal equation, TNL-2 interchangeably) composed in python, a bunch of example model systems and results are also attached. To run the code, simply modify the "run.py" file with a method and a model desired (which can also be modified accordingly), and type "python3 run.py" at your terminal (can be your local machine).

Disclaimer: this code is far from complete and requires further optimizations. For example,

1, A better performed RK-4 integrator to solve the Volterra integro-differential equation remains to be further implemented, which can be found at [Numer. Math. 40, 119-135 (1982)]. 

2, The bath spectral density discretization scheme seems not very efficient to reach to convergence when constructing the memory kernels, especially for the Drude-Lorentz model. One might consider using continuous functions to represent the bath TCF, which avoids Poincare recurrences. 

3, This method is theoretically equivalent to the hierarchical equations of motion (HEOM) with only one tier, but numerical discrepancies between them are displayed, could be numerical convergence issues. Need to further explore it. 

If you are interested in further optimizing this code, or seeking for potential collaboration opportunities, please do not hesitate to contact me! 

Update on 08/07/2024 with a tutorial note pdf file. 
