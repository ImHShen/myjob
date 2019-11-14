from hapi import *
from numpy import arange
import matplotlib.pyplot as plt
w0 = 1000.
GammaD = 0.005
Gamma0 = 0.2
Gamma2 = 0.01 * Gamma0
Delta0 = 0.002
Delta2 = 0.001 * Delta0
nuVC = 0.2
eta = 0.5
Dw = 1.
ww = arange(w0-Dw, w0+Dw, 0.01) # GRID WITH THE STEP 0.01
l1 = PROFILE_HT(w0,GammaD,Gamma0,Gamma2,Delta0,Delta2,nuVC,eta,ww)[0]

plt.plot(ww,l1)
plt.show()

