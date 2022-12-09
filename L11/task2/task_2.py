import DE_alg_RRC
import matplotlib.pyplot as plt
import numpy as np

DT = 1E-3     # timestep of 1 millisecond
T_TO_CHRG = 0.1  # charge for ms
NO_OF_SAMPLES = int(T_TO_CHRG / DT)

R1 = 22E3
R2 = 10E5
C = 1E-6

t = np.linspace(0, T_TO_CHRG, NO_OF_SAMPLES)
Vc = DE_alg_RRC.fmodel(t, [R1, R2, C])

plt.plot(t, Vc)
plt.show()
