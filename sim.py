from inferno import Inferno
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binned_statistic

x = Inferno(10)

print("Init. Lattice energy: ",x.E_lattice)

demon_energy_history = []
demon_energy_mean_hist = []
demh_int = 0
binsz = 10
for i in range(100):
    demon_energy_history.append(x.E_demon)
    demh_int += x.E_demon
    if (i + 1)% binsz == 0:
        demon_energy_mean_hist.append(demh_int/binsz)
        demh_int = 0
    plt.imshow(x.lattice,cmap='binary')
    plt.title("Demon Energy: " + str(x.E_demon))
    plt.axis("off")
    plt.pause(0.02)
    x.demon_move()
plt.show()

demon_energy_mean_hist = np.array(demon_energy_mean_hist)

B = lambda x: (1./4) * np.log(1 + 4./np.mean(x))  #1./(0.25*np.log(1 + 4./np.mean(x)))
B_mean = lambda x: (1./4) * np.log(1 + 4./x)

print("Lattice energy: ",x.E_lattice)
print("Demon energy: ",x.E_demon)

print("Lattice Temp. ",B(demon_energy_history))

plt.subplot(1,2,1)
plt.plot(demon_energy_history)
plt.subplot(1,2,2)
plt.plot(B_mean(demon_energy_mean_hist))
plt.show()
