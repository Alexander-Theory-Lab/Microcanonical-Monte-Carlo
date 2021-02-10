from inferno import Inferno
import matplotlib.pyplot as plt

x = Inferno(6)

print("Init. Lattice energy: ",x.E_lattice)


for i in range(100):
    plt.imshow(x.lattice)
    plt.title("Demon Energy: " + str(x.E_demon))
    plt.axis("off")
    plt.pause(0.01)
    x.demon_move()
plt.show()

print("Lattice energy: ",x.E_lattice)
print("Demon energy: ",x.E_demon)
