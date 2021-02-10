import numpy as np
import matplotlib.pyplot as plt


class Inferno:
    """
        Inferno:
            - Main class for implementing microcanonical Monte Carlo simulation

        :instance methods:
            calc_E_lat - calculates the energy of a given latice configuration
            demon_move - updates the lattice by moving the demon around
    """

    def __init__(self,N):
        """
            :params:
                N - size of lattice
                lattice - state of lattice
                E_lattice - energy of lattice
                E_demon - energy of the demon
        """
        self.N = N
        self.lattice = 2*np.random.randint(2, size=(N,N)) - 1
        self.E_lattice = self.calc_E_lat(self.lattice, self.N)
        self.E_demon = 0
        self.E_total = self.E_lattice + self.E_demon


    def calc_E_lat(self,lattice,N):
        """
            Calculate energy of the lattice configuration.
        """
        ETOT = 0

        # Loop over the entire lattice calculating nearest neighbor interactions
        # TODO implement in a more intelligent way especially if sims seem to slow
        for a in range(N):
            for b in range(N):
                # Grab the lattice site spin value
                s =  lattice[a, b]
                # Calculate the energy of the configuration based on 
                # nearest neighbors
                nb = lattice[(a+1)%N,b] + lattice[a,(b+1)%N] + lattice[(a-1)%N,b] + lattice[a,(b-1)%N]
                # running sum of energy of Ising latus
                ETOT += 2 * s * nb

        # Update the value of the lattice energy
        return ETOT

    def demon_move(self):
        """
            Randomly move the demon around and flip spins.
        """
        # Choose a random lattice site       
        a = np.random.randint(0, self.N)
        b = np.random.randint(0, self.N)
        # Grab the lattice site spin value
        s =  self.lattice[a, b]
        # Calculate the energy of the configuration based on 
        # nearest neighbors
        nb = self.lattice[(a+1)%self.N,b] + self.lattice[a,(b+1)%self.N] + self.lattice[(a-1)%self.N,b] + self.lattice[a,(b-1)%self.N]
        # Check the cost of flipping the spin    
        cost = 2*s*nb
        # If energetically favorable, flip and add energy to demon
        if cost < 0:
            s *= -1
            # Notice we substract the cost to maintain net0 energy
            self.E_demon -= cost
        # If it costs energy, only flip if demon has enough energy
        elif cost < self.E_demon:
            s *= -1
            self.E_demon -= cost
        # Otherwise, pass
        else:
            pass
        # Update the lattice site spin
        self.lattice[a, b] = s
