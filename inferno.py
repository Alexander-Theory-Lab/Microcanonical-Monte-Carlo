import numpy as np
import matplotlib.pyplot as plt

#methods for InfernoNetwork class
import networkx as nx
from random import sample
from itertools import count
import collections

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
        self.lattice = 2*np.random.randint(2, size=(N,N,N)) - 1
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
            self.E_lattice += cost
        # If it costs energy, only flip if demon has enough energy
        elif cost < self.E_demon:
            s *= -1
            self.E_demon -= cost
            self.E_lattice += cost
        # Otherwise, pass
        else:
            pass
        # Update the lattice site spin
        self.lattice[a, b] = s


class InfernoNetwork:
    """
        InfernoNetwork:
            - Implements microcanonical Monte Carlo simulation on a network
    """
    def __init__(self,N,dim=2,H=0):
        """
            :params:
                N - size of lattice
                dim - dimension of the lattice
                H - magnetic field strength
                lattice - state of lattice
                E_lattice - energy of lattice
                E_demon - energy of the demon
        """
        self.N = N
        self.dim = dim
        self.H = H
        self.G = self.generate_lattice(N,dim)
        self.E_demon = 0
        self.M = []


    def generate_lattice(self,N,dim):
        """
            Generates the initial periodic lattice with random spins
        """
        # Generate a blank periodic lattice as a network
        blank_network = nx.grid_graph([N]*dim,periodic=True)
        # For each node in network add a random spin -- up or down
        for node in list(blank_network.nodes):
            blank_network.node[node]['spin'] = 0.5 * (2*np.random.randint(2) - 1)
        return blank_network

    def calc_order_param(self):
        """
            Generates the initial periodic lattice with random spins
        """
        M_sum = 0
        # For each node in network add sum the spin
        for node in list(self.G.nodes):
            M_sum += self.G.node[node]['spin']
        self.M.append(M_sum)

    def demon_move(self):
        """
            Randomly move demon on network and try to flip spin
        """
        # Grab a random node from the graph
        rand_node = sample(list(self.G.nodes),1)[0]

        # Calculates the Energy required to flip a spin
        ## DE was inside for loop -- this is wrong!
        sum_spin, sum_spin_flip = 0, 0
        for neighbor in self.G.neighbors(rand_node):
            sum_spin += self.G.node[neighbor]['spin'] * self.G.node[rand_node]['spin'] + self.H * self.G.node[rand_node]['spin']
            sum_spin_flip += self.G.node[neighbor]['spin'] * (-1) * self.G.node[rand_node]['spin'] + self.H * (-1) * self.G.node[rand_node]['spin']
        DE = sum_spin_flip - sum_spin

        # If energy change is favorable flip
        if DE < 0:
            # Flip the spin
            self.G.node[rand_node]['spin'] = (-1) * self.G.node[rand_node]['spin']
            # Give the energy to the demon
            self.E_demon += np.absolute(DE)
            return True
        # If demon can afford to flip the spin, flip it
        elif self.E_demon >= DE:
            # Flip the spin
            self.G.node[rand_node]['spin'] = (-1) * self.G.node[rand_node]['spin']
            # Take energy from the demon
            self.E_demon -= np.absolute(DE)
            return True
        # The demon lacked the energy to flip, pass
        else:
            return False

    def simulate(self,N):
        """
            Simulate for N iterations
        """
        pos = nx.kamada_kawai_layout(self.G)
        plt.figure(figsize=(15,5))
        demon_hist = []
        for i in range(N):
            plt.clf()
            # Simulated annealing #
            if self.E_demon > 100:
                self.E_demon = 100
            #######################
            self.calc_order_param()
            demon_hist.append(self.E_demon)
            self.plot_stuff(pos,demon_hist,N)
            self.demon_move()
            #plt.savefig('/home/michael/demon/gif/imgage_' + str(i))
            plt.pause(0.01)
        plt.show()



    def plot_stuff(self,pos,demon_hist,N):
        """
            Handles basic visualization 
        """
        # Functional form of temperature
        B = lambda x: round((1./4) * np.log(1 + 4./np.mean(x)),2)
        round_sig = lambda f,p: float(('%.' + str(p) + 'e') % f)

        # get unique groups
        groups = set(nx.get_node_attributes(self.G,'spin').values())
        mapping = dict(zip(sorted(groups),count()))
        nodes = self.G.nodes()
        colors = [mapping[self.G.node[n]['spin']] for n in nodes]

        plt.subplot(1,3,1)
        # drawing nodes and edges separately so we can capture collection for colobar
        ec = nx.draw_networkx_edges(self.G, pos, alpha=0.2)
        nc = nx.draw_networkx_nodes(self.G, pos, nodelist=nodes, node_color=colors, 
                                    with_labels=False, node_size=100, cmap='viridis')

        plt.axis('off')
        plt.title(r'$\left< E_d \right>$ = ' + str(round_sig(np.mean(demon_hist),1)) + '\n' +  r'$\beta$ = ' + str(round_sig(B(demon_hist),1)))


        plt.subplot(1,3,2)


        degree_sequence = sorted(self.M, reverse=True)  # degree sequence
        degreeCount = collections.Counter(degree_sequence)
        deg, cnt = zip(*degreeCount.items())

        plt.bar(deg, cnt, width=0.80, edgecolor='black', color="r")
        #plt.ylabel('Number of Nodes')
        plt.xlabel('M')
        plt.title(r'$\left< M \right>$ = ' + str(round_sig(np.mean(self.M)/(self.N**self.dim),1)))


        plt.subplot(1,3,3)


        degree_sequence = sorted(demon_hist, reverse=True)  # degree sequence
        degreeCount = collections.Counter(degree_sequence)
        deg, cnt = zip(*degreeCount.items())

        plt.bar(deg, cnt, width=0.80, edgecolor='black', color="r")
        #plt.ylabel('Number of Nodes')
        plt.xlabel(r'$E_d$')
        plt.yscale('log')
        #plt.title(r'$\left< M \right>$ = ' + str(round_sig(np.mean(self.M)/(self.N**self.dim),1)))

        #plt.pause(0.01)


if __name__ == "__main__":
    # Choose a 7 x 7 periodic lattice 
    X = InfernoNetwork(5,dim=2,H=0.)
    # Simulate for 100 iterations
    X.simulate(1000)
