import numpy as np
import scipy.linalg as la
import random as rd
from tqdm import tqdm
import matplotlib.pyplot as plt

hbar = 6.58211899e-16  # in eV
m0 = 9.10938215e-31    # in kg
e0 = 1.602176487e-19   # in Coulomb
etam = ((hbar**2) * e0 * (10**20.0)) / m0
a = 100.0              # in Angstrom
t = (1000.0 * etam) / (2 * (a**2.0) * 0.023) # Hopping parameter in meV
coupling = 0.15         # Band gap in meV
Length = 300           # Number of lattice sites (Length of chain = Length*a Angstrom) 
V0 = 1.2

def generate_disorder(V0, corr_order, dis_centers, Length):
    Vimps = np.zeros(Length)
    x = np.linspace(0, Length - 1, Length)
    for i in range(dis_centers):
        x0 = rd.randrange(0, Length)
        A0 = ((-1.0) ** round(rd.random())) * rd.random() #random disorder amplitude function for a disorder center
        Vimps += A0 * np.exp(-abs(x - x0) / (corr_order / 5.0)) #You might also change x-x0 to (x-x0)^n, n being a whole number
    #Normalizing disorder profile to make amplitude equal to user supplied value of V0
    Vimps = Vimps / np.mean((Vimps**2.0))
    Vimps -= np.mean(Vimps)
    Vimps = (Vimps / np.sqrt(np.mean(Vimps**2.0))) * V0
    return Vimps

def onsite_h(i, H, mu):
    ##Onsite Chemical and Disorder Potential terms
    H[i][i] = -mu + V[int(i/4)]
    H[i + 1][i + 1] = -H[i][i]
    return H

def hopping_h(i, H, d0):
    delta = d0
    if ((i < 2 * (Length - 1))):
        ##Hopping term
        H[i][i + 2] = H[i + 2][i] = -t/2
        H[i + 1][i + 3] = H[i + 3][i + 1] = t/2
        H[i][i + 3] = -delta/2
        H[i + 3][i] = delta/2
        H[i + 1][i + 2] = delta/2
        H[i + 2][i + 1] = -delta/2
    return H

def construct_Hamil(mu):
    H_0 = np.zeros((2* (Length), 2 * (Length)), dtype=float)
    Hamiltonian = []
    for i in range(0, 2 * (Length), 2):
        H_0 = onsite_h(i, H_0, mu)
        H_0 = hopping_h(i, H_0, coupling)
    Hamiltonian = np.array(H_0)
    return Hamiltonian

mus = np.linspace(-1.4*t,1.4*t,50)
V = generate_disorder(V0, corr_order=20.0, dis_centers=80, Length=Length)
energy = []
for m0 in tqdm(mus):
    Hamiltonian = construct_Hamil(m0)
    eneg, _ = la.eigh(Hamiltonian)
    energy.append(eneg)
energy = np.array(energy)


for i in range(len(energy[0])):
    plt.plot(mus, energy[:,i], 'k')
plt.ylim(-1.2*coupling, 1.2*coupling)
plt.xlabel(r"Chemical Potential $\mu$ (in meV)")
plt.ylabel(r"Energy $\epsilon$ (in meV)")
plt.title("Energy Spectrum")
plt.show()