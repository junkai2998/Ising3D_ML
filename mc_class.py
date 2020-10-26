"""

main code of the module

this implements a lattice object, with method to simulate a monte-carlo spin flipping.

"""

import numpy as np


class Lattice_obj ():
    """
    A class of 3D cube lattice object.
    
    Attributes
    ----------------------
    L: lattice size
    alinged: alinged / random start. Default to False (random start)
    shape: (L,L,L)
    lattice: the lattice object itself
    
    Methods
    -----------------------
    
    """

    def __init__ (self,L,alinged=False):
        """
        initialize lattice class instance 
        
        parameters
        L: lattice size
        alinged: alinged: cold / hot start
        """
        self.L = L
        self.alinged = alinged
        self.shp = (self.L,self.L,self.L)
        self.spin_init()


    def spin_init(self):
        """
        Initialize spins configuration in the lattice. lattice shape is automatically passed as self.shp.
        The method also can be used to refresh spins configuration.
        """
        if self.alinged==False:
            self.lattice = np.random.choice(np.array([-1,1],dtype=np.int32),size=self.shp)

        if self.alinged==True:
            self.lattice = np.ones(shape=self.shp,dtype=np.int32)


    def magnetization(self):
        return np.sum(self.lattice)


    def pbc (self,ind):
        if ind == self.L-1:
            return 0
        else:
            return ind+1


    def energy_site(self,i,j,k):
        """calculate the interaction between site(i,j,k) with 6 (in 3D) neighbours"""
        return self.lattice[i,j,k]*(self.lattice[i-1,j,k]+self.lattice[self.pbc(i),j,k]+self.lattice[i,j-1,k]+self.lattice[i,self.pbc(j),k]+self.lattice[i,j,k-1]+self.lattice[i,j,self.pbc(k)])



    def energy_total(self):
        """calculate the total interaction energy in the lattice by heisenberg's equation. """
        sum=0
        for i in range(self.L):
            for j in range(self.L):
                for k in range(self.L):
                    sum-= self.energy_site(i,j,k) # -ve sign due to -J (ferromagnetic) in the hamitonian

        return sum/2 # correct for overcounting due to overlapping


    def flip (self,T):
        """determine if a spin is flipped according to metropolis monte-carlo rules, under a specified temperature T. """
        for i in range(self.L):
            for j in range(self.L):
                for k in range(self.L):
                    if self.energy_site(i,j,k)<0:
                        self.lattice[i][j][k]=-self.lattice[i][j][k]
                    else:
                        if np.e**(-2*self.energy_site(i,j,k)/T)>np.random.uniform(0,1):
                            self.lattice[i][j][k]=-self.lattice[i][j][k]


    def batch_flip (self,N_run,N_ss,T):
        """
        the main calculation routine to calculate the values
        
        parameters
        N_run: total number of complete monte-carlo steps (sweep through the lattice)
        N_ss: total number of complete MCS before sampling starts
        T: temperature
        """
        M=0
        m=0
        M_sq=0
        E=0
        e=0
        E_sq=0

        ### stepping only lattice; no data sampling #####
        for k in range (N_ss):
            self.flip(T)

        count = 0
        for k in range (N_ss,N_run):
            self.flip(T)
            m= self.magnetization()
            M+=m
            M_sq+=m**2
            e=self.energy_total()
            E+=e
            E_sq+=e**2

            fname = '{:.2f}i{}.npy'.format(round(T,2),count)
            np.save(fname,self.lattice)
            count += 1
        
        volume = self.L**3
        M_mean=M/(N_run-N_ss)
        M_var=M_sq/(N_run-N_ss)-M_mean**2
        E_mean=E/(N_run-N_ss)
        E_var=E_sq/(N_run-N_ss)-E_mean**2
        C=E_var/T**2/volume
        Chi=M_var/T/volume

        return T, M_mean/volume , E_mean/volume ,C , Chi # temporary take out lattice, which is global                            


