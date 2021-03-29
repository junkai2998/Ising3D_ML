import numpy as np
import matplotlib.pyplot as plt
from time import localtime,mktime,strftime
from datetime import timedelta
import argparse

def spin_init(L,alinged=False):
    if alinged==False:
        lattice= np.random.choice((-1,1),size=(L,L,L)).astype(np.int32)

    if alinged==True:
        lattice=np.ones((L,L,L),dtype=np.int32)
    return lattice


# calculation of magnetization of the lattice
def magnetization(lattice,L):
    return np.sum(lattice)



def PBC(ind,L):
    """applies to index+1, to get the first element when index==L-1"""
    if ind == L-1:
        return 0
    else:
        return ind+1



def energy_site(lattice,L,i,j,k):
    '''
    calculation of interaction energy of each spin and total lattice energy
    '''
    #print ('we selected: ',lattice[i,j,k])
    #print (lattice[i-1,j,k] , lattice[PBC(i,L),j,k] , lattice[i,j-1,k] , lattice[i,PBC(j,L),k] , lattice[i,j,k-1] , lattice[i,j,PBC(k,L)]
    return lattice[i,j,k]*(lattice[i-1,j,k]+lattice[PBC(i,L),j,k]+lattice[i,j-1,k]+lattice[i,PBC(j,L),k]+lattice[i,j,k-1]+lattice[i,j,PBC(k,L)])



def energy_total(lattice,L):
    sum=0
    for i in range(L):
        for j in range(L):
            for k in range(L):
                sum-=energy_site(lattice,L,i,j,k) # -ve sign due to -J (ferromagnetic) in the hamitonian
    
    return sum/2 # correct for overcounting due to overlapping



# determine if a spin is flipped according to monte-carlo rules
def flip(lattice,L,T):
    for i in range(L):
        for j in range(L):
            for k in range(L):
                if energy_site(lattice,L,i,j,k)<0:
                    lattice[i][j][k]=-lattice[i][j][k]
                else:
                    if np.e**(-2*energy_site(lattice,L,i,j,k)/T)>np.random.uniform(0,1):
                        lattice[i][j][k]=-lattice[i][j][k]


from random import randint
def temp_label(T):
    global sampleid
    X[sampleid] = (lattice+1)/2
    X_temp[sampleid] = T
    if T < Tc:
        X_label[sampleid] = 1
    if T > Tc:
        X_label[sampleid] = 0
    if T > Tc:
        X_label[sampleid] = randint(0,1)
    sampleid += 1


# the main calculation routine to calculate the values
def main(lattice,L,N_run,T):
    M=0
    m=0
    M_sq=0
    E=0
    e=0
    E_sq=0
    
    ### stepping only lattice; no data sampling #####
    for k in range (N_ss):
        flip(lattice,L,T)
    
    run=0    
    for k in range (N_ss,N_run):
        flip(lattice,L,T)
        m=np.sum(lattice) #magnetization(lattice,L)
        M+=m
        M_sq+=m**2
        e=energy_total(lattice,L)
        E+=e
        E_sq+=e**2

        #sampleid = count*(N_run - int(fracN_ss*N_run)) + run
        #print(sampleid,count)
        temp_label(T)
        run += 1
    
    M_mean=M/(N_run-N_ss)
    M_var=M_sq/(N_run-N_ss)-M_mean**2
    E_mean=E/(N_run-N_ss)
    E_var=E_sq/(N_run-N_ss)-E_mean**2
    C=E_var/T**2/L**3
    Chi=M_var/T/L**3

    return T,M_mean/L**3,E_mean/L**3,C,Chi # temporary take out lattice, which is global


################### Command Line Interface ##################

parser = argparse.ArgumentParser(description='parameters for monte-carlo simulation of 2D square lattice Ising model')
parser.add_argument('-L', type=int,default=10,help='square lattice size. default = 10; 50 for production run')
parser.add_argument('-N_run', type=int,default=100,help='MC updates per temperature. default = 100; 2000 for production')
parser.add_argument('-fracN_ss', type=float,default=0.5,help='fraction of runs before sampling')
parser.add_argument('-Tini',type=float,default=0.0,help='start of simulation temperature range. default to 0.0')
parser.add_argument('-Tlast',type=float,default=6.0,help='end of simulation temperature range. default to 6.0')
parser.add_argument('-dt',type=float,default=0.1,help='time step. default 0.1; 0.05 for production run')

args = parser.parse_args()

################# parsing simulation parameters #######################
L = args.L
N_run = args.N_run
fracN_ss = args.fracN_ss
DeltaT = args.dt
Tini = args.Tini
Tlast = args.Tlast
################# end of parsing simulation parameters ###############


### initialization ####
N_ss = int(fracN_ss*N_run);   ### step at which sampling of data begins
T_range = np.arange(Tini,Tlast+DeltaT,DeltaT)
sample_size = (N_run-N_ss)*T_range.shape[0]
lattice=spin_init(L,alinged=True)

###### data arrays ##########
ctn = np.zeros((T_range.shape[0],5),dtype=np.float32)
X = np.zeros((sample_size,L,L,L),dtype=np.int32)
X_label = np.zeros((sample_size),dtype=np.int32)
X_temp = np.zeros((sample_size),dtype=np.float32)
Tc = 4.5

# print informations
print(20*'=',' Sampling Summary','='*20)
print('Lattice shape = ',lattice.shape)
print('Temperature range = ',(Tini,Tlast),',at DeltaT = ',DeltaT)
print ('total run per temperature= ',N_run)
print ('runs before sampling = ',N_ss)
print ('sampling runs per temperature = ',N_run-int(fracN_ss*N_run))
print ('total number of temperature batchs = ',T_range.shape[0])
print ('total sample size = ',sample_size)
print(20*'=','end of Summary ','='*20)
print('\n')


### begin calculation ##########
print(20*'=',' Sampling in progress ','='*20)
startime = localtime()

batch = 0
sampleid = 0
for T in T_range:
    print("T=",T)
    ctn[batch][0],ctn[batch][1],ctn[batch][2],ctn[batch][3],ctn[batch][4]=main(lattice,L,N_run,T)
    X[batch] = (lattice+1)/2
    batch +=1
    #lattice=spin_init(L,alinged=True)

endtime = localtime()
duration = str(timedelta(seconds=mktime(endtime) - mktime(startime)))
print('sampling started on :',strftime('%x %X',startime),'\nsampling completed on :',strftime('%x %X',endtime),'\ntime elapsed :',duration)
print(20*'=',' end of Sampling ','='*20)
print('\n')


print(20*'=',' Building test datasets ','='*20)

# no data shuffling
fname = '/home/junkai/3D_v1.0/' + strftime('%Y%m%d%H%M',startime) + 'test_dataset.npz'
np.savez(fname,x_train=X,x_temp=X_temp)
print('dataset saved : ', np.load(fname).files)

fname = '/home/junkai/3D_v1.0/' + strftime('%Y%m%d%H%M',startime) + 'run_gen_data.txt'
np.savetxt(fname,ctn,delimiter=' ',header='L='+str(L)+' N_run='+str(N_run)+'; T M E C Chi')

print('run gen data saved as {} \n'.format(fname))
print(20*'=',' end of test dataset preparation ','='*20)
print('\n')

