import numpy as np
import os,shutil
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

    count = 0
    for k in range (N_ss,N_run):
        flip(lattice,L,T)
        m=np.sum(lattice) #magnetization(lattice,L)
        M+=m
        M_sq+=m**2
        e=energy_total(lattice,L)
        E+=e
        E_sq+=e**2

        fname = '{:.2f}i{}.npy'.format(round(T,2),count)
        np.save(os.path.join(train_dir,fname),lattice)
        count += 1
    
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
parser.add_argument('-fracN_ss', type=float,default=0.5,help='fraction of runs before sampling. default to 0.5')
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
lattice = spin_init(L,alinged=True)
N_ss = int(fracN_ss*N_run);   ### step at which sampling of data begins
T_range = np.arange(Tini,Tlast+DeltaT,DeltaT)
sample_size = (N_run-N_ss)*T_range.shape[0]
ctn = np.zeros((T_range.shape[0],5),dtype=np.float32)
Tc = 4.5
#######################


cwd = '/home/junkai/3D_v1.1'
os.chdir(cwd)
data_dir = os.path.join(cwd,'data' + strftime('%Y%m%d',localtime()))

if os.path.isdir(data_dir):
    shutil.rmtree(data_dir)
    os.mkdir(data_dir)
else:
    os.mkdir(data_dir)


# making train dir
train_dir = os.path.join(data_dir,'train')

if os.path.isdir(train_dir):
    shutil.rmtree(train_dir)
    os.mkdir(train_dir)
else:
    os.mkdir(train_dir)


# print informations
print(20*'=',' Sampling Summary','='*20)
print('Lattice shape = ',lattice.shape)
print('Temperature range = ',(Tini,Tlast),',at DeltaT = ',DeltaT)
print ('total run per temperature= ',N_run)
print ('runs before sampling = ',N_ss)
print ('sampling runs per temperature = ',N_run-int(fracN_ss*N_run))
print ('total number of temperature batchs = ',T_range.shape[0])
print ('total sample size = ',sample_size)
print(20*'=',' end of Summary ','='*20)
print('\n')


### begin calculation ##########
print(20*'=',' Sampling in progress ','='*20)
startime = localtime()
os.chdir(train_dir)

batch = 0
for T in T_range:
    print("T =",T)
    ctn[batch][0],ctn[batch][1],ctn[batch][2],ctn[batch][3],ctn[batch][4]=main(lattice,L,N_run,T)
    batch +=1


endtime = localtime()
duration = str(timedelta(seconds=mktime(endtime) - mktime(startime)))
print('code started on :',strftime('%x %X',startime),'\ncode ended on :',strftime('%x %X',endtime),'\ntime elapsed :',duration)
print(20*'=',' end of Sampling ','='*20)
print('\n')


############# generate train data ###################
os.chdir(data_dir)

file_list = []
for root,dirs,files in os.walk(train_dir):
    for file in files:
        file_list.append(file)


labels_temp = []
for i in file_list:
    ans = float(i.split('i')[0])
    labels_temp.append(ans)
    
len(labels_temp)
labels_temp = np.asarray(labels_temp)


labels = np.zeros_like(labels_temp)
for i in range(labels_temp.shape[0]):
    if labels_temp[i] < Tc:
        labels[i] = 1
    if labels_temp[i] > Tc:
        labels[i] = 0


print(20*'=',' Building train datasets ','='*20)

from sklearn.model_selection import train_test_split
test_size=0.5
x_train, x_test, y_train, y_test = train_test_split(file_list, labels, test_size=test_size, random_state=42)

fname = os.path.join(data_dir,'train_dataset.npz')
np.savez(fname , x_train=x_train,x_test=x_test,y_train=y_train,y_test=y_test)
print('dataset saved as: ', fname)
print('with labels: ',np.load(fname).files)
print('shape of x_train: ',np.shape(x_train))
print('shape of y_train: ',np.shape(y_train))
print('test split: ', test_size)
print('note: x is a list of referenced data filenames. \nFor training, please build pipeline that take referenced files into the NN.\n')


fname = os.path.join(data_dir,'run_data.txt')
np.savetxt(fname,ctn,delimiter=' ',header='L='+str(L)+' N_run='+str(N_run)+'; T M E C Chi')

print('run data saved as {} \n'.format(fname))
print(20*'=',' end of train dataset preparation ','='*20)
print('\n')












