########## 3D Ising model via Monte-Carlo v1.1 #############

(A) Components:
1. sampling.ipynb : existing code for generate test data, for testing and debugging. Items 2,3,4,5 uses its main code on MC.

2. runcli.py : CLI code to generate train data. Will output:
            a) data[date]/train folder
            b) run_data.txt that contains averaged thermodynamic quantities at temperatures
            c) train_data.npz that contains the file names in train folder with labels
            
3. rungencli.py : CLI code to generate test data. Will output:
            a) data[date]/test folder
            b) run_gen_data.txt that contains averaged thermodynamic quantities at temperatures
            c) test_data.npz that contains the file names in test folder with labels
            

4. RUN.py and 5.mc_class.py : an AIO code that combines functionalities of runcli.py and rungencli.py. It features includes:
            a) Objec oriented implementation of lattice object. 
               mc_class.py is a library that contains Lattice_obj class, which is everything we need to do MC simulation on a spin lattice.
            b) Run temperature simulations in parallel. 
            *** 00sampling.ipynb dont have these features

note: 00sampling.ipynb is for testing and debugging.

6. 01visualize_record.ipynb

7. 02train.ipynb

8. 03predict.ipynb


(B) Steps in the workflow are:
1. generate train data and test data. 
2. use 02train.ipynb to train.
3. use 03predict.ipynb to predict.
optional: use 01visualize_record.ipynb to check run data

The data structure for a complete workflow are:
        data[date]
        |-----test
        |-----train
        |-----model.h5
        |-----run_data.txt
        |-----run_gen_data.txt
        |-----test_dataset.npz
        |-----train_dataset.npz
    note: pls make sure the train,predict code are using the same data[date] dir after the train,test data has generated.


(0) new features:
1. NN is pipelined with file generator.
2. More organized data directory.Train and test data are separated by folders, together with datasets that contains filenames and labels.
3. AIO CLI for generate train and test data. with OOP and parallelism features.
4. Clearer info prints during data generation.
5. one-go cwd assigment in the code, eliminating repeated assigment of cwd in the code (cause difficulty to modify code).


(C) known issues
1. parallel code refresh spins config every new temperature. this remains to think whether it is 'good'
2. critical temperature of 3D ising lattice with pbc is set to 4.5, which is remained to confirmed
3. Training give good acc metrics when the datasets starts with +ve magnitization at low temperature, but not the other way.
4. files are generated in a quite random order, so when creating file_list, we used sort()
5. The train npy is having (-1,1) instead of (0,1), but give good training and predicting results.