########## 3D Ising model via Monte-Carlo v1.0 #############

(A) Components:
1. 00sampling.ipynb: generate train dataset. Outputs:
            a) [datetime]run_data.txt that contains averaged thermodynamic quantities at temperatures
            b) [datetime]train_dataset.npz that contains the file names in train folder with labels
            *  runcli.py is the CLI implementation, ipynb for testing and debug only.
2. 01visualize_record.ipynb
3. 02visualize_lattice.ipynb
4. 03gen_test_data.ipynb.Outputs:
            a) [datetime]run_gen_data.txt that contains averaged thermodynamic quantities at temperatures
            b) [datetime]test_dataset.npz that contains the file names in train folder with labels
            *  rungencli.py is the CLI implementation, ipynb for testing and debug only.
5. 04training.ipynb
6. 05predict.ipynb


(B) Steps in the workflow are:
1. generate train data and test data. 
2. use 04train.ipynb to train.
3. use 05predict.ipynb to predict.
optional: use 01visualize_record.ipynb to check run data

The data structure for a complete workflow are:
        root
        |-----[datetime]train_dataset.npz
        |-----[datetime]run_data.txt
        |-----[datetime]test_dataset.npz
        |-----[datetime]run_gen_data.txt
        |-----[date]model.h5
    note: pls run all workflow in the same day to get same date in the filename.


(C) New Features:
1. More organized data directory. All data placed under root dir with datetime in the filename.
2. Clearer info prints during data generation.
3. AIO CLI for generate train and test data. with OOP and parallelism features.
4. 

(D) Known Issues:
1. Large single dataset file, takes longer time to load, as well as not suitable for very large datasets. (solved in v1.1 using data generator)
